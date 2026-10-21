import { StageEvent, WorkerStageId } from '../../core/models/pipeline-stages';
import {
    CAMERA_STRIDE,
    ErrorSummary,
    ScanMetrics,
    StageTiming,
} from '../../core/models/scan-results';
import { Intrinsics, cameraCenter, fromCamera, scaleIntrinsics } from '../geometry/camera';
import { GpuContext } from '../gpu/gpu-device';
import { PointAccumulator, PointCloud } from '../fusion/point-accumulator';
import { Mesh, extractSurface, removeSmallComponents } from '../fusion/surface-nets';
import { BRICK, BrickGrid, TsdfVolume } from '../fusion/tsdf-volume';
import { median, quantile } from '../math/dense';
import { applyTransposed3, norm3, sub3 } from '../math/mat3';
import { DenseStereo } from '../mvs/dense-stereo';
import { depthRange, selectSources } from '../mvs/view-selection';
import { bundleAdjust } from '../sfm/bundle-adjuster';
import { TexturedMesh, bakeAtlas } from '../texture/atlas-baker';
import { faceCandidates } from '../texture/face-visibility';
import { absorbSmallCharts, faceAdjacency, labelFaces } from '../texture/view-labeling';
import { DepthGrid, ScanSession, StoredKeyframe } from './scan-session';

export interface Reconstruction {
    cameras: Float32Array;
    sparse: { positions: Float32Array; colors: Uint8Array };
    dense: PointCloud;
    mesh: TexturedMesh;
    atlasJpeg: Uint8Array;
    metrics: ScanMetrics;
}

/** Receives intermediate results so the viewer can show them while the next stage runs. */
export interface ReconstructionSink {
    stage(event: StageEvent): void;
    cameras(cameras: Float32Array): void;
    sparse(positions: Float32Array, colors: Uint8Array): void;
    dense(cloud: PointCloud): void;
    mesh(mesh: Mesh): void;
}

const BUNDLE_POINTS = 8000;
const MVS_SOURCES = 5;
const FILTER_SOURCES = 8;
const PHOTOMETRIC_ITERATIONS = 5;
const GEOMETRIC_ITERATIONS = 2;
/** Caps the surface resolution so meshes stay around ≤ 1 M triangles on a phone. */
const DENSE_CELL_BUDGET = 500_000;
/** 16 k bricks × 512 voxels × 8 B = 64 MB of GPU memory (and the same once read back). */
const BRICK_BUDGET = 16_000;
const GRID_CELL_BUDGET = 8_000_000;
const MIN_WEIGHT = 0.6;
const ATLAS_MAX = 4096;

/**
 * Offline stage, run once recording stops:
 *   bundle adjustment → PatchMatch stereo (photometric, then geometric) → consistency filter →
 *   dense cloud → TSDF fusion → surface nets → view selection + texture atlas.
 */
export async function reconstruct(
    session: ScanSession,
    gpu: GpuContext,
    sink: ReconstructionSink,
): Promise<Reconstruction> {
    const { map, keyframes } = session;
    if (keyframes.length < 3)
        throw new Error('record a little longer: at least 3 keyframes are needed');
    const timings: StageTiming[] = [];
    const report = (stage: WorkerStageId, fraction: number, detail: string | null = null) =>
        sink.stage({ stage, status: 'running', fraction, detail, ms: null });
    const timed = async <T>(stage: WorkerStageId, work: () => Promise<T> | T): Promise<T> => {
        report(stage, -1);
        const start = performance.now();
        const result = await work();
        if (gpu.lost) {
            throw new Error(
                `the phone reset the GPU during ${stage} (${gpu.lost}) — close other apps and try a shorter scan`,
            );
        }
        const ms = performance.now() - start;
        timings.push({ name: stage, ms });
        sink.stage({ stage, status: 'done', fraction: 1, detail: null, ms });
        return result;
    };
    const device = gpu.device;

    // 1. Bundle adjustment with the phone odometry as priors.
    const before = summarize(map.reprojectionErrors());
    let bundleIterations = 0;
    await timed('bundle', () => {
        const all = map.tracks.map((_, id) => id);
        for (const [iterations, limit] of [
            [15, 4],
            [10, 2.5],
        ]) {
            const { problem } = map.bundleProblem(BUNDLE_POINTS);
            if (problem.points.length >= 30 * 3) {
                bundleIterations += bundleAdjust(problem, iterations).iterations;
            }
            map.triangulate(all, 4);
            map.pruneObservations(limit);
            map.triangulate(all, 3);
        }
    });
    const after = summarize(map.reprojectionErrors());
    const cameras = cameraArray(session);
    sink.cameras(cameras);
    const sparse = map.points(2);
    sink.sparse(sparse.positions, sparse.colors);

    // 2. Dense stereo on the GPU.
    const { width, height } = keyframes[0].work;
    const views = map.frames.map((frame) => ({ intrinsics: frame.intrinsics, pose: frame.pose }));
    const stereo = new DenseStereo(
        device,
        width,
        height,
        views,
        keyframes.map((k) => k.work.gray),
    );
    // Phone depth at stereo resolution, made on demand (keeping all of them costs ~1 MB a view).
    const priorOf = (i: number) => {
        const depth = keyframes[i].depth;
        return depth ? upsampleDepth(depth, width, height) : null;
    };
    const sources = keyframes.map((_, i) => selectSources(map, i, MVS_SOURCES));
    const checks = keyframes.map((_, i) => selectSources(map, i, FILTER_SOURCES));
    const ranges = keyframes.map((k, i) => depthRange(map, i, k.depth?.meters ?? null));
    const total = keyframes.length;
    await timed('stereo', async () => {
        for (let i = 0; i < total; i++) {
            report('stereo', i / total, `${i + 1}/${total}`);
            await stereo.solve(i, {
                sources: sources[i],
                minDepth: ranges[i][0],
                maxDepth: ranges[i][1],
                prior: priorOf(i),
                geometric: false,
                iterations: PHOTOMETRIC_ITERATIONS,
            });
        }
    });
    await timed('geometric', async () => {
        for (let i = 0; i < total; i++) {
            report('geometric', i / total, `${i + 1}/${total}`);
            await stereo.solve(i, {
                sources: sources[i],
                minDepth: ranges[i][0],
                maxDepth: ranges[i][1],
                prior: null,
                geometric: true,
                iterations: GEOMETRIC_ITERATIONS,
            });
        }
    });

    // 3. Consistency filter → dense cloud; its cell size becomes the voxel size.
    const footprints = keyframes.map(
        (k, i) => (ranges[i][0] + ranges[i][1]) / 2 / k.work.intrinsics.fx,
    );
    const depthsSeen = map.frames.map((frame) => frame.depthHint / frame.intrinsics.fx);
    const accumulator = new PointAccumulator(
        Math.min(0.05, Math.max(0.0008, 1.6 * median(depthsSeen.length ? depthsSeen : footprints))),
    );
    await timed('filter', async () => {
        for (let i = 0; i < total; i++) {
            report('filter', i / total, `${i + 1}/${total}`);
            const packed = await stereo.filteredMap(i, checks[i]);
            addDepthMap(accumulator, packed, keyframes[i], map.frames[i].pose);
            while (accumulator.size > DENSE_CELL_BUDGET) accumulator.coarsen();
        }
    });
    const dense = accumulator.cloud();
    sink.dense(dense);
    if (dense.positions.length / 3 < 100) {
        stereo.destroy();
        throw new Error(
            'dense stereo found almost no surface — scan a textured scene with more motion',
        );
    }

    // 4. TSDF fusion and meshing.
    let mesh: Mesh = {
        positions: new Float32Array(0),
        normals: new Float32Array(0),
        indices: new Uint32Array(0),
    };
    let voxel = accumulator.cell;
    await timed('mesh', async () => {
        report('mesh', -1, 'allocating volume');
        const bounds = robustBounds(dense.positions);
        let grid: BrickGrid;
        let layout: ReturnType<BrickGrid['build']>;
        for (;;) {
            const margin = voxel * BRICK * 2;
            const origin: [number, number, number] = [
                bounds.min[0] - margin,
                bounds.min[1] - margin,
                bounds.min[2] - margin,
            ];
            const extent: [number, number, number] = [
                bounds.max[0] - bounds.min[0] + 2 * margin,
                bounds.max[1] - bounds.min[1] + 2 * margin,
                bounds.max[2] - bounds.min[2] + 2 * margin,
            ];
            grid = new BrickGrid(origin, voxel, extent);
            if (grid.cells > GRID_CELL_BUDGET) {
                voxel *= 1.4;
                continue;
            }
            markSurface(grid, dense.positions, voxel * 4);
            markPhoneDepth(grid, session, voxel * 4);
            layout = grid.build();
            if (layout.count <= BRICK_BUDGET) break;
            voxel *= 1.3;
        }
        const tsdf = new TsdfVolume(device, grid, layout, stereo.filtered, width, height);
        for (let i = 0; i < total; i++) {
            report('mesh', (0.8 * i) / total, `integrating ${i + 1}/${total}`);
            await stereo.filter(i, checks[i]);
            await tsdf.integrate({
                intrinsics: map.frames[i].intrinsics,
                pose: map.frames[i].pose,
                maxDepth: ranges[i][1],
                prior: priorOf(i),
            });
        }
        report('mesh', 0.85, 'extracting surface');
        const volume = await tsdf.read();
        tsdf.destroy();
        stereo.destroy();
        mesh = extractSurface(
            { origin: grid.origin, voxel, dims: grid.dims, lookup: layout.lookup, ...volume },
            MIN_WEIGHT,
        );
        mesh = removeSmallComponents(mesh, 0.01, 200);
    });
    if (mesh.indices.length === 0) throw new Error('fusion produced an empty surface');
    sink.mesh(mesh);

    // 5. Texture: best view per face, then the atlas from the sharpest, most frontal views.
    const texture = keyframes[0].texture;
    const focus = Math.max(...keyframes.map((k) => k.sharpness));
    const textureViews = keyframes.map((k, i) => ({
        intrinsics: k.texture.intrinsics,
        pose: map.frames[i].pose,
        width: k.texture.width,
        height: k.texture.height,
        thumb: k.thumb,
        quality: Math.sqrt(Math.max(1e-6, k.sharpness / focus)),
    }));
    const selection = await timed('views', async () => {
        const far = farthestDistance(mesh.positions, cameras) * 1.2 + 0.1;
        // Visibility only needs the z-buffer at face centroids: half resolution is plenty and
        // a quarter of the GPU work per view (the scores scale equally for every view).
        const zWidth = Math.round(texture.width / 2);
        const zHeight = Math.round(texture.height / 2);
        const candidates = await faceCandidates(
            device,
            mesh.positions,
            mesh.indices,
            textureViews.map((view) => ({
                ...view,
                intrinsics: scaleIntrinsics(
                    view.intrinsics,
                    zWidth / view.width,
                    zHeight / view.height,
                ),
            })),
            zWidth,
            zHeight,
            0.02,
            far,
            (fraction) => report('views', fraction * 0.9),
        );
        report('views', 0.9, 'growing charts');
        const adjacency = faceAdjacency(mesh.indices, mesh.positions.length / 3);
        const labels = labelFaces(candidates, adjacency);
        absorbSmallCharts(labels, candidates, adjacency, 12);
        absorbSmallCharts(labels, candidates, adjacency, 12);
        return { candidates, adjacency, labels };
    });
    const { textured, atlasJpeg } = await timed('texture', async () => {
        const baked = await bakeAtlas(
            mesh.positions,
            mesh.indices,
            selection.labels,
            selection.candidates,
            selection.adjacency,
            textureViews,
            (view) => session.images.get(view),
            ATLAS_MAX,
            (fraction) => report('texture', fraction * 0.9),
        );
        report('texture', 0.9, 'encoding JPEG');
        const jpeg = await session.images.jpeg(
            { width: baked.atlas.width, height: baked.atlas.height, data: baked.atlas.pixels },
            0.9,
        );
        return { textured: baked, atlasJpeg: jpeg };
    });

    const area = surfaceArea(mesh);
    const extent = meshExtent(mesh.positions);
    const centers = map.frames.map((frame) => cameraCenter(frame.pose));
    const shifts = map.frames.map((frame, i) => norm3(sub3(centers[i], cameraCenter(frame.prior))));
    let length = 0;
    for (let i = 1; i < centers.length; i++) length += norm3(sub3(centers[i], centers[i - 1]));
    const metrics: ScanMetrics = {
        gpu: gpu.label,
        keyframes: keyframes.length,
        blurred: session.rejected,
        depthSensor: keyframes.some((k) => k.depth !== null),
        workSize: [width, height],
        textureSize: [keyframes[0].texture.width, keyframes[0].texture.height],
        tracks: map.tracks.filter((track) => track.alive && track.point).length,
        sparsePoints: sparse.positions.length / 3,
        observations: map.reprojectionErrors().length,
        reprojection: { before, after },
        poseShift: {
            mean: shifts.reduce((sum, value) => sum + value, 0) / shifts.length,
            max: Math.max(...shifts),
        },
        bundleIterations,
        trajectory: { length, closure: norm3(sub3(centers[centers.length - 1], centers[0])) },
        densePoints: dense.positions.length / 3,
        voxel,
        meshVertices: textured.positions.length / 3,
        meshTriangles: textured.indices.length / 3,
        surfaceArea: area,
        density: area > 0 ? dense.positions.length / 3 / area : 0,
        extent,
        atlas: [textured.atlas.width, textured.atlas.height],
        charts: textured.charts,
        untextured: textured.untextured,
        timings,
    };
    return { cameras, sparse, dense, mesh: textured, atlasJpeg, metrics };
}

function summarize(errors: Float64Array): ErrorSummary {
    if (errors.length === 0) return { mean: 0, median: 0 };
    let sum = 0;
    for (const e of errors) sum += e;
    return { mean: sum / errors.length, median: median(errors) };
}

function cameraArray(session: ScanSession): Float32Array {
    const out = new Float32Array(session.keyframes.length * CAMERA_STRIDE);
    session.map.frames.forEach((frame, i) => {
        const c = cameraCenter(frame.pose);
        const k = frame.intrinsics;
        out.set(frame.pose.R, i * CAMERA_STRIDE);
        out.set(c, i * CAMERA_STRIDE + 9);
        out[i * CAMERA_STRIDE + 12] = frame.width / 2 / k.fx;
        out[i * CAMERA_STRIDE + 13] = frame.height / 2 / k.fy;
    });
    return out;
}

/** Bilinear upsampling of the phone depth grid, ignoring cells without data. */
export function upsampleDepth(grid: DepthGrid, width: number, height: number): Float32Array {
    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        const gy = Math.min(grid.height - 1, Math.max(0, ((y + 0.5) / height) * grid.height - 0.5));
        const y0 = Math.floor(gy);
        const y1 = Math.min(grid.height - 1, y0 + 1);
        const wy = gy - y0;
        for (let x = 0; x < width; x++) {
            const gx = Math.min(
                grid.width - 1,
                Math.max(0, ((x + 0.5) / width) * grid.width - 0.5),
            );
            const x0 = Math.floor(gx);
            const x1 = Math.min(grid.width - 1, x0 + 1);
            const wx = gx - x0;
            const samples = [
                [grid.meters[y0 * grid.width + x0], (1 - wx) * (1 - wy)],
                [grid.meters[y0 * grid.width + x1], wx * (1 - wy)],
                [grid.meters[y1 * grid.width + x0], (1 - wx) * wy],
                [grid.meters[y1 * grid.width + x1], wx * wy],
            ];
            let sum = 0;
            let weight = 0;
            for (const [d, w] of samples) {
                if (d > 0) {
                    sum += d * w;
                    weight += w;
                }
            }
            out[y * width + x] = weight > 0.5 ? sum / weight : 0;
        }
    }
    return out;
}

const halfTable = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
    const exponent = (h >> 10) & 31;
    const mantissa = h & 1023;
    const sign = h & 0x8000 ? -1 : 1;
    halfTable[h] =
        exponent === 0
            ? sign * Math.pow(2, -14) * (mantissa / 1024)
            : exponent === 31
              ? NaN
              : sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

/** Decodes the packed GPU depth map of one view into world-space samples. */
function addDepthMap(
    accumulator: PointAccumulator,
    packed: Uint32Array,
    keyframe: StoredKeyframe,
    pose: StoredKeyframe['prior'],
): void {
    const { width, height, intrinsics: k } = keyframe.work;
    const thumb = keyframe.thumb;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = packed[y * width + x];
            if (value === 0) continue;
            const d = halfTable[value & 0xffff];
            if (!(d > 0)) continue;
            const ex = (((value >>> 16) & 63) / 63) * 2 - 1;
            const ey = (((value >>> 22) & 63) / 63) * 2 - 1;
            let nx = ex;
            let ny = ey;
            const nz = 1 - Math.abs(ex) - Math.abs(ey);
            if (nz < 0) {
                nx = (1 - Math.abs(ey)) * (ex >= 0 ? 1 : -1);
                ny = (1 - Math.abs(ex)) * (ey >= 0 ? 1 : -1);
            }
            const world = fromCamera(pose, [((x - k.cx) / k.fx) * d, ((y - k.cy) / k.fy) * d, d]);
            const normal = applyTransposed3(pose.R, [nx, ny, nz]);
            const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
            const tx = Math.min(thumb.width - 1, x >> 1);
            const ty = Math.min(thumb.height - 1, y >> 1);
            const i = (ty * thumb.width + tx) * 3;
            accumulator.add(
                world[0],
                world[1],
                world[2],
                normal[0] / length,
                normal[1] / length,
                normal[2] / length,
                thumb.rgb[i],
                thumb.rgb[i + 1],
                thumb.rgb[i + 2],
            );
        }
    }
}

function robustBounds(positions: Float32Array): { min: number[]; max: number[] } {
    const min: number[] = [];
    const max: number[] = [];
    const count = positions.length / 3;
    const stride = Math.max(1, Math.floor(count / 50000));
    for (let a = 0; a < 3; a++) {
        const values: number[] = [];
        for (let i = 0; i < count; i += stride) values.push(positions[i * 3 + a]);
        const low = quantile(values, 0.002);
        const high = quantile(values, 0.998);
        const pad = (high - low) * 0.02;
        min.push(low - pad);
        max.push(high + pad);
    }
    return { min, max };
}

function markSurface(grid: BrickGrid, positions: Float32Array, radius: number): void {
    for (let i = 0; i < positions.length; i += 3) {
        grid.mark(positions[i], positions[i + 1], positions[i + 2], radius);
    }
}

/** Also allocates bricks where only the phone's depth saw a surface (e.g. blank walls). */
function markPhoneDepth(grid: BrickGrid, session: ScanSession, radius: number): void {
    session.keyframes.forEach((keyframe, index) => {
        const depth = keyframe.depth;
        if (!depth) return;
        const pose = session.map.frames[index].pose;
        const k: Intrinsics = scaleIntrinsics(
            keyframe.work.intrinsics,
            depth.width / keyframe.work.width,
            depth.height / keyframe.work.height,
        );
        for (let y = 0; y < depth.height; y += 2) {
            for (let x = 0; x < depth.width; x += 2) {
                const d = depth.meters[y * depth.width + x];
                if (!(d > 0.05)) continue;
                const p = fromCamera(pose, [((x - k.cx) / k.fx) * d, ((y - k.cy) / k.fy) * d, d]);
                grid.mark(p[0], p[1], p[2], radius);
            }
        }
    });
}

function farthestDistance(positions: Float32Array, cameras: Float32Array): number {
    let far = 0;
    const count = cameras.length / CAMERA_STRIDE;
    const stride = Math.max(1, Math.floor(positions.length / 3 / 4000));
    for (let c = 0; c < count; c++) {
        const cx = cameras[c * CAMERA_STRIDE + 9];
        const cy = cameras[c * CAMERA_STRIDE + 10];
        const cz = cameras[c * CAMERA_STRIDE + 11];
        for (let i = 0; i < positions.length / 3; i += stride) {
            far = Math.max(
                far,
                Math.hypot(
                    positions[i * 3] - cx,
                    positions[i * 3 + 1] - cy,
                    positions[i * 3 + 2] - cz,
                ),
            );
        }
    }
    return far;
}

function surfaceArea(mesh: Mesh): number {
    const p = mesh.positions;
    let area = 0;
    for (let f = 0; f < mesh.indices.length; f += 3) {
        const a = mesh.indices[f] * 3;
        const b = mesh.indices[f + 1] * 3;
        const c = mesh.indices[f + 2] * 3;
        const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
        const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
        area +=
            0.5 *
            Math.hypot(
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            );
    }
    return area;
}

function meshExtent(positions: Float32Array): [number, number, number] {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let a = 0; a < 3; a++) {
            min[a] = Math.min(min[a], positions[i + a]);
            max[a] = Math.max(max[a], positions[i + a]);
        }
    }
    return [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
}
