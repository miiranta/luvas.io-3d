/**
 * Numerical check of the whole reconstruction pipeline on the synthetic scene, run inside
 * headless Chrome (WebGPU) by tools/verify.mjs. Keyframes are rendered from a two-ring orbit,
 * the "phone" poses get a slow drift, then capture + reconstruction run exactly as in the app
 * and the outputs are scored against the analytic ground truth.
 */
import { cameraCenter, scaleIntrinsics } from '../src/app/vision/geometry/camera';
import { GpuContext, requestGpu } from '../src/app/vision/gpu/gpu-device';
import { rgbaToGray } from '../src/app/vision/image/gray-image';
import { DenseStereo } from '../src/app/vision/mvs/dense-stereo';
import { seededRandom, symmetricEigen } from '../src/app/vision/math/dense';
import { JpegImageStore } from '../src/app/vision/pipeline/jpeg-image-store';
import { reconstruct } from '../src/app/vision/pipeline/reconstructor';
import { ScanSession } from '../src/app/vision/pipeline/scan-session';
import { CameraState, SceneRenderer, depthGrid, lookAt, xrMatrix } from './scene-renderer';
import {
    BOX_CENTER,
    BOX_HALF,
    BOX_SIZE,
    boxDistance,
    surfaceColor,
    surfaceDistance,
    surfaceSamples,
} from './synthetic-scene';

interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

declare global {
    interface Window {
        __verify?: { checks: Check[]; log: string[]; error?: string };
    }
}

const WIDTH = 1280;
const HEIGHT = 720;
const K = { fx: 1000, fy: 1000, cx: 639.5, cy: 359.5 };
const RINGS = [
    { height: 0.2, radius: 0.42, count: 36 },
    { height: 0.36, radius: 0.36, count: 24 },
];

/**
 * Phone odometry: a world-frame drift D_i = (yaw i·0.25 mrad, translation i·0.3 mm) applied to
 * the true camera, plus 0.5 mm of jitter. Prior camera → world = D_i ∘ truth.
 */
function drifted(pose: { R: number[]; center: number[] }, index: number, random: () => number) {
    const yaw = index * 0.00025;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const D = [c, 0, s, 0, 1, 0, -s, 0, c];
    // World → camera rotation R·Dᵀ.
    const R = [0, 1, 2].flatMap((r) =>
        [0, 1, 2].map(
            (col) =>
                pose.R[r * 3] * D[col * 3] +
                pose.R[r * 3 + 1] * D[col * 3 + 1] +
                pose.R[r * 3 + 2] * D[col * 3 + 2],
        ),
    );
    const offset = [index * 0.0003, -index * 0.00015, index * 0.0002];
    const center = [0, 1, 2].map(
        (k) =>
            D[k * 3] * pose.center[0] +
            D[k * 3 + 1] * pose.center[1] +
            D[k * 3 + 2] * pose.center[2] +
            offset[k] +
            (random() - 0.5) * 0.001,
    );
    return { R, center };
}

/**
 * Rigid transform (Horn's quaternion method) taking the reconstructed camera centres onto the
 * true ones, so that surface errors are measured without the arbitrary gauge of the odometry.
 */
function alignment(from: number[][], to: number[][]): (p: number[]) => number[] {
    const mean = (points: number[][]) =>
        [0, 1, 2].map((k) => points.reduce((sum, p) => sum + p[k], 0) / points.length);
    const a = mean(from);
    const b = mean(to);
    const S = new Array(9).fill(0);
    from.forEach((p, i) => {
        for (let r = 0; r < 3; r++)
            for (let c = 0; c < 3; c++) S[r * 3 + c] += (p[r] - a[r]) * (to[i][c] - b[c]);
    });
    const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
    const N = [
        xx + yy + zz,
        yz - zy,
        zx - xz,
        xy - yx,
        yz - zy,
        xx - yy - zz,
        xy + yx,
        zx + xz,
        zx - xz,
        xy + yx,
        -xx + yy - zz,
        yz + zy,
        xy - yx,
        zx + xz,
        yz + zy,
        -xx - yy + zz,
    ];
    const { vectors } = symmetricEigen(N, 4);
    const [w, x, y, z] = vectors.slice(12, 16);
    const R = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ];
    return (p) =>
        [0, 1, 2].map(
            (r) =>
                R[r * 3] * (p[0] - a[0]) +
                R[r * 3 + 1] * (p[1] - a[1]) +
                R[r * 3 + 2] * (p[2] - a[2]) +
                b[r],
        );
}

function percentile(values: number[], q: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? NaN;
}

const log: string[] = [];
const checks: Check[] = [];
const check = (name: string, pass: boolean, detail: string) => checks.push({ name, pass, detail });

/** PatchMatch alone on true poses: per-pixel depth error against the ray-cast depth. */
async function stereoCheck(renderer: SceneRenderer, gpu: GpuContext): Promise<void> {
    const work = { width: 640, height: 360 };
    const k = scaleIntrinsics(K, work.width / WIDTH, work.height / HEIGHT);
    const views = [-2, -1, 0, 1, 2].map((step) => {
        const angle = (step / 36) * Math.PI * 2;
        const pose = lookAt([Math.sin(angle) * 0.42, 0.2, Math.cos(angle) * 0.42]);
        const pixels = renderer.capture({ ...pose, k: K }, WIDTH, HEIGHT);
        const gray = rgbaToGray(pixels, WIDTH, HEIGHT, work.width, work.height).data;
        const R = new Float64Array(pose.R);
        const t = new Float64Array(
            [0, 1, 2].map(
                (r) =>
                    -(
                        pose.R[r * 3] * pose.center[0] +
                        pose.R[r * 3 + 1] * pose.center[1] +
                        pose.R[r * 3 + 2] * pose.center[2]
                    ),
            ),
        );
        return { pose: { R, t }, truth: pose, gray };
    });
    const stereo = new DenseStereo(
        gpu.device,
        work.width,
        work.height,
        views.map((v) => ({ intrinsics: k, pose: v.pose })),
        views.map((v) => v.gray),
    );
    await stereo.solve(2, {
        sources: [0, 1, 3, 4],
        minDepth: 0.1,
        maxDepth: 2,
        prior: null,
        geometric: false,
        iterations: 5,
    });
    const packed = await stereo.depthMap(2);
    const truth = depthGrid(
        { ...views[2].truth, k },
        work.width,
        work.height,
        work.width,
        work.height,
        () => 0,
    );
    let valid = 0;
    let covered = 0;
    const errors: number[] = [];
    const costs = new Array(16).fill(0);
    for (let i = 0; i < packed.length; i++) {
        if (truth[i] <= 0) continue;
        covered++;
        if (packed[i] === 0) continue;
        valid++;
        costs[packed[i] >>> 28]++;
        errors.push(Math.abs(halfToFloat(packed[i] & 0xffff) - truth[i]) / truth[i]);
    }
    stereo.destroy();
    check(
        'PatchMatch recovers depth on true poses',
        valid > covered * 0.6 && percentile(errors, 0.5) < 0.004,
        `${((valid / Math.max(1, covered)) * 100).toFixed(1)} % of surface pixels valid, relative error median ${(percentile(errors, 0.5) * 100).toFixed(2)} % / p90 ${(percentile(errors, 0.9) * 100).toFixed(2)} %, cost histogram ${costs.join(',')}`,
    );
}

function halfToFloat(h: number): number {
    const exponent = (h >> 10) & 31;
    const mantissa = h & 1023;
    return exponent === 0
        ? (mantissa / 1024) * 2 ** -14
        : (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

async function run(): Promise<void> {
    const random = seededRandom(7);

    // WebGPU first, as in the app (the worker owns the device before any WebGL context exists).
    const gpu = await requestGpu();
    log.push(`gpu: ${gpu.label}`);
    const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    const gl = canvas.getContext('webgl2', {
        preserveDrawingBuffer: true,
    }) as WebGL2RenderingContext;
    const renderer = new SceneRenderer(gl);
    await stereoCheck(renderer, gpu);
    const session = new ScanSession(new JpegImageStore());

    const truth: { R: number[]; center: number[] }[] = [];
    const priors: { R: number[]; center: number[] }[] = [];
    let index = 0;
    const started = performance.now();
    for (const ring of RINGS) {
        for (let i = 0; i < ring.count; i++, index++) {
            const angle = (i / ring.count) * Math.PI * 2;
            const pose = lookAt([
                Math.sin(angle) * ring.radius,
                ring.height,
                Math.cos(angle) * ring.radius,
            ]);
            truth.push(pose);
            const camera: CameraState = { ...pose, k: K };
            const pixels = renderer.capture(camera, WIDTH, HEIGHT);
            const grid = depthGrid(camera, WIDTH, HEIGHT, 64, 36, () => (random() - 0.5) * 0.02);
            const prior = drifted(pose, index, random);
            priors.push(prior);
            const verdict = await session.addKeyframe({
                time: index * 0.5,
                image: { width: WIDTH, height: HEIGHT, data: pixels },
                intrinsics: K,
                xrTransform: xrMatrix(prior.R, prior.center),
                depth: { width: 64, height: 36, meters: grid },
            });
            if (!verdict.accepted) log.push(`keyframe ${index} rejected: ${verdict.reason}`);
        }
    }
    const captureMs = performance.now() - started;
    if (gl.isContextLost()) throw new Error('the WebGL context was lost while rendering keyframes');
    check(
        'capture keeps every sharp keyframe',
        session.keyframes.length === index,
        `${session.keyframes.length}/${index} keyframes, ${session.map.points(2).ids.length} live sparse points, ${(captureMs / index).toFixed(0)} ms per keyframe`,
    );

    const priorError = truth.map((pose, i) =>
        Math.hypot(...pose.center.map((v, k) => v - priors[i].center[k])),
    );

    const result = await reconstruct(session, gpu, {
        stage: (event) => {
            if (event.status === 'done')
                log.push(`${event.stage} ${((event.ms ?? 0) / 1000).toFixed(2)} s`);
        },
        cameras: () => undefined,
        sparse: () => undefined,
        dense: () => undefined,
        mesh: () => undefined,
    });
    const m = result.metrics;

    const afterError = truth.map((pose, i) => {
        const c = cameraCenter(session.map.frames[i].pose);
        return Math.hypot(c[0] - pose.center[0], c[1] - pose.center[1], c[2] - pose.center[2]);
    });
    const meanBefore = priorError.reduce((a, b) => a + b, 0) / priorError.length;
    const meanAfter = afterError.reduce((a, b) => a + b, 0) / afterError.length;
    check(
        'bundle adjustment removes the odometry drift',
        meanAfter < meanBefore / 3 && meanAfter < 0.004,
        `camera centre error ${(meanBefore * 1000).toFixed(1)} → ${(meanAfter * 1000).toFixed(2)} mm, reprojection ${m.reprojection.before.median.toFixed(2)} → ${m.reprojection.after.median.toFixed(2)} px (median), ${m.sparsePoints} sparse points`,
    );
    check(
        'reprojection error is sub-pixel after bundle adjustment',
        m.reprojection.after.median < 0.8,
        `mean ${m.reprojection.after.mean.toFixed(2)} px, median ${m.reprojection.after.median.toFixed(2)} px`,
    );

    const align = alignment(
        session.map.frames.map((frame) => cameraCenter(frame.pose)),
        truth.map((pose) => pose.center),
    );
    const alignedCentres = session.map.frames.map((frame, i) =>
        Math.hypot(...align(cameraCenter(frame.pose)).map((v, k) => v - truth[i].center[k])),
    );
    log.push(
        `camera centres after rigid alignment: mean ${((alignedCentres.reduce((x, y) => x + y, 0) / alignedCentres.length) * 1000).toFixed(2)} mm`,
    );
    const densePositions = result.dense.positions;
    const denseErrors: number[] = [];
    for (let i = 0; i < densePositions.length; i += 3 * 7) {
        denseErrors.push(
            surfaceDistance(
                align([densePositions[i], densePositions[i + 1], densePositions[i + 2]]),
            ),
        );
    }
    check(
        'dense cloud lies on the surface',
        percentile(denseErrors, 0.5) < 0.0015 && percentile(denseErrors, 0.9) < 0.005,
        `${m.densePoints} points, error median ${(percentile(denseErrors, 0.5) * 1000).toFixed(2)} mm, p90 ${(percentile(denseErrors, 0.9) * 1000).toFixed(2)} mm, voxel ${(m.voxel * 1000).toFixed(2)} mm`,
    );

    // Completeness: visible surface samples with a mesh vertex nearby.
    const mesh = result.mesh;
    const cell = 0.004;
    const buckets = new Map<string, number[]>();
    const aligned: number[][] = [];
    for (let i = 0; i < mesh.positions.length; i += 3) {
        const p = align([mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
        aligned.push(p);
        const key = `${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)},${Math.floor(p[2] / cell)}`;
        const list = buckets.get(key) ?? [];
        list.push(aligned.length - 1);
        buckets.set(key, list);
    }
    const near = (p: number[], radius: number) => {
        const [x, y, z] = p.map((v) => Math.floor(v / cell));
        for (let dx = -1; dx <= 1; dx++)
            for (let dy = -1; dy <= 1; dy++)
                for (let dz = -1; dz <= 1; dz++) {
                    for (const i of buckets.get(`${x + dx},${y + dy},${z + dz}`) ?? []) {
                        const q = aligned[i];
                        if (Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < radius) return true;
                    }
                }
        return false;
    };
    const samples = surfaceSamples(4000, random);
    const complete = samples.filter((p) => near(p, 0.003)).length / samples.length;
    const meshErrors: number[] = [];
    for (let i = 0; i < aligned.length; i += 5) meshErrors.push(surfaceDistance(aligned[i]));
    check(
        'mesh is accurate and complete',
        percentile(meshErrors, 0.5) < 0.0015 &&
            percentile(meshErrors, 0.9) < 0.004 &&
            complete > 0.85,
        `${m.meshTriangles} triangles, error median ${(percentile(meshErrors, 0.5) * 1000).toFixed(2)} mm, p90 ${(percentile(meshErrors, 0.9) * 1000).toFixed(2)} mm, completeness ${(complete * 100).toFixed(1)} % within 3 mm`,
    );

    // Metric size of the box, measured like a caliper: width and depth from the side faces at
    // mid-height (away from the fillet the fusion leaves where the box meets the table), height
    // from the top face.
    const sides = aligned.filter(
        (p) => p[1] > 0.03 && p[1] < 0.09 && Math.abs(boxDistance(p)) < 0.0015,
    );
    const top = aligned.filter(
        (p) => Math.abs(p[1] - BOX_SIZE[1]) < 0.003 && Math.abs(boxDistance(p)) < 0.0015,
    );
    // Distance between the medians of the two faces normal to axis k (vertices are assigned to
    // the face whose normalised box coordinate is largest, so other faces do not leak in).
    const span = (k: number) => {
        const other = 2 - k;
        const onFace = sides.filter(
            (p) =>
                Math.abs(p[k] - BOX_CENTER[k]) / BOX_HALF[k] >
                Math.abs(p[other] - BOX_CENTER[other]) / BOX_HALF[other],
        );
        const low = onFace.filter((p) => p[k] < BOX_CENTER[k]).map((p) => p[k]);
        const high = onFace.filter((p) => p[k] > BOX_CENTER[k]).map((p) => p[k]);
        return percentile(high, 0.5) - percentile(low, 0.5);
    };
    const size = [
        span(0),
        percentile(
            top.map((p) => p[1]),
            0.5,
        ),
        span(2),
    ];
    const sizeError = size.map((v, k) => Math.abs(v - BOX_SIZE[k]) * 1000);
    check(
        'box dimensions are metric',
        Math.max(...sizeError) < 3,
        `measured ${size.map((v) => (v * 1000).toFixed(1)).join(' × ')} mm vs ${BOX_SIZE.map((v) => (v * 1000).toFixed(0)).join(' × ')} mm`,
    );

    // Texture: atlas colour at each vertex vs the true surface colour.
    const atlas = mesh.atlas;
    const colorErrors: number[] = [];
    for (let v = 0; v < aligned.length; v += 3) {
        const p = aligned[v];
        if (surfaceDistance(p) > 0.002) continue;
        const x = Math.min(atlas.width - 1, Math.floor(mesh.uvs[v * 2] * atlas.width));
        const y = Math.min(atlas.height - 1, Math.floor(mesh.uvs[v * 2 + 1] * atlas.height));
        const o = (y * atlas.width + x) * 4;
        const expected = surfaceColor(p);
        colorErrors.push(
            (Math.abs(atlas.pixels[o] - expected[0]) +
                Math.abs(atlas.pixels[o + 1] - expected[1]) +
                Math.abs(atlas.pixels[o + 2] - expected[2])) /
                3,
        );
    }
    check(
        'texture matches the true surface colour',
        percentile(colorErrors, 0.5) < 14,
        `${m.atlas.join('×')} atlas, ${m.charts} charts, ${m.untextured} untextured faces, colour error median ${percentile(colorErrors, 0.5).toFixed(1)} / p90 ${percentile(colorErrors, 0.9).toFixed(1)} (0–255)`,
    );
}

run().then(
    () => (window.__verify = { checks, log }),
    (error: unknown) =>
        (window.__verify = {
            checks,
            log,
            error: error instanceof Error ? `${error.message}\n${error.stack}` : String(error),
        }),
);
