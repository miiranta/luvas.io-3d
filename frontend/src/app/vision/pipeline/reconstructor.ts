import { DEFAULT_QUALITY, QUALITY, Quality } from '../../core/models/quality';
import { ScanMetrics } from '../../core/models/scan-results';
import { Intrinsics, fromCamera, scaleIntrinsics } from '../geometry/camera';
import { GpuContext } from '../gpu/gpu-device';
import { fuseDepthMaps } from '../fusion/depth-fusion';
import { surfaceArea } from '../fusion/mesh';
import { PointCloud } from '../fusion/point-accumulator';
import { meshDepthMaps } from '../fusion/tsdf-meshing';
import { extent } from '../math/bounds';
import { median } from '../math/dense';
import { STEREO_METHODS, StereoRun } from '../mvs/methods';
import { ModelResult, TEXTURE_METHODS } from '../texture/methods';
import { cameraArray, farthestDistance, summarize, trajectory } from './metrics';
import { ScanSession } from './scan-session';
import { refinePoses } from './stages/refine-poses';
import { ReconstructionSink, StageRunner } from './stages/stage-runner';
import { StereoPlan, planStereo } from './stages/stereo-plan';

export type { ReconstructionSink } from './stages/stage-runner';
export type { ModelResult } from '../texture/methods';

export interface Reconstruction {
    cameras: Float32Array;
    sparse: { positions: Float32Array; colors: Uint8Array };
    dense: PointCloud;
    model: ModelResult;
    metrics: ScanMetrics;
}

/**
 * Offline stage, run when recording stops and again for every rebuild at another quality:
 *   bundle adjustment → PatchMatch stereo (photometric, then geometric) → consistency filter →
 *   dense cloud → TSDF fusion → surface nets → view selection → texture.
 * Each stage's method and parameters come from the quality profile (core/models/quality.ts);
 * the interchangeable methods live in vision/mvs/methods and vision/texture/methods.
 */
export async function reconstruct(
    session: ScanSession,
    gpu: GpuContext,
    sink: ReconstructionSink,
    quality: Quality = DEFAULT_QUALITY,
): Promise<Reconstruction> {
    const profile = QUALITY[quality];
    const { keyframes } = session;
    const live = session.map;
    if (keyframes.length < 3) {
        throw new Error('record a little longer: at least 3 keyframes are needed');
    }
    // Every build starts from the capture as it was recorded.
    session.baseline ??= live.snapshot();
    live.restore(session.baseline);
    const runner = new StageRunner(sink, gpu);
    const device = gpu.device;

    // 1. Poses.
    const before = summarize(live.reprojectionErrors());
    const refinement = await runner.run('bundle', (progress) =>
        refinePoses(session, profile.bundle, device, progress),
    );
    // The map the poses were refined on: its tracks are the sparse cloud. Poses are in both.
    const map = refinement.map;
    const after = summarize(map.reprojectionErrors());
    const cameras = cameraArray(map);
    sink.cameras(cameras);
    const sparse = map.points(2);
    sink.sparse(sparse.positions, sparse.colors);

    // 2. Depth maps.
    const plan = planStereo(
        session,
        profile,
        Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize),
    );
    let run: StereoRun | null = null;
    // Stereo memory is freed before texturing (which needs the GPU too) and on any failure.
    const release = () => {
        run?.destroy();
        run = null;
    };
    try {
        const layers = plan.frames.length;
        const stereo = await runner.run('stereo', async (progress) => {
            progress(0, 'loading images');
            const created = await STEREO_METHODS[profile.stereo.method].create(
                device,
                plan,
                profile.stereo,
            );
            run = created;
            for (let layer = 0; layer < layers; layer++) {
                progress(layer / layers, `${layer + 1}/${layers}`);
                await created.solve(layer);
            }
            return created.stereo;
        });
        await runner.run('geometric', async (progress) => {
            for (let layer = 0; layer < layers && profile.geometric > 0; layer++) {
                progress(layer / layers, `${layer + 1}/${layers}`);
                await stereo.solve(layer, {
                    sources: plan.sources[layer],
                    minDepth: plan.ranges[layer][0],
                    maxDepth: plan.ranges[layer][1],
                    prior: null,
                    geometric: true,
                    iterations: profile.geometric,
                });
            }
        });

        // 3. Dense cloud; its cell size becomes the voxel size.
        const footprint = median(live.frames.map((frame) => frame.depthHint / frame.intrinsics.fx));
        const fused = await runner.run('filter', (progress) =>
            fuseDepthMaps(
                stereo,
                plan.frames.map((frame, layer) => ({
                    ...plan.views[layer],
                    checks: plan.checks[layer],
                    needed: plan.needed[layer],
                    thumb: keyframes[frame].thumb,
                })),
                Math.min(0.05, Math.max(0.0008, profile.mesh.voxelFactor * footprint)),
                profile.mesh.cellBudget,
                progress,
            ),
        );
        const dense = fused.cloud;
        sink.dense(dense);
        if (dense.positions.length / 3 < 100) {
            throw new Error(
                'dense stereo found almost no surface — scan a textured scene with more motion',
            );
        }

        // 4. Mesh.
        const meshed = await runner.run('mesh', (progress) =>
            meshDepthMaps(
                device,
                stereo,
                plan.frames.map((_, layer) => ({
                    ...plan.views[layer],
                    checks: plan.checks[layer],
                    needed: plan.meshNeeded[layer],
                    maxDepth: plan.ranges[layer][1],
                    prior: () => plan.prior(layer, plan.width, plan.height),
                })),
                dense.positions,
                phoneDepthPoints(session, plan),
                fused.cell,
                profile.mesh.confidence ? median(live.frames.map((frame) => frame.depthHint)) : 0,
                progress,
            ),
        );
        release();
        const mesh = meshed.mesh;
        if (mesh.indices.length === 0) throw new Error('fusion produced an empty surface');
        sink.mesh(mesh);

        // 5. Texture.
        const method = TEXTURE_METHODS[profile.texture];
        const withFocal = (k: Intrinsics): Intrinsics => ({
            ...k,
            fx: k.fx * refinement.focalScale,
            fy: k.fy * refinement.focalScale,
        });
        const focus = Math.max(...plan.frames.map((i) => keyframes[i].sharpness));
        const input = {
            device,
            mesh,
            views: plan.frames.map((i) => {
                const k = keyframes[i];
                return {
                    intrinsics: withFocal(k.texture.intrinsics),
                    pose: live.frames[i].pose,
                    width: k.texture.width,
                    height: k.texture.height,
                    thumb: { ...k.thumb, intrinsics: withFocal(k.thumb.intrinsics) },
                    quality: Math.sqrt(Math.max(1e-6, k.sharpness / focus)),
                };
            }),
            image: (view: number) => session.images.get(plan.frames[view]),
            jpeg: session.images.jpeg.bind(session.images),
            visibility: profile.visibility,
            far: farthestDistance(mesh.positions, cameras) * 1.2 + 0.1,
        };
        const selection = await runner.run('views', (progress) =>
            method.selectViews(input, progress),
        );
        const textured = await runner.run('texture', (progress) =>
            method.bake(input, selection, progress),
        );

        const area = surfaceArea(mesh);
        const path = trajectory(map);
        return {
            cameras,
            sparse,
            dense,
            model: textured.model,
            metrics: {
                quality,
                gpu: gpu.label,
                keyframes: keyframes.length,
                keyframesUsed: layers,
                tracking: {
                    breaks: live.breaks.size,
                    fallbacks: session.fallbacks,
                    realigned: session.realigned,
                },
                focalCorrection: refinement.focalScale - 1,
                stereoSize: [plan.width, plan.height],
                blurred: session.rejected,
                depthSensor: keyframes.some((k) => k.depth !== null),
                workSize: [keyframes[0].work.width, keyframes[0].work.height],
                textureSize: [keyframes[0].texture.width, keyframes[0].texture.height],
                tracks: map.tracks.filter((track) => track.alive && track.point).length,
                sparsePoints: sparse.positions.length / 3,
                observations: map.reprojectionErrors().length,
                reprojection: { before, after },
                poseShift: path.shift,
                bundleIterations: refinement.iterations,
                trajectory: { length: path.length, closure: path.closure },
                densePoints: dense.positions.length / 3,
                voxel: meshed.voxel,
                meshVertices: textured.model.positions.length / 3,
                meshTriangles: textured.model.indices.length / 3,
                surfaceArea: area,
                density: area > 0 ? dense.positions.length / 3 / area : 0,
                extent: extent(mesh.positions),
                atlas: textured.atlas,
                charts: textured.charts,
                untextured: textured.untextured,
                timings: runner.timings,
            },
        };
    } finally {
        release();
    }
}

/** World points of the phone depth of the used keyframes (allocates volume for blank walls). */
function phoneDepthPoints(session: ScanSession, plan: StereoPlan): Float32Array {
    const points: number[] = [];
    for (const index of plan.frames) {
        const { depth, work } = session.keyframes[index];
        if (!depth) continue;
        const pose = session.map.frames[index].pose;
        const k = scaleIntrinsics(
            work.intrinsics,
            depth.width / work.width,
            depth.height / work.height,
        );
        for (let y = 0; y < depth.height; y += 2) {
            for (let x = 0; x < depth.width; x += 2) {
                const d = depth.meters[y * depth.width + x];
                if (!(d > 0.05)) continue;
                points.push(
                    ...fromCamera(pose, [((x - k.cx) / k.fx) * d, ((y - k.cy) / k.fy) * d, d]),
                );
            }
        }
    }
    return Float32Array.from(points);
}
