import { scaleIntrinsics } from '../../src/app/vision/geometry/camera';
import { GpuContext } from '../../src/app/vision/gpu/gpu-device';
import { rgbaToGray } from '../../src/app/vision/image/gray-image';
import { DenseStereo } from '../../src/app/vision/mvs/dense-stereo';
import { halfToFloat } from '../../src/app/vision/mvs/depth-maps';
import { SceneRenderer, depthGrid, lookAt } from '../scene-renderer';
import { check, percentile } from './report';
import { HEIGHT, K, WIDTH } from './trajectory';

/** PatchMatch alone on true poses: per-pixel depth error against the ray-cast depth. */
export async function stereoCheck(renderer: SceneRenderer, gpu: GpuContext): Promise<void> {
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
    );
    views.forEach((v, layer) => stereo.setImage(layer, v.gray));
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
