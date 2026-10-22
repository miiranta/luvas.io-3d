import { extractSift, siftDistance } from '../../src/app/vision/features/sift-features';
import { GpuSift } from '../../src/app/vision/features/sift/gpu-sift';
import { GpuContext } from '../../src/app/vision/gpu/gpu-device';
import { rgbaToGray } from '../../src/app/vision/image/gray-image';
import { SceneRenderer, lookAt } from '../scene-renderer';
import { check, percentile } from './report';
import { HEIGHT, K, WIDTH } from './trajectory';

/** The GPU SIFT must find the CPU reference's keypoints with the same descriptors, faster. */
export async function siftCheck(renderer: SceneRenderer, gpu: GpuContext): Promise<void> {
    const pose = lookAt([0.3, 0.25, 0.32]);
    const pixels = renderer.capture({ ...pose, k: K }, WIDTH, HEIGHT);
    const gray = rgbaToGray(pixels, WIDTH, HEIGHT, 640, 360);
    const budget = 2000;
    let started = performance.now();
    const cpu = extractSift(gray, budget);
    const cpuMs = performance.now() - started;
    const sift = new GpuSift(gpu.device, 640, 360);
    await sift.extract(gray, budget);
    started = performance.now();
    const runs = 5;
    let features = await sift.extract(gray, budget);
    for (let i = 1; i < runs; i++) features = await sift.extract(gray, budget);
    const gpuMs = (performance.now() - started) / runs;
    sift.destroy();

    const offsets: number[] = [];
    const distances: number[] = [];
    for (let i = 0; i < features.count; i++) {
        let best = -1;
        let error = 0.25;
        for (let j = 0; j < cpu.count; j++) {
            if (cpu.level[j] !== features.level[i]) continue;
            const e = Math.hypot(cpu.x[j] - features.x[i], cpu.y[j] - features.y[i]);
            if (e < error) {
                error = e;
                best = j;
            }
        }
        if (best < 0) continue;
        offsets.push(error);
        distances.push(siftDistance(features.descriptors, i * 32, cpu.descriptors, best * 32));
    }
    const shared = offsets.length / Math.max(1, Math.max(features.count, cpu.count));
    check(
        'GPU SIFT matches the CPU reference',
        shared > 0.9 && percentile(offsets, 0.5) < 0.01 && percentile(distances, 0.9) < 40,
        `${features.count} / ${cpu.count} keypoints (GPU / CPU), ${(shared * 100).toFixed(1)} % shared, offset median ${percentile(offsets, 0.5).toFixed(4)} px, descriptor distance median ${percentile(distances, 0.5).toFixed(1)} / p90 ${percentile(distances, 0.9).toFixed(1)} of 512, ${gpuMs.toFixed(1)} ms vs ${cpuMs.toFixed(0)} ms per frame`,
    );
}
