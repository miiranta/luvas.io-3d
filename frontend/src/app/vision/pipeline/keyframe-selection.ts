import { Pose, cameraCenter } from '../geometry/camera';
import { median } from '../math/dense';
import { rotationAngle } from '../math/mat3';

export interface SelectionFrame {
    pose: Pose;
    /** Median scene depth seen by the frame (m). */
    depthHint: number;
    sharpness: number;
}

/** Same spacing measure as the capture: a baseline of 10 % of the depth or 6° of rotation. */
const BASELINE = 0.1;
const ANGLE = (6 * Math.PI) / 180;

/**
 * Keyframes kept for dense reconstruction (ascending indices). While over budget, the frame
 * that adds least is dropped: the one closest (in viewpoint) to another kept frame, with blurry
 * frames counting as closer. The spread of viewpoints is therefore preserved.
 */
export function selectKeyframes(frames: SelectionFrame[], budget: number): number[] {
    const n = frames.length;
    if (n <= budget) return frames.map((_, i) => i);
    const centres = frames.map((frame) => cameraCenter(frame.pose));
    const focus = median(frames.map((frame) => frame.sharpness)) || 1;
    // Blur discount: a frame half as sharp as the median looks twice as redundant.
    const quality = frames.map((frame) => Math.min(1, Math.max(0.2, frame.sharpness / focus)));
    const distance = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const depth = Math.max(0.15, Math.min(frames[i].depthHint, frames[j].depthHint));
            const baseline = Math.hypot(
                centres[i][0] - centres[j][0],
                centres[i][1] - centres[j][1],
                centres[i][2] - centres[j][2],
            );
            const change = Math.max(
                baseline / (BASELINE * depth),
                rotationAngle(frames[i].pose.R, frames[j].pose.R) / ANGLE,
            );
            distance[i * n + j] = distance[j * n + i] = change;
        }
    }
    const kept = new Uint8Array(n).fill(1);
    let count = n;
    while (count > budget) {
        let drop = -1;
        let lowest = Infinity;
        for (let i = 0; i < n; i++) {
            if (!kept[i]) continue;
            let nearest = Infinity;
            for (let j = 0; j < n; j++) {
                if (j !== i && kept[j]) nearest = Math.min(nearest, distance[i * n + j]);
            }
            const value = nearest * quality[i];
            if (value < lowest) {
                lowest = value;
                drop = i;
            }
        }
        kept[drop] = 0;
        count--;
    }
    const selected: number[] = [];
    for (let i = 0; i < n; i++) if (kept[i]) selected.push(i);
    return selected;
}
