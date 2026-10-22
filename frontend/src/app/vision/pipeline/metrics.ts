import { CAMERA_STRIDE, ErrorSummary } from '../../core/models/scan-results';
import { cameraCenter } from '../geometry/camera';
import { median } from '../math/dense';
import { norm3, sub3 } from '../math/mat3';
import { SparseMap } from '../sfm/sparse-map';

export function summarize(errors: Float64Array): ErrorSummary {
    if (errors.length === 0) return { mean: 0, median: 0 };
    let sum = 0;
    for (const e of errors) sum += e;
    return { mean: sum / errors.length, median: median(errors) };
}

/** Cameras for drawing (see CAMERA_STRIDE): rotation, centre and half field of view. */
export function cameraArray(map: SparseMap): Float32Array {
    const out = new Float32Array(map.frames.length * CAMERA_STRIDE);
    map.frames.forEach((frame, i) => {
        const k = frame.intrinsics;
        out.set(frame.pose.R, i * CAMERA_STRIDE);
        out.set(cameraCenter(frame.pose), i * CAMERA_STRIDE + 9);
        out[i * CAMERA_STRIDE + 12] = frame.width / 2 / k.fx;
        out[i * CAMERA_STRIDE + 13] = frame.height / 2 / k.fy;
    });
    return out;
}

/** Path length, start ↔ end distance and how far adjustment moved the cameras (m). */
export function trajectory(map: SparseMap): {
    length: number;
    closure: number;
    shift: { mean: number; max: number };
} {
    const centres = map.frames.map((frame) => cameraCenter(frame.pose));
    const shifts = map.frames.map((frame, i) => norm3(sub3(centres[i], cameraCenter(frame.prior))));
    let length = 0;
    for (let i = 1; i < centres.length; i++) length += norm3(sub3(centres[i], centres[i - 1]));
    return {
        length,
        closure: norm3(sub3(centres[centres.length - 1], centres[0])),
        shift: {
            mean: shifts.reduce((sum, value) => sum + value, 0) / shifts.length,
            max: Math.max(...shifts),
        },
    };
}

/** Largest distance from any camera to (a sample of) the points: the z-buffer far plane. */
export function farthestDistance(positions: Float32Array, cameras: Float32Array): number {
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
