import { solveCholesky } from '../math/dense';
import { expSO3, multiply3 } from '../math/mat3';
import { Intrinsics, Pose, clonePose } from './camera';

export interface Resection {
    pose: Pose;
    inliers: number;
    /** RMS reprojection error of the inliers (px). */
    rms: number;
}

const HUBER = 2;

/**
 * Camera pose from 2D–3D correspondences (Levenberg–Marquardt with a Huber loss, starting from
 * `initial`), trimming outliers in three rounds (16, 6 and 3 px). Returns null when too few
 * correspondences survive.
 */
export function resect(
    k: Intrinsics,
    initial: Pose,
    points: Float64Array,
    uv: Float64Array,
    minInliers = 12,
): Resection | null {
    let pose = clonePose(initial);
    let active = Array.from({ length: points.length / 3 }, (_, i) => i);
    for (const limit of [16, 6, 3]) {
        pose = optimise(k, pose, points, uv, active, 15);
        active = active.filter((i) => error(k, pose, points, uv, i) < limit);
        if (active.length < minInliers) return null;
    }
    pose = optimise(k, pose, points, uv, active, 8);
    let sum = 0;
    for (const i of active) sum += error(k, pose, points, uv, i) ** 2;
    return { pose, inliers: active.length, rms: Math.sqrt(sum / active.length) };
}

function error(
    k: Intrinsics,
    pose: Pose,
    points: Float64Array,
    uv: Float64Array,
    i: number,
): number {
    const { R, t } = pose;
    const X = points.subarray(i * 3, i * 3 + 3);
    const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    if (z <= 1e-4) return Infinity;
    return Math.hypot((k.fx * x) / z + k.cx - uv[i * 2], (k.fy * y) / z + k.cy - uv[i * 2 + 1]);
}

function cost(
    k: Intrinsics,
    pose: Pose,
    points: Float64Array,
    uv: Float64Array,
    active: number[],
): number {
    let sum = 0;
    for (const i of active) {
        const e = Math.min(error(k, pose, points, uv, i), 1e3);
        sum += e <= HUBER ? e * e : 2 * HUBER * e - HUBER * HUBER;
    }
    return sum;
}

function optimise(
    k: Intrinsics,
    start: Pose,
    points: Float64Array,
    uv: Float64Array,
    active: number[],
    iterations: number,
): Pose {
    let pose = start;
    let current = cost(k, pose, points, uv, active);
    let lambda = 1e-3;
    const jc = new Float64Array(12);
    for (let iteration = 0; iteration < iterations; iteration++) {
        const H = new Float64Array(36);
        const g = new Float64Array(6);
        const { R, t } = pose;
        for (const i of active) {
            const X = points.subarray(i * 3, i * 3 + 3);
            const rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2];
            const ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2];
            const rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
            const x = rx + t[0];
            const y = ry + t[1];
            const z = rz + t[2];
            if (z <= 1e-4) continue;
            const iz = 1 / z;
            const ru = k.fx * x * iz + k.cx - uv[i * 2];
            const rv = k.fy * y * iz + k.cy - uv[i * 2 + 1];
            const norm = Math.hypot(ru, rv);
            const w = norm <= HUBER ? 1 : HUBER / norm;
            const a0 = k.fx * iz;
            const a2 = -k.fx * x * iz * iz;
            const b1 = k.fy * iz;
            const b2 = -k.fy * y * iz * iz;
            // Same left-perturbation Jacobian as the bundle adjuster.
            jc[0] = a2 * ry;
            jc[1] = a0 * rz - a2 * rx;
            jc[2] = -a0 * ry;
            jc[3] = a0;
            jc[4] = 0;
            jc[5] = a2;
            jc[6] = b2 * ry - b1 * rz;
            jc[7] = -b2 * rx;
            jc[8] = b1 * rx;
            jc[9] = 0;
            jc[10] = b1;
            jc[11] = b2;
            for (let r = 0; r < 6; r++) {
                g[r] -= w * (jc[r] * ru + jc[6 + r] * rv);
                for (let s = 0; s < 6; s++)
                    H[r * 6 + s] += w * (jc[r] * jc[s] + jc[6 + r] * jc[6 + s]);
            }
        }
        let improved = false;
        for (let attempt = 0; attempt < 6 && !improved; attempt++) {
            const damped = new Float64Array(H);
            for (let d = 0; d < 6; d++) damped[d * 7] = damped[d * 7] * (1 + lambda) + 1e-9;
            const delta = solveCholesky(damped, g, 6);
            if (!delta) break;
            const next: Pose = {
                R: multiply3(expSO3([delta[0], delta[1], delta[2]]), pose.R),
                t: new Float64Array([
                    pose.t[0] + delta[3],
                    pose.t[1] + delta[4],
                    pose.t[2] + delta[5],
                ]),
            };
            const nextCost = cost(k, next, points, uv, active);
            if (nextCost < current) {
                const gain = (current - nextCost) / Math.max(1e-12, current);
                pose = next;
                current = nextCost;
                lambda = Math.max(1e-7, lambda / 3);
                improved = true;
                if (gain < 1e-7) return pose;
            } else {
                lambda *= 4;
            }
        }
        if (!improved) break;
    }
    return pose;
}
