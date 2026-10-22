import { Intrinsics, Pose } from '../../geometry/camera';
import { expSO3, multiply3 } from '../../math/mat3';

export interface BundleProblem {
    intrinsics: Intrinsics[];
    /** Current poses, refined in place. */
    poses: Pose[];
    /** Phone (ARCore) poses: consecutive pairs become relative-motion priors. */
    priors: Pose[];
    /** xyz per point, refined in place. */
    points: Float64Array;
    camera: Int32Array;
    point: Int32Array;
    u: Float64Array;
    v: Float64Array;
    /**
     * Keyframe indices i whose odometry link to i − 1 is unreliable (tracking was lost or the
     * phone pose jumped): no relative-motion prior is added across them.
     */
    breaks?: ReadonlySet<number>;
}

export interface BundleOptions {
    iterations: number;
    /** Also refine one focal-length scale shared by all keyframes (self-calibration). */
    calibrateFocal?: boolean;
    /** Starting focal scale (default 1). */
    focalScale?: number;
}

export interface BundleReport {
    iterations: number;
    initialRms: number;
    finalRms: number;
    /** Refined focal scale (1 when not calibrated). */
    focalScale: number;
}

export interface Layout {
    /** Size of the reduced (camera + intrinsic) system. */
    n: number;
    /** Parameters touched by one observation on the camera side (6, or 7 with the focal). */
    rows: number;
    /** Index of the shared focal scale, −1 when not calibrated. */
    focal: number;
}

/** Global parameter index of row r of observation o's camera-side Jacobian. */
export function parameter(layout: Layout, camera: number, r: number): number {
    return r < 6 ? camera * 6 + r : layout.focal;
}

export function applyPoseDelta(pose: Pose, delta: Float64Array, offset: number): Pose {
    const rotation = expSO3([delta[offset], delta[offset + 1], delta[offset + 2]]);
    return {
        R: multiply3(rotation, pose.R),
        t: new Float64Array([
            pose.t[0] + delta[offset + 3],
            pose.t[1] + delta[offset + 4],
            pose.t[2] + delta[offset + 5],
        ]),
    };
}

export function observationError(
    problem: BundleProblem,
    poses: Pose[],
    points: Float64Array,
    scale: number,
    o: number,
): number {
    const k = problem.intrinsics[problem.camera[o]];
    const { R, t } = poses[problem.camera[o]];
    const p = problem.point[o] * 3;
    const x = R[0] * points[p] + R[1] * points[p + 1] + R[2] * points[p + 2] + t[0];
    const y = R[3] * points[p] + R[4] * points[p + 1] + R[5] * points[p + 2] + t[1];
    const z = R[6] * points[p] + R[7] * points[p + 1] + R[8] * points[p + 2] + t[2];
    if (z <= 1e-4) return 1e3;
    return Math.hypot(
        (k.fx * scale * x) / z + k.cx - problem.u[o],
        (k.fy * scale * y) / z + k.cy - problem.v[o],
    );
}

export function observationErrors(problem: BundleProblem, scale = 1): Float64Array {
    const errors = new Float64Array(problem.camera.length);
    for (let o = 0; o < errors.length; o++) {
        errors[o] = observationError(problem, problem.poses, problem.points, scale, o);
    }
    return errors;
}

export function rmsError(problem: BundleProblem, scale: number): number {
    const errors = observationErrors(problem, scale);
    let sum = 0;
    for (const e of errors) sum += Math.min(e, 10) ** 2;
    return Math.sqrt(sum / Math.max(1, errors.length));
}

export function groupByPoint(problem: BundleProblem): { offsets: Int32Array; order: Int32Array } {
    const points = problem.points.length / 3;
    const offsets = new Int32Array(points + 1);
    for (const p of problem.point) offsets[p + 1]++;
    for (let p = 0; p < points; p++) offsets[p + 1] += offsets[p];
    const cursor = offsets.slice(0, -1);
    const order = new Int32Array(problem.point.length);
    for (let o = 0; o < problem.point.length; o++) order[cursor[problem.point[o]]++] = o;
    return { offsets, order };
}
