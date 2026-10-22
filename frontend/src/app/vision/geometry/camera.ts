import {
    Mat3,
    Vec3,
    apply3,
    applyTransposed3,
    multiply3,
    orthonormalize3,
    transpose3,
} from '../math/mat3';

/** Pinhole intrinsics in pixels of a specific image size. */
export interface Intrinsics {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
}

/**
 * Rigid world → camera transform in computer-vision axes (x right, y down, z forward):
 * x_cam = R · X + t.
 */
export interface Pose {
    R: Mat3;
    t: Float64Array;
}

export function scaleIntrinsics(k: Intrinsics, sx: number, sy = sx): Intrinsics {
    // Pixel centres stay aligned: (c + 0.5) · s − 0.5.
    return {
        fx: k.fx * sx,
        fy: k.fy * sy,
        cx: (k.cx + 0.5) * sx - 0.5,
        cy: (k.cy + 0.5) * sy - 0.5,
    };
}

export function clonePose(pose: Pose): Pose {
    return { R: new Float64Array(pose.R), t: new Float64Array(pose.t) };
}

export function cameraCenter(pose: Pose): Vec3 {
    const c = applyTransposed3(pose.R, pose.t);
    return [-c[0], -c[1], -c[2]];
}

/** Viewing direction (camera +z) in world coordinates. */
export function opticalAxis(pose: Pose): Vec3 {
    return [pose.R[6], pose.R[7], pose.R[8]];
}

export function toCamera(pose: Pose, x: ArrayLike<number>): Vec3 {
    const p = apply3(pose.R, x);
    return [p[0] + pose.t[0], p[1] + pose.t[1], p[2] + pose.t[2]];
}

export function fromCamera(pose: Pose, x: ArrayLike<number>): Vec3 {
    return applyTransposed3(pose.R, [x[0] - pose.t[0], x[1] - pose.t[1], x[2] - pose.t[2]]);
}

/** Projects a world point; returns [u, v, depth] (depth ≤ 0 means behind the camera). */
export function project(k: Intrinsics, pose: Pose, x: ArrayLike<number>): Vec3 {
    const p = toCamera(pose, x);
    return [(k.fx * p[0]) / p[2] + k.cx, (k.fy * p[1]) / p[2] + k.cy, p[2]];
}

/** Unit-depth ray of pixel (u, v) in camera coordinates. */
export function pixelRay(k: Intrinsics, u: number, v: number): Vec3 {
    return [(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1];
}

/** Transform taking camera-a coordinates to camera-b coordinates. */
export function relativePose(a: Pose, b: Pose): Pose {
    const R = multiply3(b.R, transpose3(a.R));
    const rt = apply3(R, a.t);
    return { R, t: new Float64Array([b.t[0] - rt[0], b.t[1] - rt[1], b.t[2] - rt[2]]) };
}

/** Fundamental matrix with x_bᵀ · F · x_a = 0 (row-major). */
export function fundamentalMatrix(ka: Intrinsics, kb: Intrinsics, a: Pose, b: Pose): Mat3 {
    const { R, t } = relativePose(a, b);
    const tx = new Float64Array([0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0]);
    const essential = multiply3(tx, R);
    const kaInv = new Float64Array([
        1 / ka.fx,
        0,
        -ka.cx / ka.fx,
        0,
        1 / ka.fy,
        -ka.cy / ka.fy,
        0,
        0,
        1,
    ]);
    const kbInvT = new Float64Array([
        1 / kb.fx,
        0,
        0,
        0,
        1 / kb.fy,
        0,
        -kb.cx / kb.fx,
        -kb.cy / kb.fy,
        1,
    ]);
    return multiply3(kbInvT, multiply3(essential, kaInv));
}

/**
 * Converts a WebXR view transform (column-major camera → world, OpenGL axes: y up, looking
 * down −z) into a world → camera pose with computer-vision axes.
 */
export function poseFromXrTransform(matrix: ArrayLike<number>): Pose {
    // Columns 1 and 2 flip sign to go from GL camera axes to CV camera axes.
    const cameraToWorld = new Float64Array([
        matrix[0],
        -matrix[4],
        -matrix[8],
        matrix[1],
        -matrix[5],
        -matrix[9],
        matrix[2],
        -matrix[6],
        -matrix[10],
    ]);
    const R = orthonormalize3(transpose3(cameraToWorld));
    const center = [matrix[12], matrix[13], matrix[14]];
    const rc = apply3(R, center);
    return { R, t: new Float64Array([-rc[0], -rc[1], -rc[2]]) };
}

/**
 * Intrinsics from a WebXR / OpenGL projection matrix (column-major) for an image of
 * width × height whose first row is the top of the view.
 */
export function intrinsicsFromProjection(
    projection: ArrayLike<number>,
    width: number,
    height: number,
): Intrinsics {
    return {
        fx: (projection[0] * width) / 2,
        fy: (projection[5] * height) / 2,
        cx: ((1 - projection[8]) * width) / 2 - 0.5,
        cy: ((1 + projection[9]) * height) / 2 - 0.5,
    };
}

/** Column-major 4×4 view matrix (world → GL camera) for rendering a CV pose. */
export function glViewMatrix(pose: Pose): Float32Array {
    const { R, t } = pose;
    // GL camera = diag(1, −1, −1) · CV camera.
    return new Float32Array([
        R[0],
        -R[3],
        -R[6],
        0,
        R[1],
        -R[4],
        -R[7],
        0,
        R[2],
        -R[5],
        -R[8],
        0,
        t[0],
        -t[1],
        -t[2],
        1,
    ]);
}

/** World → world rigid transform X' = R·X + t. */
export interface WorldTransform {
    R: Float64Array;
    t: Float64Array;
}

/**
 * The world correction that moves camera `from` onto `to` (both world → camera):
 * to = from ∘ G, i.e. G = from⁻¹ ∘ to.
 */
export function worldCorrection(from: Pose, to: Pose): WorldTransform {
    const R = multiply3(transpose3(from.R), to.R);
    const d = [to.t[0] - from.t[0], to.t[1] - from.t[1], to.t[2] - from.t[2]];
    return { R, t: new Float64Array(applyTransposed3(from.R, d)) };
}

/** Applies a world correction to a pose: pose ∘ G. */
export function correctPose(pose: Pose, correction: WorldTransform): Pose {
    const rt = apply3(pose.R, correction.t);
    return {
        R: multiply3(pose.R, correction.R),
        t: new Float64Array([rt[0] + pose.t[0], rt[1] + pose.t[1], rt[2] + pose.t[2]]),
    };
}

/** Composition a ∘ b of world corrections (b first). */
export function composeCorrections(a: WorldTransform, b: WorldTransform): WorldTransform {
    const at = apply3(a.R, b.t);
    return {
        R: multiply3(a.R, b.R),
        t: new Float64Array([at[0] + a.t[0], at[1] + a.t[1], at[2] + a.t[2]]),
    };
}
