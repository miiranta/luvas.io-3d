/** Row-major 3×3 matrix. */
export type Mat3 = Float64Array;
export type Vec3 = [number, number, number];

export function identity3(): Mat3 {
    return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

export function multiply3(a: Mat3, b: Mat3): Mat3 {
    const out = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        }
    }
    return out;
}

export function transpose3(a: Mat3): Mat3 {
    return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}

export function apply3(a: Mat3, v: ArrayLike<number>): Vec3 {
    return [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ];
}

/** Rᵀ v without building the transpose. */
export function applyTransposed3(a: Mat3, v: ArrayLike<number>): Vec3 {
    return [
        a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
        a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
        a[2] * v[0] + a[5] * v[1] + a[8] * v[2],
    ];
}

export function dot3(a: ArrayLike<number>, b: ArrayLike<number>): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a: ArrayLike<number>, b: ArrayLike<number>): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function sub3(a: ArrayLike<number>, b: ArrayLike<number>): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function norm3(a: ArrayLike<number>): number {
    return Math.hypot(a[0], a[1], a[2]);
}

export function normalize3(a: ArrayLike<number>): Vec3 {
    const length = norm3(a) || 1;
    return [a[0] / length, a[1] / length, a[2] / length];
}

/** Rodrigues: rotation vector (axis · angle) → rotation matrix. */
export function expSO3(w: ArrayLike<number>): Mat3 {
    const theta = norm3(w);
    if (theta < 1e-12) {
        return new Float64Array([1, -w[2], w[1], w[2], 1, -w[0], -w[1], w[0], 1]);
    }
    const x = w[0] / theta;
    const y = w[1] / theta;
    const z = w[2] / theta;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const k = 1 - c;
    return new Float64Array([
        c + x * x * k,
        x * y * k - z * s,
        x * z * k + y * s,
        y * x * k + z * s,
        c + y * y * k,
        y * z * k - x * s,
        z * x * k - y * s,
        z * y * k + x * s,
        c + z * z * k,
    ]);
}

/** Inverse of expSO3: rotation matrix → rotation vector. */
export function logSO3(r: Mat3): Vec3 {
    const cosTheta = Math.min(1, Math.max(-1, (r[0] + r[4] + r[8] - 1) / 2));
    const theta = Math.acos(cosTheta);
    const v: Vec3 = [r[7] - r[5], r[2] - r[6], r[3] - r[1]];
    if (theta < 1e-6) return [v[0] / 2, v[1] / 2, v[2] / 2];
    if (Math.PI - theta < 1e-4) {
        // Near 180°: recover the axis from the diagonal of (R + I) / 2.
        const xx = Math.sqrt(Math.max(0, (r[0] + 1) / 2));
        const yy = Math.sqrt(Math.max(0, (r[4] + 1) / 2));
        const zz = Math.sqrt(Math.max(0, (r[8] + 1) / 2));
        const axis = normalize3([xx, r[1] + r[3] >= 0 ? yy : -yy, r[2] + r[6] >= 0 ? zz : -zz]);
        return [axis[0] * theta, axis[1] * theta, axis[2] * theta];
    }
    const scale = theta / (2 * Math.sin(theta));
    return [v[0] * scale, v[1] * scale, v[2] * scale];
}

export function rotationAngle(a: Mat3, b: Mat3): number {
    // Angle of a·bᵀ from its trace.
    let trace = 0;
    for (let i = 0; i < 3; i++) {
        trace += a[i * 3] * b[i * 3] + a[i * 3 + 1] * b[i * 3 + 1] + a[i * 3 + 2] * b[i * 3 + 2];
    }
    return Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2)));
}

/** Closest rotation to a near-orthonormal matrix (Gram–Schmidt on the rows). */
export function orthonormalize3(a: Mat3): Mat3 {
    const x = normalize3([a[0], a[1], a[2]]);
    let y: Vec3 = [a[3], a[4], a[5]];
    const d = dot3(x, y);
    y = normalize3([y[0] - d * x[0], y[1] - d * x[1], y[2] - d * x[2]]);
    const z = cross3(x, y);
    return new Float64Array([...x, ...y, ...z]);
}

/** Inverse of a 3×3 (adjugate / determinant); returns false when singular. */
export function invert3(a: ArrayLike<number>, out: Float64Array): boolean {
    const c00 = a[4] * a[8] - a[5] * a[7];
    const c01 = a[5] * a[6] - a[3] * a[8];
    const c02 = a[3] * a[7] - a[4] * a[6];
    const det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    if (!(Math.abs(det) > 1e-18)) return false;
    const inv = 1 / det;
    out[0] = c00 * inv;
    out[1] = (a[2] * a[7] - a[1] * a[8]) * inv;
    out[2] = (a[1] * a[5] - a[2] * a[4]) * inv;
    out[3] = c01 * inv;
    out[4] = (a[0] * a[8] - a[2] * a[6]) * inv;
    out[5] = (a[2] * a[3] - a[0] * a[5]) * inv;
    out[6] = c02 * inv;
    out[7] = (a[1] * a[6] - a[0] * a[7]) * inv;
    out[8] = (a[0] * a[4] - a[1] * a[3]) * inv;
    return true;
}
