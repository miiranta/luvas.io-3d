import { invert3 } from '../math/mat3';
import { Intrinsics, Pose, cameraCenter } from './camera';

export interface Observation {
    k: Intrinsics;
    pose: Pose;
    u: number;
    v: number;
}

export interface Triangulated {
    point: [number, number, number];
    /** Largest angle (radians) between any two viewing rays. */
    angle: number;
    /** Root-mean-square reprojection error in pixels. */
    rms: number;
    /** Largest single reprojection error in pixels. */
    worst: number;
}

const scratch = new Float64Array(9);

/**
 * Multi-view triangulation: least-squares ray intersection (closed form) followed by a few
 * Gauss–Newton steps on the reprojection error.
 */
export function triangulate(observations: Observation[]): Triangulated | null {
    if (observations.length < 2) return null;
    const a = new Float64Array(9);
    const b = new Float64Array(3);
    const rays: [number, number, number][] = [];
    for (const { k, pose, u, v } of observations) {
        const x = (u - k.cx) / k.fx;
        const y = (v - k.cy) / k.fy;
        const R = pose.R;
        // World ray direction = Rᵀ · (x, y, 1), normalised.
        let dx = R[0] * x + R[3] * y + R[6];
        let dy = R[1] * x + R[4] * y + R[7];
        let dz = R[2] * x + R[5] * y + R[8];
        const length = Math.hypot(dx, dy, dz);
        dx /= length;
        dy /= length;
        dz /= length;
        rays.push([dx, dy, dz]);
        const c = cameraCenter(pose);
        // Accumulate (I − d dᵀ) for the point-to-ray distance.
        const m = [1 - dx * dx, -dx * dy, -dx * dz, -dy * dx, 1 - dy * dy, -dy * dz];
        a[0] += m[0];
        a[1] += m[1];
        a[2] += m[2];
        a[4] += m[4];
        a[5] += m[5];
        a[8] += 1 - dz * dz;
        b[0] += m[0] * c[0] + m[1] * c[1] + m[2] * c[2];
        b[1] += m[3] * c[0] + m[4] * c[1] + m[5] * c[2];
        b[2] += -dz * dx * c[0] - dz * dy * c[1] + (1 - dz * dz) * c[2];
    }
    a[3] = a[1];
    a[6] = a[2];
    a[7] = a[5];
    if (!invert3(a, scratch)) return null;
    const point: [number, number, number] = [
        scratch[0] * b[0] + scratch[1] * b[1] + scratch[2] * b[2],
        scratch[3] * b[0] + scratch[4] * b[1] + scratch[5] * b[2],
        scratch[6] * b[0] + scratch[7] * b[1] + scratch[8] * b[2],
    ];

    let angle = 0;
    for (let i = 0; i < rays.length; i++) {
        for (let j = i + 1; j < rays.length; j++) {
            const c = rays[i][0] * rays[j][0] + rays[i][1] * rays[j][1] + rays[i][2] * rays[j][2];
            angle = Math.max(angle, Math.acos(Math.min(1, c)));
        }
    }

    if (!refinePoint(point, observations)) return null;
    let sum = 0;
    let worst = 0;
    for (const observation of observations) {
        const error = reprojectionError(point, observation);
        if (!Number.isFinite(error)) return null;
        sum += error * error;
        worst = Math.max(worst, error);
    }
    return { point, angle, rms: Math.sqrt(sum / observations.length), worst };
}

export function reprojectionError(point: ArrayLike<number>, o: Observation): number {
    const R = o.pose.R;
    const t = o.pose.t;
    const x = R[0] * point[0] + R[1] * point[1] + R[2] * point[2] + t[0];
    const y = R[3] * point[0] + R[4] * point[1] + R[5] * point[2] + t[1];
    const z = R[6] * point[0] + R[7] * point[1] + R[8] * point[2] + t[2];
    if (z <= 1e-6) return Infinity;
    return Math.hypot((o.k.fx * x) / z + o.k.cx - o.u, (o.k.fy * y) / z + o.k.cy - o.v);
}

/** Gauss–Newton on the point only; returns false if the point ends behind a camera. */
function refinePoint(point: [number, number, number], observations: Observation[]): boolean {
    const h = new Float64Array(9);
    const g = new Float64Array(3);
    for (let iteration = 0; iteration < 4; iteration++) {
        h.fill(0);
        g.fill(0);
        for (const { k, pose, u, v } of observations) {
            const R = pose.R;
            const t = pose.t;
            const x = R[0] * point[0] + R[1] * point[1] + R[2] * point[2] + t[0];
            const y = R[3] * point[0] + R[4] * point[1] + R[5] * point[2] + t[1];
            const z = R[6] * point[0] + R[7] * point[1] + R[8] * point[2] + t[2];
            if (z <= 1e-6) return false;
            const iz = 1 / z;
            const ru = k.fx * x * iz + k.cx - u;
            const rv = k.fy * y * iz + k.cy - v;
            // d(u, v)/d(camera point) · R
            const ju = [k.fx * iz, 0, -k.fx * x * iz * iz];
            const jv = [0, k.fy * iz, -k.fy * y * iz * iz];
            const jU = [
                ju[0] * R[0] + ju[2] * R[6],
                ju[0] * R[1] + ju[2] * R[7],
                ju[0] * R[2] + ju[2] * R[8],
            ];
            const jV = [
                jv[1] * R[3] + jv[2] * R[6],
                jv[1] * R[4] + jv[2] * R[7],
                jv[1] * R[5] + jv[2] * R[8],
            ];
            for (let r = 0; r < 3; r++) {
                g[r] += jU[r] * ru + jV[r] * rv;
                for (let c = 0; c < 3; c++) h[r * 3 + c] += jU[r] * jU[c] + jV[r] * jV[c];
            }
        }
        if (!invert3(h, scratch)) return false;
        const dx = scratch[0] * g[0] + scratch[1] * g[1] + scratch[2] * g[2];
        const dy = scratch[3] * g[0] + scratch[4] * g[1] + scratch[5] * g[2];
        const dz = scratch[6] * g[0] + scratch[7] * g[1] + scratch[8] * g[2];
        point[0] -= dx;
        point[1] -= dy;
        point[2] -= dz;
        if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) < 1e-9) break;
    }
    return true;
}
