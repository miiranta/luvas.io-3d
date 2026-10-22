import { symmetricEigen } from '../../src/app/vision/math/dense';

export type TruePose = { R: number[]; center: number[] };

export const WIDTH = 1280;
export const HEIGHT = 720;
export const K = { fx: 1000, fy: 1000, cx: 639.5, cy: 359.5 };
/** Two rings around the scene: height, radius and keyframes of each. */
export const RINGS = [
    { height: 0.2, radius: 0.42, count: 36 },
    { height: 0.36, radius: 0.36, count: 24 },
];

/**
 * Phone odometry: a world-frame drift D_i = (yaw i·0.25 mrad, translation i·0.3 mm) applied to
 * the true camera, plus 0.5 mm of jitter. Prior camera → world = D_i ∘ truth.
 */
export function drifted(pose: TruePose, index: number, random: () => number) {
    const yaw = index * 0.00025;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const D = [c, 0, s, 0, 1, 0, -s, 0, c];
    // World → camera rotation R·Dᵀ.
    const R = [0, 1, 2].flatMap((r) =>
        [0, 1, 2].map(
            (col) =>
                pose.R[r * 3] * D[col * 3] +
                pose.R[r * 3 + 1] * D[col * 3 + 1] +
                pose.R[r * 3 + 2] * D[col * 3 + 2],
        ),
    );
    const offset = [index * 0.0003, -index * 0.00015, index * 0.0002];
    const center = [0, 1, 2].map(
        (k) =>
            D[k * 3] * pose.center[0] +
            D[k * 3 + 1] * pose.center[1] +
            D[k * 3 + 2] * pose.center[2] +
            offset[k] +
            (random() - 0.5) * 0.001,
    );
    return { R, center };
}

/** A silent tracking jump: 4 cm and 1.5° of yaw applied to every later phone pose. */
export function jumped(pose: TruePose): TruePose {
    const yaw = (1.5 * Math.PI) / 180;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const D = [c, 0, s, 0, 1, 0, -s, 0, c];
    const R = [0, 1, 2].flatMap((r) =>
        [0, 1, 2].map(
            (col) =>
                pose.R[r * 3] * D[col * 3] +
                pose.R[r * 3 + 1] * D[col * 3 + 1] +
                pose.R[r * 3 + 2] * D[col * 3 + 2],
        ),
    );
    const center = [0, 1, 2].map(
        (k) =>
            D[k * 3] * pose.center[0] +
            D[k * 3 + 1] * pose.center[1] +
            D[k * 3 + 2] * pose.center[2] +
            [0.03, 0.01, -0.025][k],
    );
    return { R, center };
}

/**
 * Rigid transform (Horn's quaternion method) taking the reconstructed camera centres onto the
 * true ones, so that surface errors are measured without the arbitrary gauge of the odometry.
 */
export function alignment(from: number[][], to: number[][]): (p: number[]) => number[] {
    const mean = (points: number[][]) =>
        [0, 1, 2].map((k) => points.reduce((sum, p) => sum + p[k], 0) / points.length);
    const a = mean(from);
    const b = mean(to);
    const S = new Array(9).fill(0);
    from.forEach((p, i) => {
        for (let r = 0; r < 3; r++)
            for (let c = 0; c < 3; c++) S[r * 3 + c] += (p[r] - a[r]) * (to[i][c] - b[c]);
    });
    const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
    const N = [
        xx + yy + zz,
        yz - zy,
        zx - xz,
        xy - yx,
        yz - zy,
        xx - yy - zz,
        xy + yx,
        zx + xz,
        zx - xz,
        xy + yx,
        -xx + yy - zz,
        yz + zy,
        xy - yx,
        zx + xz,
        yz + zy,
        -xx - yy + zz,
    ];
    const { vectors } = symmetricEigen(N, 4);
    const [w, x, y, z] = vectors.slice(12, 16);
    const R = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ];
    return (p) =>
        [0, 1, 2].map(
            (r) =>
                R[r * 3] * (p[0] - a[0]) +
                R[r * 3 + 1] * (p[1] - a[1]) +
                R[r * 3 + 2] * (p[2] - a[2]) +
                b[r],
        );
}
