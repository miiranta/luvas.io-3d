import { BundleProblem, Layout, parameter } from './problem';

/** Huber threshold (px) on the reprojection error. */
export const HUBER = 1.5;

/** Gauss–Newton normal equations of the reprojection terms, split for the Schur complement. */
export function accumulateObservations(
    problem: BundleProblem,
    scale: number,
    layout: Layout,
    U: Float64Array,
    V: Float64Array,
    W: Float64Array,
    bc: Float64Array,
    bp: Float64Array,
): void {
    const { n, rows } = layout;
    const jc = new Float64Array(14);
    const jp = new Float64Array(6);
    const index = new Int32Array(7);
    for (let o = 0; o < problem.camera.length; o++) {
        const c = problem.camera[o];
        const p = problem.point[o];
        const k = problem.intrinsics[c];
        const fx = k.fx * scale;
        const fy = k.fy * scale;
        const { R, t } = problem.poses[c];
        const X = problem.points.subarray(p * 3, p * 3 + 3);
        const rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2];
        const ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2];
        const rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
        const x = rx + t[0];
        const y = ry + t[1];
        const z = rz + t[2];
        if (z <= 1e-4) {
            W.fill(0, o * rows * 3, (o + 1) * rows * 3);
            continue;
        }
        const iz = 1 / z;
        const ru = fx * x * iz + k.cx - problem.u[o];
        const rv = fy * y * iz + k.cy - problem.v[o];
        const norm = Math.hypot(ru, rv);
        const weight = norm <= HUBER ? 1 : HUBER / norm;
        // Projection Jacobian rows.
        const a0 = fx * iz;
        const a2 = -fx * x * iz * iz;
        const b1 = fy * iz;
        const b2 = -fy * y * iz * iz;
        // d x_cam / d ω = −[q]× with q = R X (left update R ← exp(ω) R).
        // Row u: [a0, 0, a2] · (−[q]×); row v: [0, b1, b2] · (−[q]×). Columns 0–5 per camera,
        // column 6 = d/d(focal scale) = (fx₀ x/z, fy₀ y/z).
        jc[0] = a2 * ry;
        jc[1] = a0 * rz - a2 * rx;
        jc[2] = -a0 * ry;
        jc[3] = a0;
        jc[4] = 0;
        jc[5] = a2;
        jc[6] = k.fx * x * iz;
        jc[7] = b2 * ry - b1 * rz;
        jc[8] = -b2 * rx;
        jc[9] = b1 * rx;
        jc[10] = 0;
        jc[11] = b1;
        jc[12] = b2;
        jc[13] = k.fy * y * iz;
        for (let j = 0; j < 3; j++) {
            jp[j] = a0 * R[j] + a2 * R[6 + j];
            jp[3 + j] = b1 * R[3 + j] + b2 * R[6 + j];
        }
        for (let r = 0; r < rows; r++) index[r] = parameter(layout, c, r);
        for (let r = 0; r < rows; r++) {
            const ur = jc[r];
            const vr = jc[7 + r];
            bc[index[r]] -= weight * (ur * ru + vr * rv);
            const row = index[r] * n;
            for (let s = 0; s < rows; s++)
                U[row + index[s]] += weight * (ur * jc[s] + vr * jc[7 + s]);
            for (let s = 0; s < 3; s++) {
                W[(o * rows + r) * 3 + s] = weight * (ur * jp[s] + vr * jp[3 + s]);
            }
        }
        for (let r = 0; r < 3; r++) {
            bp[p * 3 + r] -= weight * (jp[r] * ru + jp[3 + r] * rv);
            for (let s = 0; s < 3; s++) {
                V[p * 9 + r * 3 + s] += weight * (jp[r] * jp[s] + jp[3 + r] * jp[3 + s]);
            }
        }
    }
}

/** S = U − Σ W V⁻¹ Wᵀ and rhs = bc − Σ W V⁻¹ bp, grouping observations by point. */
export function reduceCameraSystem(
    problem: BundleProblem,
    layout: Layout,
    byPoint: { offsets: Int32Array; order: Int32Array },
    U: Float64Array,
    W: Float64Array,
    Vinv: Float64Array,
    valid: Uint8Array,
    bc: Float64Array,
    bp: Float64Array,
): { S: Float64Array; rhs: Float64Array } {
    const { n, rows } = layout;
    const S = U;
    const rhs = new Float64Array(bc);
    const wv = new Float64Array(rows * 3);
    const rowIndex = new Int32Array(rows);
    const columnIndex = new Int32Array(rows);
    for (let p = 0; p < byPoint.offsets.length - 1; p++) {
        if (!valid[p]) continue;
        const vi = Vinv.subarray(p * 9, p * 9 + 9);
        const start = byPoint.offsets[p];
        const end = byPoint.offsets[p + 1];
        for (let a = start; a < end; a++) {
            const oa = byPoint.order[a];
            const ca = problem.camera[oa];
            for (let r = 0; r < rows; r++) {
                rowIndex[r] = parameter(layout, ca, r);
                const w = (oa * rows + r) * 3;
                for (let s = 0; s < 3; s++) {
                    wv[r * 3 + s] = W[w] * vi[s] + W[w + 1] * vi[3 + s] + W[w + 2] * vi[6 + s];
                }
                rhs[rowIndex[r]] -=
                    wv[r * 3] * bp[p * 3] +
                    wv[r * 3 + 1] * bp[p * 3 + 1] +
                    wv[r * 3 + 2] * bp[p * 3 + 2];
            }
            for (let b = start; b < end; b++) {
                const ob = byPoint.order[b];
                const cb = problem.camera[ob];
                for (let s = 0; s < rows; s++) columnIndex[s] = parameter(layout, cb, s);
                for (let r = 0; r < rows; r++) {
                    const row = rowIndex[r] * n;
                    for (let s = 0; s < rows; s++) {
                        const w = (ob * rows + s) * 3;
                        S[row + columnIndex[s]] -=
                            wv[r * 3] * W[w] + wv[r * 3 + 1] * W[w + 1] + wv[r * 3 + 2] * W[w + 2];
                    }
                }
            }
        }
    }
    return { S, rhs };
}

export function backSubstitute(
    problem: BundleProblem,
    layout: Layout,
    W: Float64Array,
    Vinv: Float64Array,
    valid: Uint8Array,
    bp: Float64Array,
    deltaC: Float64Array,
): Float64Array {
    const { rows } = layout;
    const residual = new Float64Array(bp);
    for (let o = 0; o < problem.camera.length; o++) {
        const c = problem.camera[o];
        const p = problem.point[o] * 3;
        for (let s = 0; s < 3; s++) {
            let sum = 0;
            for (let r = 0; r < rows; r++) {
                sum += W[(o * rows + r) * 3 + s] * deltaC[parameter(layout, c, r)];
            }
            residual[p + s] -= sum;
        }
    }
    const delta = new Float64Array(bp.length);
    for (let p = 0; p < bp.length / 3; p++) {
        if (!valid[p]) continue;
        const vi = Vinv.subarray(p * 9, p * 9 + 9);
        for (let r = 0; r < 3; r++) {
            delta[p * 3 + r] =
                vi[r * 3] * residual[p * 3] +
                vi[r * 3 + 1] * residual[p * 3 + 1] +
                vi[r * 3 + 2] * residual[p * 3 + 2];
        }
    }
    return delta;
}
