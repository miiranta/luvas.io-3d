import { Pose, relativePose } from '../../geometry/camera';
import { logSO3, multiply3, transpose3 } from '../../math/mat3';
import { BundleProblem, applyPoseDelta } from './problem';

/** Odometry residuals beyond this many sigmas count linearly (robust to pose jumps). */
const PRIOR_HUBER = 3;
const ROTATION_SIGMA = (0.25 * Math.PI) / 180;
const ROTATION_SIGMA_RELATIVE = 0.03;
const TRANSLATION_SIGMA = 0.002;
const TRANSLATION_SIGMA_RELATIVE = 0.03;
const GAUGE_SIGMA = 1e-5;

export interface PriorTerm {
    /** Robust cost contribution of the term. */
    cost(poses: Pose[]): number;
    accumulate(poses: Pose[], U: Float64Array, bc: Float64Array, n: number): void;
}

/** Huber on a whitened residual vector: cost and IRLS weight. */
function huber(r: Float64Array, delta: number): { cost: number; weight: number } {
    let squared = 0;
    for (const value of r) squared += value * value;
    const norm = Math.sqrt(squared);
    if (norm <= delta) return { cost: squared, weight: 1 };
    return { cost: 2 * delta * norm - delta * delta, weight: delta / norm };
}

/**
 * Relative-motion factors between consecutive keyframes (skipped across breaks, Huber-robust)
 * plus a gauge anchor on the first keyframe.
 */
export function buildPriors(problem: BundleProblem): PriorTerm[] {
    const terms: PriorTerm[] = [];
    const anchor = problem.priors[0];
    if (anchor) {
        const residual = (poses: Pose[]) => {
            const r = logSO3(multiply3(poses[0].R, transpose3(anchor.R)));
            return new Float64Array([
                r[0] / GAUGE_SIGMA,
                r[1] / GAUGE_SIGMA,
                r[2] / GAUGE_SIGMA,
                (poses[0].t[0] - anchor.t[0]) / GAUGE_SIGMA,
                (poses[0].t[1] - anchor.t[1]) / GAUGE_SIGMA,
                (poses[0].t[2] - anchor.t[2]) / GAUGE_SIGMA,
            ]);
        };
        terms.push({
            cost: (poses) => residual(poses).reduce((sum, value) => sum + value * value, 0),
            accumulate: (poses, U, bc, n) => {
                const r = residual(poses);
                // The gauge term is linear enough near the anchor to use J = I / σ.
                for (let i = 0; i < 6; i++) {
                    U[i * n + i] += 1 / (GAUGE_SIGMA * GAUGE_SIGMA);
                    bc[i] -= r[i] / GAUGE_SIGMA;
                }
            },
        });
    }
    for (let i = 0; i + 1 < problem.priors.length; i++) {
        const j = i + 1;
        if (problem.breaks?.has(j)) continue;
        const measured = relativePose(problem.priors[i], problem.priors[j]);
        const distance = Math.hypot(...measured.t);
        const angle = Math.hypot(...logSO3(measured.R));
        const sigmaR = ROTATION_SIGMA + ROTATION_SIGMA_RELATIVE * angle;
        const sigmaT = TRANSLATION_SIGMA + TRANSLATION_SIGMA_RELATIVE * distance;
        const residualOf = (a: Pose, b: Pose) => {
            const estimate = relativePose(a, b);
            const r = logSO3(multiply3(estimate.R, transpose3(measured.R)));
            return new Float64Array([
                r[0] / sigmaR,
                r[1] / sigmaR,
                r[2] / sigmaR,
                (estimate.t[0] - measured.t[0]) / sigmaT,
                (estimate.t[1] - measured.t[1]) / sigmaT,
                (estimate.t[2] - measured.t[2]) / sigmaT,
            ]);
        };
        terms.push({
            cost: (poses) => huber(residualOf(poses[i], poses[j]), PRIOR_HUBER).cost,
            accumulate: (poses, U, bc, n) => {
                const r0 = residualOf(poses[i], poses[j]);
                const { weight } = huber(r0, PRIOR_HUBER);
                // Numeric Jacobian (12 columns) by central differences.
                const J = new Float64Array(6 * 12);
                const step = 1e-6;
                const delta = new Float64Array(6);
                for (let col = 0; col < 12; col++) {
                    delta.fill(0);
                    delta[col % 6] = step;
                    const plusA = col < 6 ? applyPoseDelta(poses[i], delta, 0) : poses[i];
                    const plusB = col < 6 ? poses[j] : applyPoseDelta(poses[j], delta, 0);
                    delta[col % 6] = -step;
                    const minusA = col < 6 ? applyPoseDelta(poses[i], delta, 0) : poses[i];
                    const minusB = col < 6 ? poses[j] : applyPoseDelta(poses[j], delta, 0);
                    const rp = residualOf(plusA, plusB);
                    const rm = residualOf(minusA, minusB);
                    for (let row = 0; row < 6; row++) {
                        J[row * 12 + col] = (rp[row] - rm[row]) / (2 * step);
                    }
                }
                const index = (col: number) => (col < 6 ? i * 6 + col : j * 6 + col - 6);
                for (let a = 0; a < 12; a++) {
                    let g = 0;
                    for (let row = 0; row < 6; row++) g += J[row * 12 + a] * r0[row];
                    bc[index(a)] -= weight * g;
                    for (let b = 0; b < 12; b++) {
                        let h = 0;
                        for (let row = 0; row < 6; row++) h += J[row * 12 + a] * J[row * 12 + b];
                        U[index(a) * n + index(b)] += weight * h;
                    }
                }
            },
        });
    }
    return terms;
}
