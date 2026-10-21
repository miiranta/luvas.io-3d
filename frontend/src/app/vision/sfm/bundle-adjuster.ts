import { Intrinsics, Pose, relativePose } from '../geometry/camera';
import { expSO3, invert3, logSO3, multiply3, transpose3 } from '../math/mat3';

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
}

export interface BundleReport {
    iterations: number;
    initialRms: number;
    finalRms: number;
}

const HUBER = 1.5;
const ROTATION_SIGMA = (0.25 * Math.PI) / 180;
const ROTATION_SIGMA_RELATIVE = 0.03;
const TRANSLATION_SIGMA = 0.002;
const TRANSLATION_SIGMA_RELATIVE = 0.03;
const GAUGE_SIGMA = 1e-5;

/**
 * Levenberg–Marquardt bundle adjustment of camera poses and points. Points are eliminated
 * with the Schur complement and the reduced camera system is solved with block-Jacobi
 * preconditioned conjugate gradients. The phone's own odometry (relative motion between
 * consecutive keyframes) enters as Gaussian priors, which keeps metric scale and gravity.
 */
export function bundleAdjust(problem: BundleProblem, maxIterations: number): BundleReport {
    const cameras = problem.poses.length;
    const points = problem.points.length / 3;
    const observations = problem.camera.length;
    const cameraParams = cameras * 6;

    const priors = buildPriors(problem);
    let lambda = 1e-3;
    let cost = totalCost(problem, problem.poses, problem.points, priors);
    const initialRms = rmsError(problem);

    const U = new Float64Array(cameraParams * cameraParams);
    const V = new Float64Array(points * 9);
    const Vinv = new Float64Array(points * 9);
    const W = new Float64Array(observations * 18);
    const bc = new Float64Array(cameraParams);
    const bp = new Float64Array(points * 3);

    let iteration = 0;
    for (; iteration < maxIterations; iteration++) {
        U.fill(0);
        V.fill(0);
        bc.fill(0);
        bp.fill(0);
        accumulateObservations(problem, U, V, W, bc, bp, cameraParams);
        for (const prior of priors) prior.accumulate(problem.poses, U, bc, cameraParams);

        let improved = false;
        for (let attempt = 0; attempt < 6 && !improved; attempt++) {
            const damped = new Float64Array(U);
            for (let i = 0; i < cameraParams; i++) {
                damped[i * cameraParams + i] *= 1 + lambda;
                damped[i * cameraParams + i] += 1e-9;
            }
            const valid = new Uint8Array(points);
            for (let p = 0; p < points; p++) {
                const block = V.slice(p * 9, p * 9 + 9);
                block[0] *= 1 + lambda;
                block[4] *= 1 + lambda;
                block[8] *= 1 + lambda;
                if (invert3(block, Vinv.subarray(p * 9, p * 9 + 9))) valid[p] = 1;
            }
            const { S, rhs } = reduceCameraSystem(problem, damped, W, Vinv, valid, bc, bp);
            const deltaC = conjugateGradients(S, rhs, cameraParams);
            const deltaP = backSubstitute(problem, W, Vinv, valid, bp, deltaC);

            const poses = problem.poses.map((pose, c) => applyPoseDelta(pose, deltaC, c * 6));
            const updated = new Float64Array(problem.points);
            for (let i = 0; i < updated.length; i++) updated[i] += deltaP[i];
            const nextCost = totalCost(problem, poses, updated, priors);
            if (nextCost < cost) {
                const relative = (cost - nextCost) / cost;
                cost = nextCost;
                problem.poses.forEach((pose, c) => {
                    pose.R.set(poses[c].R);
                    pose.t.set(poses[c].t);
                });
                problem.points.set(updated);
                lambda = Math.max(1e-7, lambda / 3);
                improved = true;
                if (relative < 1e-5) iteration = maxIterations;
            } else {
                lambda *= 4;
            }
        }
        if (!improved) break;
    }
    return {
        iterations: Math.min(iteration, maxIterations),
        initialRms,
        finalRms: rmsError(problem),
    };
}

function accumulateObservations(
    problem: BundleProblem,
    U: Float64Array,
    V: Float64Array,
    W: Float64Array,
    bc: Float64Array,
    bp: Float64Array,
    n: number,
): void {
    const jc = new Float64Array(12);
    const jp = new Float64Array(6);
    for (let o = 0; o < problem.camera.length; o++) {
        const c = problem.camera[o];
        const p = problem.point[o];
        const k = problem.intrinsics[c];
        const { R, t } = problem.poses[c];
        const X = problem.points.subarray(p * 3, p * 3 + 3);
        const rx = R[0] * X[0] + R[1] * X[1] + R[2] * X[2];
        const ry = R[3] * X[0] + R[4] * X[1] + R[5] * X[2];
        const rz = R[6] * X[0] + R[7] * X[1] + R[8] * X[2];
        const x = rx + t[0];
        const y = ry + t[1];
        const z = rz + t[2];
        if (z <= 1e-4) {
            W.fill(0, o * 18, o * 18 + 18);
            continue;
        }
        const iz = 1 / z;
        const ru = k.fx * x * iz + k.cx - problem.u[o];
        const rv = k.fy * y * iz + k.cy - problem.v[o];
        const norm = Math.hypot(ru, rv);
        const weight = norm <= HUBER ? 1 : HUBER / norm;
        // Projection Jacobian rows.
        const a0 = k.fx * iz;
        const a2 = -k.fx * x * iz * iz;
        const b1 = k.fy * iz;
        const b2 = -k.fy * y * iz * iz;
        // d x_cam / d ω = −[q]× with q = R X (left update R ← exp(ω) R).
        // Row u: [a0, 0, a2] · (−[q]×); row v: [0, b1, b2] · (−[q]×).
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
        for (let j = 0; j < 3; j++) {
            jp[j] = a0 * R[j] + a2 * R[6 + j];
            jp[3 + j] = b1 * R[3 + j] + b2 * R[6 + j];
        }
        const base = c * 6;
        for (let r = 0; r < 6; r++) {
            const ur = jc[r];
            const vr = jc[6 + r];
            bc[base + r] -= weight * (ur * ru + vr * rv);
            const row = (base + r) * n + base;
            for (let s = 0; s < 6; s++) U[row + s] += weight * (ur * jc[s] + vr * jc[6 + s]);
            for (let s = 0; s < 3; s++) {
                W[o * 18 + r * 3 + s] = weight * (ur * jp[s] + vr * jp[3 + s]);
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
function reduceCameraSystem(
    problem: BundleProblem,
    U: Float64Array,
    W: Float64Array,
    Vinv: Float64Array,
    valid: Uint8Array,
    bc: Float64Array,
    bp: Float64Array,
): { S: Float64Array; rhs: Float64Array } {
    const n = bc.length;
    const S = U;
    const rhs = new Float64Array(bc);
    const byPoint = groupByPoint(problem);
    const wv = new Float64Array(18);
    for (let p = 0; p < byPoint.offsets.length - 1; p++) {
        if (!valid[p]) continue;
        const vi = Vinv.subarray(p * 9, p * 9 + 9);
        const start = byPoint.offsets[p];
        const end = byPoint.offsets[p + 1];
        for (let a = start; a < end; a++) {
            const oa = byPoint.order[a];
            const ca = problem.camera[oa];
            for (let r = 0; r < 6; r++) {
                for (let s = 0; s < 3; s++) {
                    wv[r * 3 + s] =
                        W[oa * 18 + r * 3] * vi[s] +
                        W[oa * 18 + r * 3 + 1] * vi[3 + s] +
                        W[oa * 18 + r * 3 + 2] * vi[6 + s];
                }
                rhs[ca * 6 + r] -=
                    wv[r * 3] * bp[p * 3] +
                    wv[r * 3 + 1] * bp[p * 3 + 1] +
                    wv[r * 3 + 2] * bp[p * 3 + 2];
            }
            for (let b = start; b < end; b++) {
                const ob = byPoint.order[b];
                const cb = problem.camera[ob];
                for (let r = 0; r < 6; r++) {
                    const row = (ca * 6 + r) * n + cb * 6;
                    for (let s = 0; s < 6; s++) {
                        S[row + s] -=
                            wv[r * 3] * W[ob * 18 + s * 3] +
                            wv[r * 3 + 1] * W[ob * 18 + s * 3 + 1] +
                            wv[r * 3 + 2] * W[ob * 18 + s * 3 + 2];
                    }
                }
            }
        }
    }
    return { S, rhs };
}

function backSubstitute(
    problem: BundleProblem,
    W: Float64Array,
    Vinv: Float64Array,
    valid: Uint8Array,
    bp: Float64Array,
    deltaC: Float64Array,
): Float64Array {
    const residual = new Float64Array(bp);
    for (let o = 0; o < problem.camera.length; o++) {
        const c = problem.camera[o] * 6;
        const p = problem.point[o] * 3;
        for (let s = 0; s < 3; s++) {
            let sum = 0;
            for (let r = 0; r < 6; r++) sum += W[o * 18 + r * 3 + s] * deltaC[c + r];
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

/** Block-Jacobi (6×6) preconditioned conjugate gradients on the dense camera system. */
function conjugateGradients(S: Float64Array, b: Float64Array, n: number): Float64Array {
    const blocks = n / 6;
    const preconditioner = new Float64Array(blocks * 36);
    for (let c = 0; c < blocks; c++) {
        const block = new Float64Array(36);
        for (let r = 0; r < 6; r++) {
            for (let s = 0; s < 6; s++) block[r * 6 + s] = S[(c * 6 + r) * n + c * 6 + s];
        }
        preconditioner.set(invertSpd6(block), c * 36);
    }
    const applyM = (v: Float64Array, out: Float64Array) => {
        for (let c = 0; c < blocks; c++) {
            for (let r = 0; r < 6; r++) {
                let sum = 0;
                for (let s = 0; s < 6; s++)
                    sum += preconditioner[c * 36 + r * 6 + s] * v[c * 6 + s];
                out[c * 6 + r] = sum;
            }
        }
    };
    const x = new Float64Array(n);
    const r = new Float64Array(b);
    const z = new Float64Array(n);
    applyM(r, z);
    const p = new Float64Array(z);
    const q = new Float64Array(n);
    let rz = dot(r, z);
    const target = 1e-10 * dot(b, b);
    for (let iteration = 0; iteration < 300 && dot(r, r) > target; iteration++) {
        for (let i = 0; i < n; i++) {
            let sum = 0;
            const row = i * n;
            for (let j = 0; j < n; j++) sum += S[row + j] * p[j];
            q[i] = sum;
        }
        const alpha = rz / (dot(p, q) || 1e-300);
        for (let i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        applyM(r, z);
        const next = dot(r, z);
        const beta = next / (rz || 1e-300);
        rz = next;
        for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    return x;
}

function invertSpd6(block: Float64Array): Float64Array {
    // Gauss–Jordan with partial pivoting; falls back to the diagonal when singular.
    const a = new Float64Array(block);
    const inv = new Float64Array(36);
    for (let i = 0; i < 6; i++) inv[i * 6 + i] = 1;
    for (let col = 0; col < 6; col++) {
        let pivot = col;
        for (let row = col + 1; row < 6; row++) {
            if (Math.abs(a[row * 6 + col]) > Math.abs(a[pivot * 6 + col])) pivot = row;
        }
        const value = a[pivot * 6 + col];
        if (Math.abs(value) < 1e-30) {
            const diagonal = new Float64Array(36);
            for (let i = 0; i < 6; i++) diagonal[i * 6 + i] = 1 / Math.max(1e-30, block[i * 6 + i]);
            return diagonal;
        }
        for (let k = 0; k < 6; k++) {
            [a[col * 6 + k], a[pivot * 6 + k]] = [a[pivot * 6 + k], a[col * 6 + k]];
            [inv[col * 6 + k], inv[pivot * 6 + k]] = [inv[pivot * 6 + k], inv[col * 6 + k]];
        }
        for (let k = 0; k < 6; k++) {
            a[col * 6 + k] /= value;
            inv[col * 6 + k] /= value;
        }
        for (let row = 0; row < 6; row++) {
            if (row === col) continue;
            const factor = a[row * 6 + col];
            if (factor === 0) continue;
            for (let k = 0; k < 6; k++) {
                a[row * 6 + k] -= factor * a[col * 6 + k];
                inv[row * 6 + k] -= factor * inv[col * 6 + k];
            }
        }
    }
    return inv;
}

function dot(a: Float64Array, b: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

function applyPoseDelta(pose: Pose, delta: Float64Array, offset: number): Pose {
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

interface PriorTerm {
    residual(poses: Pose[]): Float64Array;
    accumulate(poses: Pose[], U: Float64Array, bc: Float64Array, n: number): void;
}

/** Relative-motion factors between consecutive keyframes plus a gauge anchor on the first. */
function buildPriors(problem: BundleProblem): PriorTerm[] {
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
            residual,
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
            residual: (poses) => residualOf(poses[i], poses[j]),
            accumulate: (poses, U, bc, n) => {
                const r0 = residualOf(poses[i], poses[j]);
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
                    bc[index(a)] -= g;
                    for (let b = 0; b < 12; b++) {
                        let h = 0;
                        for (let row = 0; row < 6; row++) h += J[row * 12 + a] * J[row * 12 + b];
                        U[index(a) * n + index(b)] += h;
                    }
                }
            },
        });
    }
    return terms;
}

function totalCost(
    problem: BundleProblem,
    poses: Pose[],
    points: Float64Array,
    priors: PriorTerm[],
): number {
    let cost = 0;
    for (let o = 0; o < problem.camera.length; o++) {
        const error = observationError(problem, poses, points, o);
        cost += error <= HUBER ? error * error : 2 * HUBER * error - HUBER * HUBER;
    }
    for (const prior of priors) {
        for (const value of prior.residual(poses)) cost += value * value;
    }
    return cost;
}

function observationError(
    problem: BundleProblem,
    poses: Pose[],
    points: Float64Array,
    o: number,
): number {
    const k = problem.intrinsics[problem.camera[o]];
    const { R, t } = poses[problem.camera[o]];
    const p = problem.point[o] * 3;
    const x = R[0] * points[p] + R[1] * points[p + 1] + R[2] * points[p + 2] + t[0];
    const y = R[3] * points[p] + R[4] * points[p + 1] + R[5] * points[p + 2] + t[1];
    const z = R[6] * points[p] + R[7] * points[p + 1] + R[8] * points[p + 2] + t[2];
    if (z <= 1e-4) return 1e3;
    return Math.hypot((k.fx * x) / z + k.cx - problem.u[o], (k.fy * y) / z + k.cy - problem.v[o]);
}

export function observationErrors(problem: BundleProblem): Float64Array {
    const errors = new Float64Array(problem.camera.length);
    for (let o = 0; o < errors.length; o++) {
        errors[o] = observationError(problem, problem.poses, problem.points, o);
    }
    return errors;
}

function rmsError(problem: BundleProblem): number {
    const errors = observationErrors(problem);
    let sum = 0;
    for (const e of errors) sum += Math.min(e, 10) ** 2;
    return Math.sqrt(sum / Math.max(1, errors.length));
}

function groupByPoint(problem: BundleProblem): { offsets: Int32Array; order: Int32Array } {
    const points = problem.points.length / 3;
    const offsets = new Int32Array(points + 1);
    for (const p of problem.point) offsets[p + 1]++;
    for (let p = 0; p < points; p++) offsets[p + 1] += offsets[p];
    const cursor = offsets.slice(0, -1);
    const order = new Int32Array(problem.point.length);
    for (let o = 0; o < problem.point.length; o++) order[cursor[problem.point[o]]++] = o;
    return { offsets, order };
}
