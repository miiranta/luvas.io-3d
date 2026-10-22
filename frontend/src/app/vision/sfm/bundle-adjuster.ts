import { invert3 } from '../math/mat3';
import { blockPcg } from '../math/block-pcg';
import {
    HUBER,
    accumulateObservations,
    backSubstitute,
    reduceCameraSystem,
} from './bundle/normal-equations';
import { PriorTerm, buildPriors } from './bundle/odometry-priors';
import {
    BundleOptions,
    BundleProblem,
    BundleReport,
    Layout,
    applyPoseDelta,
    groupByPoint,
    observationError,
    rmsError,
} from './bundle/problem';
import { Pose } from '../geometry/camera';

export type { BundleOptions, BundleProblem, BundleReport } from './bundle/problem';
export { observationErrors } from './bundle/problem';

/** Weak prior keeping the self-calibrated focal near the phone's value. */
const FOCAL_SIGMA = 0.05;

/**
 * Levenberg–Marquardt bundle adjustment of camera poses and points (and optionally one shared
 * focal scale). Points are eliminated with the Schur complement and the reduced camera system
 * is solved with block-Jacobi preconditioned conjugate gradients. The phone's own odometry
 * (relative motion between consecutive keyframes) enters as robust (Huber) priors, which keeps
 * metric scale and gravity without letting a tracking jump drag the solution.
 */
export function bundleAdjust(problem: BundleProblem, options: BundleOptions): BundleReport {
    const cameras = problem.poses.length;
    const points = problem.points.length / 3;
    const calibrate = options.calibrateFocal ?? false;
    const n = cameras * 6 + (calibrate ? 1 : 0);
    const layout: Layout = { n, rows: calibrate ? 7 : 6, focal: calibrate ? n - 1 : -1 };
    let scale = options.focalScale ?? 1;

    const priors = buildPriors(problem);
    let lambda = 1e-3;
    let cost = totalCost(problem, problem.poses, problem.points, scale, priors, calibrate);
    const initialRms = rmsError(problem, scale);

    const U = new Float64Array(n * n);
    const V = new Float64Array(points * 9);
    const Vinv = new Float64Array(points * 9);
    const W = new Float64Array(problem.camera.length * layout.rows * 3);
    const bc = new Float64Array(n);
    const bp = new Float64Array(points * 3);
    const byPoint = groupByPoint(problem);

    let iteration = 0;
    for (; iteration < options.iterations; iteration++) {
        U.fill(0);
        V.fill(0);
        bc.fill(0);
        bp.fill(0);
        accumulateObservations(problem, scale, layout, U, V, W, bc, bp);
        for (const prior of priors) prior.accumulate(problem.poses, U, bc, n);
        if (calibrate) {
            U[layout.focal * n + layout.focal] += 1 / (FOCAL_SIGMA * FOCAL_SIGMA);
            bc[layout.focal] -= (scale - 1) / (FOCAL_SIGMA * FOCAL_SIGMA);
        }

        let improved = false;
        for (let attempt = 0; attempt < 6 && !improved; attempt++) {
            const damped = new Float64Array(U);
            for (let i = 0; i < n; i++) {
                damped[i * n + i] *= 1 + lambda;
                damped[i * n + i] += 1e-9;
            }
            const valid = new Uint8Array(points);
            for (let p = 0; p < points; p++) {
                const block = V.slice(p * 9, p * 9 + 9);
                block[0] *= 1 + lambda;
                block[4] *= 1 + lambda;
                block[8] *= 1 + lambda;
                if (invert3(block, Vinv.subarray(p * 9, p * 9 + 9))) valid[p] = 1;
            }
            const { S, rhs } = reduceCameraSystem(
                problem,
                layout,
                byPoint,
                damped,
                W,
                Vinv,
                valid,
                bc,
                bp,
            );
            const blocks = Array.from({ length: cameras }, (_, c) => ({ offset: c * 6, size: 6 }));
            if (calibrate) blocks.push({ offset: layout.focal, size: 1 });
            const deltaC = blockPcg(S, rhs, n, blocks);
            const deltaP = backSubstitute(problem, layout, W, Vinv, valid, bp, deltaC);

            const poses = problem.poses.map((pose, c) => applyPoseDelta(pose, deltaC, c * 6));
            const updated = new Float64Array(problem.points);
            for (let i = 0; i < updated.length; i++) updated[i] += deltaP[i];
            const nextScale = calibrate ? scale + deltaC[layout.focal] : scale;
            const nextCost = totalCost(problem, poses, updated, nextScale, priors, calibrate);
            if (nextCost < cost) {
                const relative = (cost - nextCost) / cost;
                cost = nextCost;
                problem.poses.forEach((pose, c) => {
                    pose.R.set(poses[c].R);
                    pose.t.set(poses[c].t);
                });
                problem.points.set(updated);
                scale = nextScale;
                lambda = Math.max(1e-7, lambda / 3);
                improved = true;
                if (relative < 1e-5) iteration = options.iterations;
            } else {
                lambda *= 4;
            }
        }
        if (!improved) break;
    }
    return {
        iterations: Math.min(iteration, options.iterations),
        initialRms,
        finalRms: rmsError(problem, scale),
        focalScale: scale,
    };
}

function totalCost(
    problem: BundleProblem,
    poses: Pose[],
    points: Float64Array,
    scale: number,
    priors: PriorTerm[],
    calibrate: boolean,
): number {
    let cost = 0;
    for (let o = 0; o < problem.camera.length; o++) {
        const error = observationError(problem, poses, points, scale, o);
        cost += error <= HUBER ? error * error : 2 * HUBER * error - HUBER * HUBER;
    }
    for (const prior of priors) cost += prior.cost(poses);
    if (calibrate) cost += ((scale - 1) / FOCAL_SIGMA) ** 2;
    return cost;
}
