import { QualityProfile } from '../../../core/models/quality';
import { FEATURE_METHODS } from '../../features/methods';
import { bundleAdjust } from '../../sfm/bundle-adjuster';
import { SparseMap } from '../../sfm/sparse-map';
import { ScanSession } from '../scan-session';
import { buildFeatureMap } from './feature-map';
import { Progress } from './stage-runner';

export interface PoseRefinement {
    /** The map the poses were refined on (the live one, or one with re-extracted features). */
    map: SparseMap;
    iterations: number;
    /** Product of the focal corrections applied to every keyframe (1 = unchanged). */
    focalScale: number;
}

/** Bundle rounds: LM iterations, then the outlier limit (px) applied after the round. */
const ROUNDS: [number, number][] = [
    [15, 6],
    [12, 4],
    [10, 2.5],
];

/**
 * Bundle adjustment of the keyframe poses. With the live (ORB) features it runs directly on the
 * map built while recording. With another feature method, one coarse round on the live map
 * first removes the drift (and the focal error), then the features are re-extracted, matched
 * along tight epipolar bands of those poses and adjusted again; the refined poses are copied
 * back to the live map, which the later stages read.
 */
export async function refinePoses(
    session: ScanSession,
    settings: QualityProfile['bundle'],
    device: GPUDevice | null,
    progress: Progress,
): Promise<PoseRefinement> {
    const live = session.map;
    const rounds = ROUNDS.slice(ROUNDS.length - settings.rounds);
    if (settings.features === 'orb') {
        const result = adjust(live, rounds, settings, (f, d) => progress(f, d));
        return { map: live, ...result };
    }
    const first = adjust(live, ROUNDS.slice(0, 1), settings, (f) => progress(f * 0.1, 'ORB'));
    const map = await buildFeatureMap(session, FEATURE_METHODS[settings.features], device, (f, d) =>
        progress(0.1 + f * 0.6, d),
    );
    // The live round already removed the gross outliers: one round fewer on the new map.
    const second = adjust(map, rounds.slice(1), settings, (f, d) => progress(0.7 + f * 0.3, d));
    map.frames.forEach((frame, i) => {
        live.frames[i].pose.R.set(frame.pose.R);
        live.frames[i].pose.t.set(frame.pose.t);
    });
    return {
        map,
        iterations: first.iterations + second.iterations,
        focalScale: first.focalScale * second.focalScale,
    };
}

/**
 * Rounds of decreasing outlier tolerance. Before the first, tracks that drift broke (typically
 * loop closures) get a point from their consistent part, so the loops constrain the poses. A
 * self-calibrated focal is applied to the keyframe intrinsics in place.
 */
function adjust(
    map: SparseMap,
    rounds: [number, number][],
    settings: QualityProfile['bundle'],
    progress: Progress,
): { iterations: number; focalScale: number } {
    const all = map.tracks.map((_, id) => id);
    map.seedPoints();
    let iterations = 0;
    let focalScale = 1;
    rounds.forEach(([steps, outliers], round) => {
        progress(round / rounds.length, `round ${round + 1}/${rounds.length}`);
        const { problem } = map.bundleProblem(settings.points);
        if (problem.points.length >= 30 * 3) {
            const result = bundleAdjust(problem, {
                iterations: steps,
                calibrateFocal: settings.calibrateFocal,
            });
            iterations += result.iterations;
            if (result.focalScale !== 1) {
                focalScale *= result.focalScale;
                for (const frame of map.frames) {
                    frame.intrinsics.fx *= result.focalScale;
                    frame.intrinsics.fy *= result.focalScale;
                }
            }
        }
        map.triangulate(all, outliers);
        map.pruneObservations(outliers);
        map.triangulate(all, Math.min(outliers, 3));
    });
    return { iterations, focalScale };
}
