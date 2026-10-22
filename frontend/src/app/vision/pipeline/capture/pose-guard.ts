import {
    Pose,
    WorldTransform,
    cameraCenter,
    clonePose,
    correctPose,
    project,
    relativePose,
    worldCorrection,
} from '../../geometry/camera';
import { resect } from '../../geometry/resection';
import { bruteForceMatch } from '../../matching/brute-force-matcher';
import { fundamentalInliers } from '../../matching/fundamental-ransac';
import { MatchList } from '../../matching/match-list';
import { logSO3 } from '../../math/mat3';
import { median } from '../../math/dense';
import { SparseMap } from '../../sfm/sparse-map';

/** Guided match counts below which a consecutive pair is re-checked without the pose prior. */
const WEAK_LINK = 25;
/** Unguided inliers that prove the images overlap (so a weak guided link means a bad pose). */
const STRONG_LINK = 40;
/** Motion between consecutive keyframes faster than this is a tracking jump. */
const MAX_SPEED = 1.5;
const MAX_TURN = 3;
/** Median reprojection error (px) of known points above which the new pose has jumped. */
const JUMP_ERROR = 2.5;
const ANCHOR_POINTS = 15;
const HISTORY = 8;

/**
 * Decides, keyframe by keyframe, whether the phone pose can be trusted. Tracking losses and
 * implausible motion break the odometry link; a consecutive pair whose guided matching
 * collapsed is re-matched without the pose; and a pose that disagrees with the points the map
 * already has is re-anchored by resection, with the rest of its tracking segment following.
 */
export class PoseGuard {
    /** Correction of the current tracking segment (phone pose → map pose), after a jump. */
    private correction: WorldTransform | null = null;
    private lastSegment: number | undefined;
    /** Guided match counts of the recent consecutive pairs (to spot a sudden collapse). */
    private readonly linkHistory: number[] = [];
    /** Recent median reprojection errors of new frames against the map (to spot a spike). */
    private readonly anchorHistory: number[] = [];
    /** Pairs that needed unguided matching, and segments re-anchored by resection. */
    fallbacks = 0;
    realigned = 0;

    constructor(private readonly map: SparseMap) {}

    /** The map pose of a new keyframe: the phone pose, corrected if its segment was re-anchored. */
    initialPose(prior: Pose): Pose {
        return this.correction ? correctPose(prior, this.correction) : clonePose(prior);
    }

    /** A tracking loss reported by the phone, or a physically implausible jump. */
    isBreak(time: number, prior: Pose, segment: number | undefined): boolean {
        const changed = segment !== this.lastSegment;
        this.lastSegment = segment;
        const previous = this.map.frames.at(-1);
        if (!previous) return false;
        if (changed) return true;
        const dt = Math.max(0.05, time - previous.time);
        const motion = relativePose(previous.prior, prior);
        const distance = Math.hypot(...cameraCenter(motion));
        const turn = Math.hypot(...logSO3(motion.R));
        return distance / dt > MAX_SPEED || turn / dt > MAX_TURN;
    }

    /**
     * Checks the guided matches of keyframe `index` with the previous one and returns the
     * matches to link with: the guided ones, or ones re-matched without the pose if it was wrong.
     */
    checkConsecutive(index: number, previous: number, guided: MatchList): MatchList {
        // Weak: few matches in absolute terms, or far fewer than the recent pairs had (a wrong
        // relative pose pushes most true matches out of the epipolar band).
        const count = guided.length / 2;
        const usual = this.linkHistory.length >= 3 ? median(this.linkHistory) : 0;
        const weak = count < Math.max(WEAK_LINK, 0.35 * usual) || this.map.breaks.has(index);
        this.linkHistory.push(count);
        if (this.linkHistory.length > HISTORY) this.linkHistory.shift();
        const rescued = weak ? this.rescue(index, previous, count) : null;
        return this.anchor(index, previous, rescued ?? guided, rescued !== null);
    }

    /**
     * Matches a consecutive pair without trusting the poses (all descriptors + F-RANSAC). Returns
     * the matches when the images clearly overlap although guided matching failed — the phone
     * pose is wrong, so the odometry link becomes a break.
     */
    private rescue(index: number, previous: number, guided: number): MatchList | null {
        const unguided = this.unguided(index, previous);
        if (unguided.length / 2 < STRONG_LINK || unguided.length / 2 < 2 * guided) return null;
        this.fallbacks++;
        this.map.breaks.add(index);
        return unguided;
    }

    private unguided(index: number, previous: number): MatchList {
        const frame = this.map.frames[index];
        const other = this.map.frames[previous];
        return fundamentalInliers(
            frame.features,
            other.features,
            bruteForceMatch(frame.features, other.features),
            2,
            index,
        );
    }

    /**
     * Checks the new frame's pose against the points the map already has (seen through its
     * matches with the previous frame). If they reproject badly — a pose jump, even one along
     * the epipolar lines that guided matching cannot notice — the frame is re-anchored by
     * resection, the rest of its tracking segment follows, and the odometry link is broken.
     */
    private anchor(
        index: number,
        previous: number,
        matches: MatchList,
        rescued: boolean,
    ): MatchList {
        const frame = this.map.frames[index];
        const pairs = this.correspondences(index, previous, matches);
        if (!pairs) return matches;
        const { world, image } = pairs;
        const errors: number[] = [];
        for (let i = 0; i < world.length / 3; i++) {
            const [u, v, z] = project(
                frame.intrinsics,
                frame.pose,
                world.subarray(i * 3, i * 3 + 3),
            );
            errors.push(z > 0 ? Math.hypot(u - image[i * 2], v - image[i * 2 + 1]) : 1e3);
        }
        const before = median(errors);
        // A jump is a spike: errors far above what the recent frames had. A constant offset (for
        // example a slightly wrong focal length) is left to bundle adjustment.
        const usual = this.anchorHistory.length >= 3 ? median(this.anchorHistory) : Infinity;
        const spike = before > JUMP_ERROR && before > 3 * usual;
        if (!spike && !this.map.breaks.has(index)) {
            this.anchorHistory.push(before);
            if (this.anchorHistory.length > HISTORY) this.anchorHistory.shift();
            return matches;
        }
        // Guided matches made with a wrong pose are unreliable: re-match without it.
        const rematched = rescued ? matches : this.unguided(index, previous);
        const solid = this.correspondences(index, previous, rematched);
        if (!solid) return matches;
        const result = resect(
            frame.intrinsics,
            frame.pose,
            solid.world,
            solid.image,
            ANCHOR_POINTS,
        );
        // Accept the new pose if it brings the error back to what neighbouring frames have.
        const typical = Number.isFinite(usual) ? usual : 1;
        if (!result || result.rms > Math.max(1.5, 1.5 * typical) || result.rms > before / 2) {
            return matches;
        }
        this.map.breaks.add(index);
        this.correction = worldCorrection(frame.prior, result.pose);
        frame.pose.R.set(result.pose.R);
        frame.pose.t.set(result.pose.t);
        this.realigned++;
        return rematched;
    }

    /** 2D–3D pairs: features of the new frame matched to triangulated tracks of `previous`. */
    private correspondences(
        index: number,
        previous: number,
        matches: MatchList,
    ): { world: Float64Array; image: Float64Array } | null {
        const frame = this.map.frames[index];
        const other = this.map.frames[previous];
        const points: number[] = [];
        const uv: number[] = [];
        for (let m = 0; m < matches.length; m += 2) {
            const track = other.track[matches[m + 1]];
            const point = track >= 0 ? this.map.tracks[track].point : null;
            if (!point) continue;
            points.push(point[0], point[1], point[2]);
            uv.push(frame.features.x[matches[m]], frame.features.y[matches[m]]);
        }
        if (points.length < 3 * ANCHOR_POINTS) return null;
        return { world: Float64Array.from(points), image: Float64Array.from(uv) };
    }
}
