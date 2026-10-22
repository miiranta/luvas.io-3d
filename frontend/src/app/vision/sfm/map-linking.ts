import { fundamentalInliers } from '../matching/fundamental-ransac';
import { guidedMatch } from '../matching/guided-matcher';
import { MatchList } from '../matching/match-list';
import { SparseMap, pairKey } from './sparse-map';

/** Epipolar bands (px): consecutive keyframes trust the odometry; loop pairs allow drift. */
export const TIGHT_BAND = 4;
const LOOSE_BAND = 40;
const LOOSE_AFTER_SECONDS = 5;
const RECENT_LINKS = 3;
const LINK_LIMIT = 7;
/** Matches below which a pair is not linked. */
export const MIN_MATCHES = 12;

/** Depths searched along a ray when nothing better is known. */
const ANY_DEPTH: [number, number] = [0.05, 15];

/**
 * Guided matching of frames i and j of the map along the epipolar lines of their poses, for
 * points of frame i within the depth range.
 */
export function matchPair(
    map: SparseMap,
    i: number,
    j: number,
    band: number,
    range = ANY_DEPTH,
): MatchList {
    const a = map.frames[i];
    const b = map.frames[j];
    return guidedMatch(
        a.features,
        b.features,
        a.intrinsics,
        b.intrinsics,
        a.pose,
        b.pose,
        b.width,
        b.height,
        { band, minDepth: range[0], maxDepth: range[1] },
    );
}

/** Adds the matches of a pair to the map's tracks (or just marks the pair if there are few). */
export function linkPair(
    map: SparseMap,
    i: number,
    j: number,
    matches: MatchList,
    touched: Set<number>,
): void {
    if (matches.length < 2 * MIN_MATCHES) {
        map.matched.add(pairKey(i, j));
        return;
    }
    for (const id of map.link(i, j, matches)) touched.add(id);
}

/**
 * Links frame `index` with the frames that see the same tracks plus the most recent ones by
 * guided matching. While recording, pairs far apart in time (where drift has piled up) get a
 * wider band checked by F-RANSAC; with `refined` poses every pair uses the tight band.
 */
export function linkCovisible(
    map: SparseMap,
    index: number,
    touched: Set<number>,
    range = ANY_DEPTH,
    refined = false,
): void {
    const frame = map.frames[index];
    for (const other of map.covisible(index, LINK_LIMIT, RECENT_LINKS)) {
        const loose =
            !refined && Math.abs(frame.time - map.frames[other].time) > LOOSE_AFTER_SECONDS;
        let matches = matchPair(map, index, other, loose ? LOOSE_BAND : TIGHT_BAND, range);
        if (loose) {
            matches = fundamentalInliers(
                frame.features,
                map.frames[other].features,
                matches,
                2,
                index,
            );
        }
        linkPair(map, index, other, matches, touched);
    }
}

/** Triangulates the touched tracks and refreshes the frame's depth hint. */
export function finishLinking(map: SparseMap, index: number, touched: Set<number>): void {
    map.triangulate(touched);
    const depth = map.medianDepth(index);
    if (Number.isFinite(depth)) map.frames[index].depthHint = depth;
}
