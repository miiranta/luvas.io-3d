import { FeatureExtractor, FeatureMethod } from '../../features/methods';
import { clonePose } from '../../geometry/camera';
import {
    TIGHT_BAND,
    finishLinking,
    linkCovisible,
    linkPair,
    matchPair,
} from '../../sfm/map-linking';
import { SparseMap } from '../../sfm/sparse-map';
import { depthRange } from '../../mvs/view-selection';
import { ScanSession, StoredKeyframe } from '../scan-session';
import { Progress } from './stage-runner';

/**
 * A new sparse map from features re-extracted with another method, linked in capture order like
 * the live map but along tight epipolar bands of the already refined poses, within each frame's
 * depth range. Frames share their intrinsics with the live map, so a focal correction applies
 * to both.
 */
export async function buildFeatureMap(
    session: ScanSession,
    method: FeatureMethod,
    device: GPUDevice | null,
    progress: Progress,
): Promise<SparseMap> {
    const live = session.map;
    const map = new SparseMap();
    for (const index of live.breaks) map.breaks.add(index);
    const { keyframes } = session;
    const count = keyframes.length;
    const extractor = method.extractor(device, keyframes[0].work.width, keyframes[0].work.height);
    // Depths seen by each frame in the refined live map bound the epipolar search.
    const ranges = keyframes.map((k, i) => depthRange(live, i, k.depth?.meters ?? null));
    try {
        for (let index = 0; index < count; index++) {
            await linkFrame(map, live, index, extractor, keyframes, ranges[index]);
            if (index % 4 === 0) progress(index / count, `${method.name} ${index + 1}/${count}`);
        }
    } finally {
        extractor.destroy();
    }
    return map;
}

async function linkFrame(
    map: SparseMap,
    live: SparseMap,
    index: number,
    extractor: FeatureExtractor,
    keyframes: StoredKeyframe[],
    range: [number, number],
): Promise<void> {
    const { work } = keyframes[index];
    const features = await extractor.extract({
        width: work.width,
        height: work.height,
        data: work.gray,
    });
    const frame = live.frames[index];
    map.addFrame({
        ...frame,
        pose: clonePose(frame.pose),
        features,
        track: new Int32Array(features.count).fill(-1),
    });
    const touched = new Set<number>();
    if (index > 0) {
        const matches = matchPair(map, index, index - 1, TIGHT_BAND, range);
        linkPair(map, index, index - 1, matches, touched);
    }
    linkCovisible(map, index, touched, range, true);
    finishLinking(map, index, touched);
}
