import { QualityProfile } from '../../../core/models/quality';
import { Intrinsics, scaleIntrinsics } from '../../geometry/camera';
import { downsampleGray, rgbaToGray } from '../../image/gray-image';
import { upsampleDepth } from '../../mvs/depth-maps';
import { StereoInput } from '../../mvs/methods';
import { depthRange, selectSources } from '../../mvs/view-selection';
import { selectKeyframes } from '../keyframe-selection';
import { ScanSession } from '../scan-session';

/** The keyframes used for dense reconstruction and everything each one needs. */
export interface StereoPlan extends StereoInput {
    /** Keyframe index of each layer. */
    frames: number[];
    /** Views checked by the consistency filter, and how many must agree, per layer. */
    checks: number[][];
    needed: number[];
    /** The same for the depths integrated into the TSDF. */
    meshNeeded: number[];
}

/**
 * Picks the stereo resolution and the keyframes for dense reconstruction (quality budget, and
 * as many as the GPU's storage-buffer limit allows at this resolution), then their stereo
 * sources, filter views and depth ranges, expressed in layers (positions among the chosen
 * keyframes). Above the work resolution the gray images come from the full keyframes.
 */
export function planStereo(
    session: ScanSession,
    profile: QualityProfile,
    maxBytes: number,
): StereoPlan {
    const { map, keyframes } = session;
    const work = keyframes[0].work;
    const wanted = Math.min(profile.views, keyframes.length);
    const views = (scale: number) =>
        Math.floor(
            maxBytes / (Math.round(work.width * scale) * Math.round(work.height * scale) * 4),
        );
    // Above the work resolution only while the GPU still holds every wanted view at that size.
    let scale = profile.stereo.scale;
    while (scale > 1 && views(scale) < wanted) scale = Math.max(1, scale - 0.25);
    const width = Math.round(work.width * scale);
    const height = Math.round(work.height * scale);
    const byMemory = views(scale);
    const frames = selectKeyframes(
        keyframes.map((k, i) => ({
            pose: map.frames[i].pose,
            depthHint: map.frames[i].depthHint,
            sharpness: k.sharpness,
        })),
        Math.min(profile.views, byMemory),
    );
    const allowed = new Set(frames);
    const layerOf = new Map(frames.map((frame, layer) => [frame, layer]));
    const toLayers = (list: number[]) => list.map((frame) => layerOf.get(frame) as number);
    const intrinsics: Intrinsics[] = frames.map((i) =>
        scaleIntrinsics(map.frames[i].intrinsics, width / work.width, height / work.height),
    );
    const checks = frames.map((i) =>
        toLayers(selectSources(map, i, profile.filter.views, allowed)),
    );
    return {
        frames,
        width,
        height,
        views: frames.map((i, layer) => ({
            intrinsics: intrinsics[layer],
            pose: map.frames[i].pose,
        })),
        sources: frames.map((i) =>
            toLayers(selectSources(map, i, profile.stereo.sources, allowed)),
        ),
        checks,
        needed: checks.map((list) => agreeing(list.length, profile.filter.agreement)),
        meshNeeded: checks.map((list) =>
            agreeing(list.length, profile.filter.meshAgreement ?? profile.filter.agreement),
        ),
        ranges: frames.map((i) => depthRange(map, i, keyframes[i].depth?.meters ?? null)),
        async gray(layer, w, h) {
            const keyframe = keyframes[frames[layer]];
            if (w === work.width && h === work.height) return keyframe.work.gray;
            if (w < work.width) {
                const image = { width: work.width, height: work.height, data: keyframe.work.gray };
                return downsampleGray(image, w, h).data;
            }
            // Finer than the work images: from the full-resolution keyframe.
            const image = await session.images.get(frames[layer]);
            return rgbaToGray(image.data, image.width, image.height, w, h).data;
        },
        // Made on demand: keeping a full-resolution prior per view costs ~1 MB each.
        prior(layer, w, h) {
            const depth = keyframes[frames[layer]].depth;
            return depth ? upsampleDepth(depth, w, h) : null;
        },
    };
}

/** Views that must agree out of `count` for a fraction (at least 2). */
function agreeing(count: number, fraction: number): number {
    return Math.min(count, Math.max(2, Math.round(fraction * count)));
}
