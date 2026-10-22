import { FEATURE_METHODS } from '../features/methods';
import { poseFromXrTransform } from '../geometry/camera';
import { median } from '../math/dense';
import { MapSnapshot, SparseMap } from '../sfm/sparse-map';
import { TIGHT_BAND, finishLinking, linkCovisible, linkPair, matchPair } from '../sfm/map-linking';
import { DepthPreview } from './capture/depth-preview';
import {
    BlurGate,
    ImageStore,
    KeyframeInput,
    KeyframeVerdict,
    StoredKeyframe,
    medianDepth,
    prepareFrame,
    storeKeyframe,
} from './capture/keyframe';
import { PoseGuard } from './capture/pose-guard';

export type { DepthGrid } from '../mvs/depth-maps';
export type {
    ImageStore,
    KeyframeInput,
    KeyframeVerdict,
    StoredKeyframe,
} from './capture/keyframe';
export { WORK_PIXELS } from './capture/keyframe';

/** Capture budget; each quality level then uses its own subset of the keyframes. */
export const MAX_KEYFRAMES = 180;
/** Live tracking always uses ORB: fast enough for every keyframe on the phone. */
const LIVE_FEATURES = FEATURE_METHODS.orb;

/**
 * Everything gathered while recording: the keyframes and the sparse map, which is extended
 * after every keyframe (features, guided matching against covisible keyframes, triangulation)
 * so that the AR preview shows real triangulated structure and the final reconstruction starts
 * with tracks already built.
 */
export class ScanSession {
    readonly map = new SparseMap();
    readonly keyframes: StoredKeyframe[] = [];
    private readonly preview = new DepthPreview();
    private readonly blur = new BlurGate();
    private readonly guard = new PoseGuard(this.map);
    private size: { width: number; height: number } | null = null;
    /** The map as recorded, restored before every build so rebuilds are repeatable. */
    baseline: MapSnapshot | null = null;

    constructor(readonly images: ImageStore) {}

    /** Frames dropped as blurry. */
    get rejected(): number {
        return this.blur.rejected;
    }

    /** Consecutive pairs that needed unguided matching. */
    get fallbacks(): number {
        return this.guard.fallbacks;
    }

    /** Tracking segments re-anchored by resection. */
    get realigned(): number {
        return this.guard.realigned;
    }

    async addKeyframe(input: KeyframeInput): Promise<KeyframeVerdict> {
        const { image } = input;
        if (this.keyframes.length >= MAX_KEYFRAMES) return { accepted: false, reason: 'full' };
        if (this.size && (image.width !== this.size.width || image.height !== this.size.height)) {
            return { accepted: false, reason: 'orientation' };
        }
        const frame = prepareFrame(image);
        if (!this.blur.accept(frame.sharpness)) return { accepted: false, reason: 'blurry' };
        this.size ??= { width: image.width, height: image.height };

        const prior = poseFromXrTransform(input.xrTransform);
        const index = this.keyframes.length;
        if (this.guard.isBreak(input.time, prior, input.segment)) this.map.breaks.add(index);
        const keyframe = storeKeyframe(input, frame, prior);
        this.keyframes.push(keyframe);
        await this.images.put(index, image);

        const depthMedian = input.depth ? medianDepth(input.depth) : NaN;
        const previous = this.map.frames.at(-1);
        const features = LIVE_FEATURES.extract(frame.gray, LIVE_FEATURES.budget);
        this.map.addFrame({
            intrinsics: keyframe.work.intrinsics,
            width: keyframe.work.width,
            height: keyframe.work.height,
            pose: this.guard.initialPose(prior),
            prior,
            features,
            track: new Int32Array(features.count).fill(-1),
            rgb: keyframe.thumb.rgb,
            thumbWidth: keyframe.thumb.width,
            thumbHeight: keyframe.thumb.height,
            time: input.time,
            depthHint: Number.isFinite(depthMedian) ? depthMedian : (previous?.depthHint ?? 1),
        });
        this.link(index);
        this.preview.add(keyframe, this.map.frames[index].pose);
        return { accepted: true, index };
    }

    private link(index: number): void {
        const touched = new Set<number>();
        // The consecutive pair first: it is the one that tells whether the pose can be trusted.
        if (index > 0) {
            const guided = matchPair(this.map, index, index - 1, TIGHT_BAND);
            const matches = this.guard.checkConsecutive(index, index - 1, guided);
            linkPair(this.map, index, index - 1, matches, touched);
        }
        linkCovisible(this.map, index, touched);
        finishLinking(this.map, index, touched);
    }

    /** Points for the live AR overlay: phone-depth cloud plus triangulated sparse points. */
    previewCloud(): { positions: Float32Array; colors: Uint8Array } {
        const dense = this.preview.cloud();
        const sparse = this.map.points(2);
        const positions = new Float32Array(dense.positions.length + sparse.positions.length);
        positions.set(dense.positions, 0);
        positions.set(sparse.positions, dense.positions.length);
        const colors = new Uint8Array(dense.colors.length + sparse.colors.length);
        colors.set(dense.colors, 0);
        colors.set(sparse.colors, dense.colors.length);
        return { positions, colors };
    }

    /** Median scene depth over the latest keyframes (guides keyframe spacing). */
    sceneDepth(): number {
        const recent = this.map.frames.slice(-5).map((frame) => frame.depthHint);
        return recent.length ? median(recent) : 1;
    }
}
