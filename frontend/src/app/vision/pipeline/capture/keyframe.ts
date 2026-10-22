import { Intrinsics, Pose, scaleIntrinsics } from '../../geometry/camera';
import { GrayImage, downsampleRgb, rgbaToGray, sharpness } from '../../image/gray-image';
import { median } from '../../math/dense';
import { DepthGrid } from '../../mvs/depth-maps';
import { RgbaImage } from '../../texture/texture-types';

export interface KeyframeInput {
    /** Seconds since recording started. */
    time: number;
    image: RgbaImage;
    /** Intrinsics of `image`. */
    intrinsics: Intrinsics;
    /** WebXR view transform (camera → world, column-major, OpenGL axes). */
    xrTransform: ArrayLike<number>;
    depth: DepthGrid | null;
    /** Tracking segment: changes whenever the phone lost tracking in between. */
    segment?: number;
}

export interface StoredKeyframe {
    time: number;
    texture: { width: number; height: number; intrinsics: Intrinsics };
    work: { width: number; height: number; intrinsics: Intrinsics; gray: Uint8Array };
    thumb: { width: number; height: number; intrinsics: Intrinsics; rgb: Uint8Array };
    prior: Pose;
    depth: DepthGrid | null;
    sharpness: number;
}

/** Where full-resolution keyframe images live (compressed in the browser, raw in tests). */
export interface ImageStore {
    put(index: number, image: RgbaImage): Promise<void>;
    get(index: number): Promise<RgbaImage>;
    jpeg(image: RgbaImage, quality: number): Promise<Uint8Array>;
}

export interface KeyframeVerdict {
    accepted: boolean;
    reason?: string;
    index?: number;
}

/** Pixel budget of the stereo / feature resolution (640 × 360 for 16:9). */
export const WORK_PIXELS = 640 * 360;
const BLUR_RATIO = 0.5;
const SHARPNESS_HISTORY = 10;

/** The keyframe at work resolution, before the blur gate decides whether to keep it. */
export interface PreparedFrame {
    gray: GrayImage;
    sharpness: number;
}

export function prepareFrame(image: RgbaImage): PreparedFrame {
    const scale = Math.min(1, Math.sqrt(WORK_PIXELS / (image.width * image.height)));
    const width = Math.round(image.width * scale);
    const height = Math.round(image.height * scale);
    const gray = rgbaToGray(image.data, image.width, image.height, width, height);
    return { gray, sharpness: sharpness(gray) };
}

/** Rejects frames much blurrier than the recent ones (motion blur, refocusing). */
export class BlurGate {
    private readonly history: number[] = [];
    rejected = 0;

    accept(focus: number): boolean {
        if (this.history.length >= 4 && focus < BLUR_RATIO * median(this.history)) {
            this.rejected++;
            return false;
        }
        this.history.push(focus);
        if (this.history.length > SHARPNESS_HISTORY) this.history.shift();
        return true;
    }
}

/** Everything kept of an accepted keyframe: work-resolution gray, thumbnail and the prior. */
export function storeKeyframe(
    input: KeyframeInput,
    frame: PreparedFrame,
    prior: Pose,
): StoredKeyframe {
    const { image } = input;
    const { gray } = frame;
    const thumbWidth = Math.round(gray.width / 2);
    const thumbHeight = Math.round(gray.height / 2);
    return {
        time: input.time,
        texture: { width: image.width, height: image.height, intrinsics: input.intrinsics },
        work: {
            width: gray.width,
            height: gray.height,
            intrinsics: scaleIntrinsics(
                input.intrinsics,
                gray.width / image.width,
                gray.height / image.height,
            ),
            gray: gray.data,
        },
        thumb: {
            width: thumbWidth,
            height: thumbHeight,
            intrinsics: scaleIntrinsics(
                input.intrinsics,
                thumbWidth / image.width,
                thumbHeight / image.height,
            ),
            rgb: downsampleRgb(image.data, image.width, image.height, thumbWidth, thumbHeight),
        },
        prior,
        depth: input.depth,
        sharpness: frame.sharpness,
    };
}

/** Median of the phone depth (every third sample), NaN when there is too little of it. */
export function medianDepth(grid: DepthGrid): number {
    const values: number[] = [];
    for (let i = 0; i < grid.meters.length; i += 3)
        if (grid.meters[i] > 0) values.push(grid.meters[i]);
    return values.length > 20 ? median(values) : NaN;
}
