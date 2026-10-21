import { extractFeatures } from '../features/orb-features';
import {
    Intrinsics,
    Pose,
    clonePose,
    fromCamera,
    poseFromXrTransform,
    scaleIntrinsics,
} from '../geometry/camera';
import { downsampleRgb, rgbaToGray, sharpness } from '../image/gray-image';
import { fundamentalInliers, guidedMatch } from '../matching/guided-matcher';
import { PointAccumulator } from '../fusion/point-accumulator';
import { median } from '../math/dense';
import { SparseMap, pairKey } from '../sfm/sparse-map';
import { RgbaImage } from '../texture/atlas-baker';

/** Phone depth sampled on a regular grid of normalised view coordinates (0 = no data). */
export interface DepthGrid {
    width: number;
    height: number;
    meters: Float32Array;
}

export interface KeyframeInput {
    /** Seconds since recording started. */
    time: number;
    image: RgbaImage;
    /** Intrinsics of `image`. */
    intrinsics: Intrinsics;
    /** WebXR view transform (camera → world, column-major, OpenGL axes). */
    xrTransform: ArrayLike<number>;
    depth: DepthGrid | null;
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
export const MAX_KEYFRAMES = 120;
const FEATURE_BUDGET = 1600;
const BLUR_RATIO = 0.5;
const RECENT_LINKS = 3;
const LINK_LIMIT = 7;
const LOOSE_AFTER_SECONDS = 5;
/** Epipolar bands (px): consecutive keyframes trust the odometry; loop pairs allow drift. */
const TIGHT_BAND = 4;
const LOOSE_BAND = 40;
const PREVIEW_CELL = 0.015;
const PREVIEW_LIMIT = 60000;

/**
 * Everything gathered while recording: the keyframes and the sparse map, which is extended
 * after every keyframe (features, guided matching against covisible keyframes, triangulation)
 * so that the AR preview shows real triangulated structure and the final reconstruction starts
 * with tracks already built.
 */
export class ScanSession {
    readonly map = new SparseMap();
    readonly keyframes: StoredKeyframe[] = [];
    private readonly preview = new PointAccumulator(PREVIEW_CELL);
    private readonly sharpnessHistory: number[] = [];
    private size: { width: number; height: number } | null = null;
    rejected = 0;

    constructor(readonly images: ImageStore) {}

    async addKeyframe(input: KeyframeInput): Promise<KeyframeVerdict> {
        const { image } = input;
        if (this.keyframes.length >= MAX_KEYFRAMES) return { accepted: false, reason: 'full' };
        if (this.size && (image.width !== this.size.width || image.height !== this.size.height)) {
            return { accepted: false, reason: 'orientation' };
        }
        const scale = Math.min(1, Math.sqrt(WORK_PIXELS / (image.width * image.height)));
        const workWidth = Math.round(image.width * scale);
        const workHeight = Math.round(image.height * scale);
        const gray = rgbaToGray(image.data, image.width, image.height, workWidth, workHeight);
        const focus = sharpness(gray);
        if (
            this.sharpnessHistory.length >= 4 &&
            focus < BLUR_RATIO * median(this.sharpnessHistory)
        ) {
            this.rejected++;
            return { accepted: false, reason: 'blurry' };
        }
        this.sharpnessHistory.push(focus);
        if (this.sharpnessHistory.length > 10) this.sharpnessHistory.shift();
        this.size ??= { width: image.width, height: image.height };

        const thumbWidth = Math.round(workWidth / 2);
        const thumbHeight = Math.round(workHeight / 2);
        const workK = scaleIntrinsics(
            input.intrinsics,
            workWidth / image.width,
            workHeight / image.height,
        );
        const thumbK = scaleIntrinsics(
            input.intrinsics,
            thumbWidth / image.width,
            thumbHeight / image.height,
        );
        const prior = poseFromXrTransform(input.xrTransform);
        const keyframe: StoredKeyframe = {
            time: input.time,
            texture: { width: image.width, height: image.height, intrinsics: input.intrinsics },
            work: { width: workWidth, height: workHeight, intrinsics: workK, gray: gray.data },
            thumb: {
                width: thumbWidth,
                height: thumbHeight,
                intrinsics: thumbK,
                rgb: downsampleRgb(image.data, image.width, image.height, thumbWidth, thumbHeight),
            },
            prior,
            depth: input.depth,
            sharpness: focus,
        };
        const index = this.keyframes.length;
        this.keyframes.push(keyframe);
        await this.images.put(index, image);

        const depthMedian = input.depth ? medianDepth(input.depth) : NaN;
        const previous = this.map.frames.at(-1);
        const features = extractFeatures(gray, FEATURE_BUDGET);
        this.map.addFrame({
            intrinsics: workK,
            width: workWidth,
            height: workHeight,
            pose: clonePose(prior),
            prior,
            features,
            track: new Int32Array(features.count).fill(-1),
            rgb: keyframe.thumb.rgb,
            thumbWidth,
            thumbHeight,
            time: input.time,
            depthHint: Number.isFinite(depthMedian) ? depthMedian : (previous?.depthHint ?? 1),
        });
        this.link(index);
        this.addDepthPreview(keyframe);
        return { accepted: true, index };
    }

    private link(index: number): void {
        const frame = this.map.frames[index];
        const touched = new Set<number>();
        for (const other of this.map.covisible(index, LINK_LIMIT, RECENT_LINKS)) {
            const neighbour = this.map.frames[other];
            const loose = Math.abs(frame.time - neighbour.time) > LOOSE_AFTER_SECONDS;
            let matches = guidedMatch(
                frame.features,
                neighbour.features,
                frame.intrinsics,
                neighbour.intrinsics,
                frame.pose,
                neighbour.pose,
                neighbour.width,
                neighbour.height,
                { band: loose ? LOOSE_BAND : TIGHT_BAND, minDepth: 0.05, maxDepth: 15 },
            );
            if (loose)
                matches = fundamentalInliers(frame.features, neighbour.features, matches, 2, index);
            if (matches.length < 2 * 12) {
                this.map.matched.add(pairKey(index, other));
                continue;
            }
            for (const id of this.map.link(index, other, matches)) touched.add(id);
        }
        this.map.triangulate(touched);
        const depth = this.map.medianDepth(index);
        if (Number.isFinite(depth)) frame.depthHint = depth;
    }

    /** Back-projects the phone depth into the preview cloud (coarse, for the AR overlay). */
    private addDepthPreview(keyframe: StoredKeyframe): void {
        const grid = keyframe.depth;
        if (!grid) return;
        const k = scaleIntrinsics(
            keyframe.work.intrinsics,
            grid.width / keyframe.work.width,
            grid.height / keyframe.work.height,
        );
        const t = keyframe.thumb;
        for (let y = 0; y < grid.height; y += 2) {
            for (let x = 0; x < grid.width; x += 2) {
                const d = grid.meters[y * grid.width + x];
                if (!(d > 0.05 && d < 6)) continue;
                const world = fromCamera(keyframe.prior, [
                    ((x - k.cx) / k.fx) * d,
                    ((y - k.cy) / k.fy) * d,
                    d,
                ]);
                const tx = Math.min(
                    t.width - 1,
                    Math.round(((x + 0.5) / grid.width) * t.width - 0.5),
                );
                const ty = Math.min(
                    t.height - 1,
                    Math.round(((y + 0.5) / grid.height) * t.height - 0.5),
                );
                const i = (ty * t.width + tx) * 3;
                this.preview.add(
                    world[0],
                    world[1],
                    world[2],
                    0,
                    0,
                    0,
                    t.rgb[i],
                    t.rgb[i + 1],
                    t.rgb[i + 2],
                );
            }
        }
        while (this.preview.size > PREVIEW_LIMIT) this.preview.coarsen();
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

function medianDepth(grid: DepthGrid): number {
    const values: number[] = [];
    for (let i = 0; i < grid.meters.length; i += 3)
        if (grid.meters[i] > 0) values.push(grid.meters[i]);
    return values.length > 20 ? median(values) : NaN;
}
