import { GrayImage } from '../image/gray-image';
import { DescriptorMetric, FeatureSet } from './feature-set';
import { selectOnGrid } from './grid-selection';
import { GradientCache, SIFT_DIMENSIONS, describe, dominantOrientation } from './sift/descriptor';
import { detectKeypoints } from './sift/keypoints';
import { buildScaleSpace } from './sift/scale-space';

/** 128 bytes packed in 32 u32 words per keypoint. */
const WORDS = SIFT_DIMENSIONS / 4;
const CELL = 24;

/**
 * SIFT (Lowe 2004): difference-of-Gaussians extrema located to sub-pixel accuracy, one
 * dominant orientation each, RootSIFT descriptors. Slower than ORB but better localised and
 * more distinctive, which is what bundle adjustment needs.
 */
export function extractSift(image: GrayImage, budget: number): FeatureSet {
    const octaves = buildScaleSpace(image);
    const picked = selectOnGrid(detectKeypoints(octaves), image.width, image.height, CELL, budget);
    const gradients = new GradientCache(octaves);
    const bytes = new Uint8Array(picked.length * SIFT_DIMENSIONS);
    const angles = new Float32Array(picked.length);
    picked.forEach((keypoint, k) => {
        const g = gradients.of(keypoint);
        angles[k] = dominantOrientation(keypoint, g);
        describe(keypoint, angles[k], g, bytes, k * SIFT_DIMENSIONS);
    });
    return {
        count: picked.length,
        x: Float32Array.from(picked, (k) => k.x),
        y: Float32Array.from(picked, (k) => k.y),
        level: Uint8Array.from(picked, (k) => k.octave),
        angle: angles,
        descriptors: new Uint32Array(bytes.buffer),
        metric: SIFT_METRIC,
    };
}

/**
 * Euclidean distance between byte-quantised RootSIFT vectors (512 = unit length), checked
 * against the bound every 8 words (partial distance elimination).
 */
export function siftDistance(
    a: Uint32Array,
    ia: number,
    b: Uint32Array,
    ib: number,
    bound = Infinity,
): number {
    const limit = bound * bound;
    let sum = 0;
    for (let w = 0; w < WORDS; w++) {
        const x = a[ia + w];
        const y = b[ib + w];
        const d0 = (x & 255) - (y & 255);
        const d1 = ((x >>> 8) & 255) - ((y >>> 8) & 255);
        const d2 = ((x >>> 16) & 255) - ((y >>> 16) & 255);
        const d3 = (x >>> 24) - (y >>> 24);
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        if ((w & 7) === 7 && sum > limit) break;
    }
    return Math.sqrt(sum);
}

export const SIFT_METRIC: DescriptorMetric = {
    name: 'sift',
    words: WORDS,
    distance: siftDistance,
    // Hellinger distance ≈ 0.6 between unit RootSIFT vectors.
    maxDistance: 0.6 * 512,
};
