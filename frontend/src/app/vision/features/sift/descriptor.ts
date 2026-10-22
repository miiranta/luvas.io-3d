import { Octave } from './scale-space';
import { SiftKeypoint } from './keypoints';

/** 4 × 4 spatial cells × 8 orientation bins. */
export const SIFT_DIMENSIONS = 128;
const CELLS = 4;
const BINS = 8;
/** Width of a descriptor cell in multiples of the keypoint blur. */
const CELL_SCALE = 3;
const CLAMP = 0.2;
const ORIENTATION_BINS = 36;
/** Quantisation of the unit-length RootSIFT vector to bytes. */
const QUANTISATION = 512;

/** Gradient magnitude and direction of one Gaussian image (computed once, on demand). */
export interface Gradients {
    width: number;
    height: number;
    magnitude: Float32Array;
    angle: Float32Array;
}

export class GradientCache {
    private readonly cache = new Map<number, Gradients>();

    constructor(private readonly octaves: Octave[]) {}

    /** Gradients of the Gaussian image nearest to the keypoint's scale. */
    of(keypoint: SiftKeypoint): Gradients {
        const layer = Math.min(
            this.octaves[keypoint.octave].gaussians.length - 1,
            Math.max(0, Math.round(keypoint.scale)),
        );
        const key = keypoint.octave * 16 + layer;
        let gradients = this.cache.get(key);
        if (!gradients) {
            const { width, height, gaussians } = this.octaves[keypoint.octave];
            gradients = gradientsOf(gaussians[layer], width, height);
            this.cache.set(key, gradients);
        }
        return gradients;
    }
}

function gradientsOf(image: Float32Array, width: number, height: number): Gradients {
    const magnitude = new Float32Array(width * height);
    const angle = new Float32Array(width * height);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const i = y * width + x;
            const gx = image[i + 1] - image[i - 1];
            const gy = image[i + width] - image[i - width];
            magnitude[i] = Math.sqrt(gx * gx + gy * gy);
            angle[i] = Math.atan2(gy, gx);
        }
    }
    return { width, height, magnitude, angle };
}

/** Dominant gradient direction around the keypoint (36-bin histogram, interpolated peak). */
export function dominantOrientation(keypoint: SiftKeypoint, g: Gradients): number {
    const sigma = 1.5 * keypoint.sigma;
    const radius = Math.round(3 * sigma);
    const cx = Math.round(keypoint.ox);
    const cy = Math.round(keypoint.oy);
    const histogram = new Float64Array(ORIENTATION_BINS);
    const falloff = -1 / (2 * sigma * sigma);
    for (let dy = -radius; dy <= radius; dy++) {
        const y = cy + dy;
        if (y < 1 || y >= g.height - 1) continue;
        for (let dx = -radius; dx <= radius; dx++) {
            const x = cx + dx;
            if (x < 1 || x >= g.width - 1) continue;
            const i = y * g.width + x;
            const bin =
                Math.round((ORIENTATION_BINS * g.angle[i]) / (2 * Math.PI)) + ORIENTATION_BINS;
            histogram[bin % ORIENTATION_BINS] +=
                g.magnitude[i] * Math.exp((dx * dx + dy * dy) * falloff);
        }
    }
    const n = ORIENTATION_BINS;
    const smooth = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        smooth[i] =
            (histogram[(i + n - 2) % n] + histogram[(i + 2) % n]) / 16 +
            ((histogram[(i + n - 1) % n] + histogram[(i + 1) % n]) * 4) / 16 +
            (histogram[i] * 6) / 16;
    }
    let peak = 0;
    for (let i = 1; i < n; i++) if (smooth[i] > smooth[peak]) peak = i;
    const left = smooth[(peak + n - 1) % n];
    const right = smooth[(peak + 1) % n];
    const curvature = left - 2 * smooth[peak] + right;
    const offset = curvature < 0 ? (0.5 * (left - right)) / curvature : 0;
    return ((peak + offset) / n) * 2 * Math.PI;
}

/**
 * Lowe's descriptor: gradients in a 4 × 4 grid of cells rotated to the keypoint orientation,
 * 8 direction bins each with trilinear interpolation and a Gaussian window, clamped at 0.2 of
 * the norm for illumination invariance; then RootSIFT (L1 normalise + square root, so that L2
 * distance compares histograms by the Hellinger kernel), quantised to bytes.
 */
export function describe(
    keypoint: SiftKeypoint,
    angle: number,
    g: Gradients,
    out: Uint8Array,
    offset: number,
    histogram = new Float64Array((CELLS + 2) * (CELLS + 2) * (BINS + 2)),
): void {
    const cellWidth = CELL_SCALE * keypoint.sigma;
    const radius = Math.round(cellWidth * Math.SQRT2 * (CELLS + 1) * 0.5);
    const cos = Math.cos(angle) / cellWidth;
    const sin = Math.sin(angle) / cellWidth;
    const falloff = -1 / (0.5 * CELLS * CELLS);
    const stride = BINS + 2;
    const rowStride = (CELLS + 2) * stride;
    const binScale = BINS / (2 * Math.PI);
    const centre = CELLS / 2 - 0.5;
    histogram.fill(0);
    const { width, height, magnitude, angle: direction } = g;
    const cx = Math.round(keypoint.ox);
    const cy = Math.round(keypoint.oy);
    const x0 = Math.max(1, cx - radius);
    const x1 = Math.min(width - 2, cx + radius);
    for (let y = Math.max(1, cy - radius); y <= Math.min(height - 2, cy + radius); y++) {
        // Offset from the sub-pixel centre in cell units, in the keypoint's frame, stepped along x.
        const py = y - keypoint.oy;
        const px0 = x0 - keypoint.ox;
        let u = cos * px0 + sin * py;
        let v = -sin * px0 + cos * py;
        for (let x = x0; x <= x1; x++, u += cos, v -= sin) {
            const column = u + centre;
            const row = v + centre;
            if (row <= -1 || row >= CELLS || column <= -1 || column >= CELLS) continue;
            const i = y * width + x;
            let bin = (direction[i] - angle) * binScale;
            if (bin < 0) bin += BINS;
            if (bin < 0) bin += BINS;
            if (bin >= BINS) bin -= BINS;
            const value = magnitude[i] * Math.exp((u * u + v * v) * falloff);
            const r0 = Math.floor(row);
            const c0 = Math.floor(column);
            const o0 = Math.floor(bin);
            const fr = row - r0;
            const fc = column - c0;
            const fo = bin - o0;
            const top = value * (1 - fr);
            const bottom = value * fr;
            const base = (r0 + 1) * rowStride + (c0 + 1) * stride + o0;
            const v00 = top * (1 - fc);
            const v01 = top * fc;
            const v10 = bottom * (1 - fc);
            const v11 = bottom * fc;
            histogram[base] += v00 * (1 - fo);
            histogram[base + 1] += v00 * fo;
            histogram[base + stride] += v01 * (1 - fo);
            histogram[base + stride + 1] += v01 * fo;
            histogram[base + rowStride] += v10 * (1 - fo);
            histogram[base + rowStride + 1] += v10 * fo;
            histogram[base + rowStride + stride] += v11 * (1 - fo);
            histogram[base + rowStride + stride + 1] += v11 * fo;
        }
    }
    const descriptor = new Float64Array(SIFT_DIMENSIONS);
    for (let r = 0; r < CELLS; r++) {
        for (let c = 0; c < CELLS; c++) {
            const base = (r + 1) * rowStride + (c + 1) * stride;
            histogram[base] += histogram[base + BINS];
            histogram[base + 1] += histogram[base + BINS + 1];
            for (let k = 0; k < BINS; k++) {
                descriptor[(r * CELLS + c) * BINS + k] = histogram[base + k];
            }
        }
    }
    let norm = 0;
    for (const value of descriptor) norm += value * value;
    const limit = CLAMP * Math.sqrt(norm);
    let l1 = 0;
    for (let k = 0; k < SIFT_DIMENSIONS; k++) {
        descriptor[k] = Math.min(descriptor[k], limit);
        l1 += descriptor[k];
    }
    for (let k = 0; k < SIFT_DIMENSIONS; k++) {
        const root = l1 > 0 ? Math.sqrt(descriptor[k] / l1) : 0;
        out[offset + k] = Math.min(255, Math.round(root * QUANTISATION));
    }
}
