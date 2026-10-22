import { GrayImage } from '../../image/gray-image';

/** Blur of the base image of every octave (Lowe's σ₀). */
export const SIGMA0 = 1.6;
/** Blur assumed already present in the input image. */
const INPUT_SIGMA = 0.5;
/** Scales per octave at which extrema are searched. */
export const INTERVALS = 3;
/**
 * The input is upsampled 2× (bilinear) before the first octave, as in Lowe's paper: about
 * four times more stable keypoints, mostly at the finest, best-localised scales.
 */
export const UPSCALE = 2;
const MIN_SIZE = 24;

export interface Octave {
    width: number;
    height: number;
    /** INTERVALS + 3 Gaussian images (values in [0, 1]); image i has blur σ₀·2^(i/INTERVALS). */
    gaussians: Float32Array[];
    /** INTERVALS + 2 differences of consecutive Gaussians. */
    dogs: Float32Array[];
}

/** Octave sizes for an input image, from the upsampled first one until they get too small. */
export function octaveSizes(
    inputWidth: number,
    inputHeight: number,
): { width: number; height: number }[] {
    const sizes: { width: number; height: number }[] = [];
    let width = inputWidth * UPSCALE;
    let height = inputHeight * UPSCALE;
    while (Math.min(width, height) >= MIN_SIZE) {
        sizes.push({ width, height });
        width = Math.floor(width / 2);
        height = Math.floor(height / 2);
    }
    return sizes;
}

/**
 * Blurs applied incrementally: [0] takes the input to σ₀, [i] takes Gaussian image i − 1 of an
 * octave to image i (σ₀·kⁱ⁻¹ → σ₀·kⁱ with k = 2^(1/INTERVALS)).
 */
export function blurSteps(): number[] {
    const k = Math.pow(2, 1 / INTERVALS);
    const steps = [Math.sqrt(SIGMA0 * SIGMA0 - (UPSCALE * INPUT_SIGMA) ** 2)];
    for (let i = 1; i < INTERVALS + 3; i++) {
        const previous = SIGMA0 * Math.pow(k, i - 1);
        steps.push(Math.sqrt((previous * k) ** 2 - previous ** 2));
    }
    return steps;
}

/** One half (centre first) of a normalised ±3σ Gaussian kernel. */
export function gaussianKernel(sigma: number): Float32Array {
    const radius = Math.max(1, Math.ceil(sigma * 3));
    const kernel = new Float32Array(radius + 1);
    let sum = 0;
    for (let i = 0; i <= radius; i++) {
        kernel[i] = Math.exp((-i * i) / (2 * sigma * sigma));
        sum += i === 0 ? kernel[i] : 2 * kernel[i];
    }
    for (let i = 0; i <= radius; i++) kernel[i] /= sum;
    return kernel;
}

/** Gaussian scale space and difference-of-Gaussians pyramid of an 8-bit image. */
export function buildScaleSpace(image: GrayImage): Octave[] {
    let width = image.width * UPSCALE;
    let height = image.height * UPSCALE;
    let base: Float32Array = upsample(image);
    const steps = blurSteps();
    base = gaussianBlur(base, width, height, steps[0]);
    const octaves: Octave[] = [];
    while (Math.min(width, height) >= MIN_SIZE) {
        const gaussians: Float32Array[] = [base];
        for (let i = 1; i < INTERVALS + 3; i++) {
            gaussians.push(gaussianBlur(gaussians[i - 1], width, height, steps[i]));
        }
        const dogs: Float32Array[] = [];
        for (let i = 0; i < INTERVALS + 2; i++) {
            const dog = new Float32Array(width * height);
            const a = gaussians[i];
            const b = gaussians[i + 1];
            for (let p = 0; p < dog.length; p++) dog[p] = b[p] - a[p];
            dogs.push(dog);
        }
        octaves.push({ width, height, gaussians, dogs });
        // The next octave starts from the image with twice the base blur, subsampled.
        const source = gaussians[INTERVALS];
        const nextWidth = Math.floor(width / 2);
        const nextHeight = Math.floor(height / 2);
        base = new Float32Array(nextWidth * nextHeight);
        for (let y = 0; y < nextHeight; y++) {
            for (let x = 0; x < nextWidth; x++)
                base[y * nextWidth + x] = source[y * 2 * width + x * 2];
        }
        width = nextWidth;
        height = nextHeight;
    }
    return octaves;
}

/**
 * Bilinear UPSCALE× enlargement to [0, 1] values; output pixel X samples input coordinate
 * X / UPSCALE (pixel centres at integers, so octave coordinates map back by a plain factor).
 */
export function upsample(image: GrayImage): Float32Array {
    const { width, height, data } = image;
    const w = width * UPSCALE;
    const h = height * UPSCALE;
    const out = new Float32Array(w * h);
    for (let Y = 0; Y < h; Y++) {
        const y = Y / UPSCALE;
        const y0 = Math.min(height - 1, Math.floor(y));
        const y1 = Math.min(height - 1, y0 + 1);
        const fy = y - y0;
        for (let X = 0; X < w; X++) {
            const x = X / UPSCALE;
            const x0 = Math.min(width - 1, Math.floor(x));
            const x1 = Math.min(width - 1, x0 + 1);
            const fx = x - x0;
            const top = data[y0 * width + x0] * (1 - fx) + data[y0 * width + x1] * fx;
            const bottom = data[y1 * width + x0] * (1 - fx) + data[y1 * width + x1] * fx;
            out[Y * w + X] = (top * (1 - fy) + bottom * fy) / 255;
        }
    }
    return out;
}

/**
 * Separable Gaussian blur with a ±3σ kernel; borders replicate the edge pixel. Each line is
 * copied into a padded buffer first so the inner loop needs no bounds checks.
 */
export function gaussianBlur(
    source: Float32Array,
    width: number,
    height: number,
    sigma: number,
): Float32Array {
    const kernel = gaussianKernel(sigma);
    const radius = kernel.length - 1;
    const out = new Float32Array(width * height);
    const padded = new Float32Array(Math.max(width, height) + 2 * radius);
    const pass = (data: Float32Array, start: number, step: number, count: number) => {
        for (let n = 0; n < count; n++) padded[radius + n] = data[start + n * step];
        const first = padded[radius];
        const last = padded[radius + count - 1];
        for (let n = 0; n < radius; n++) {
            padded[n] = first;
            padded[radius + count + n] = last;
        }
        for (let n = 0; n < count; n++) {
            const c = radius + n;
            let value = kernel[0] * padded[c];
            for (let i = 1; i <= radius; i++) value += kernel[i] * (padded[c - i] + padded[c + i]);
            out[start + n * step] = value;
        }
    };
    for (let y = 0; y < height; y++) pass(source, y * width, 1, width);
    for (let x = 0; x < width; x++) pass(out, x, width, height);
    return out;
}
