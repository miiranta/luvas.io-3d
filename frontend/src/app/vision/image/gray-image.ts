/** Single-channel 8-bit image. */
export interface GrayImage {
    width: number;
    height: number;
    data: Uint8Array;
}

/**
 * RGBA → luma with area-average downsampling to the requested size. Pixel centres map as
 * (x + 0.5)·s − 0.5 (as scaleIntrinsics assumes), also for non-integer ratios.
 */
export function rgbaToGray(
    rgba: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    targetWidth: number,
    targetHeight: number,
): GrayImage {
    if (width % targetWidth !== 0 || height % targetHeight !== 0) {
        const luma = new Float32Array(width * height);
        for (let i = 0; i < luma.length; i++) {
            luma[i] = (77 * rgba[i * 4] + 150 * rgba[i * 4 + 1] + 29 * rgba[i * 4 + 2]) / 256;
        }
        return resampleArea(luma, width, height, targetWidth, targetHeight);
    }
    const data = new Uint8Array(targetWidth * targetHeight);
    const sx = width / targetWidth;
    const sy = height / targetHeight;
    for (let y = 0; y < targetHeight; y++) {
        const y0 = Math.floor(y * sy);
        const y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
        for (let x = 0; x < targetWidth; x++) {
            const x0 = Math.floor(x * sx);
            const x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
            let sum = 0;
            for (let yy = y0; yy < y1; yy++) {
                let i = (yy * width + x0) * 4;
                for (let xx = x0; xx < x1; xx++, i += 4) {
                    sum += 77 * rgba[i] + 150 * rgba[i + 1] + 29 * rgba[i + 2];
                }
            }
            data[y * targetWidth + x] = Math.round(sum / (256 * (y1 - y0) * (x1 - x0)));
        }
    }
    return { width: targetWidth, height: targetHeight, data };
}

/** RGBA → packed RGB with area-average downsampling. */
export function downsampleRgb(
    rgba: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    targetWidth: number,
    targetHeight: number,
): Uint8Array {
    const out = new Uint8Array(targetWidth * targetHeight * 3);
    const sx = width / targetWidth;
    const sy = height / targetHeight;
    for (let y = 0; y < targetHeight; y++) {
        const y0 = Math.floor(y * sy);
        const y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
        for (let x = 0; x < targetWidth; x++) {
            const x0 = Math.floor(x * sx);
            const x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
            let r = 0;
            let g = 0;
            let b = 0;
            for (let yy = y0; yy < y1; yy++) {
                let i = (yy * width + x0) * 4;
                for (let xx = x0; xx < x1; xx++, i += 4) {
                    r += rgba[i];
                    g += rgba[i + 1];
                    b += rgba[i + 2];
                }
            }
            const n = (y1 - y0) * (x1 - x0);
            const o = (y * targetWidth + x) * 3;
            out[o] = r / n;
            out[o + 1] = g / n;
            out[o + 2] = b / n;
        }
    }
    return out;
}

/** Bilinear resize of a gray image (used to build the feature pyramid). */
export function resizeGray(image: GrayImage, width: number, height: number): GrayImage {
    const data = new Uint8Array(width * height);
    const sx = image.width / width;
    const sy = image.height / height;
    for (let y = 0; y < height; y++) {
        const fy = Math.min(image.height - 1.001, Math.max(0, (y + 0.5) * sy - 0.5));
        const y0 = Math.floor(fy);
        const wy = fy - y0;
        for (let x = 0; x < width; x++) {
            const fx = Math.min(image.width - 1.001, Math.max(0, (x + 0.5) * sx - 0.5));
            const x0 = Math.floor(fx);
            const wx = fx - x0;
            const i = y0 * image.width + x0;
            const top = image.data[i] * (1 - wx) + image.data[i + 1] * wx;
            const bottom =
                image.data[i + image.width] * (1 - wx) + image.data[i + image.width + 1] * wx;
            data[y * width + x] = Math.round(top * (1 - wy) + bottom * wy);
        }
    }
    return { width, height, data };
}

/** Separable [1 4 6 4 1] / 16 blur (σ ≈ 1), clamped at the borders. */
export function blurGray(image: GrayImage): GrayImage {
    const { width, height } = image;
    const temp = new Uint16Array(width * height);
    const src = image.data;
    for (let y = 0; y < height; y++) {
        const row = y * width;
        for (let x = 0; x < width; x++) {
            const a = src[row + Math.max(0, x - 2)];
            const b = src[row + Math.max(0, x - 1)];
            const c = src[row + x];
            const d = src[row + Math.min(width - 1, x + 1)];
            const e = src[row + Math.min(width - 1, x + 2)];
            temp[row + x] = a + 4 * b + 6 * c + 4 * d + e;
        }
    }
    const data = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        const ym2 = Math.max(0, y - 2) * width;
        const ym1 = Math.max(0, y - 1) * width;
        const yp1 = Math.min(height - 1, y + 1) * width;
        const yp2 = Math.min(height - 1, y + 2) * width;
        const row = y * width;
        for (let x = 0; x < width; x++) {
            const sum =
                temp[ym2 + x] +
                4 * temp[ym1 + x] +
                6 * temp[row + x] +
                4 * temp[yp1 + x] +
                temp[yp2 + x];
            data[row + x] = (sum + 128) >> 8;
        }
    }
    return { width, height, data };
}

/** Variance of the Laplacian: a standard focus / motion-blur measure. */
export function sharpness(image: GrayImage): number {
    const { width, height, data } = image;
    let sum = 0;
    let sumSq = 0;
    let n = 0;
    for (let y = 1; y < height - 1; y += 2) {
        for (let x = 1; x < width - 1; x += 2) {
            const i = y * width + x;
            const l = data[i - 1] + data[i + 1] + data[i - width] + data[i + width] - 4 * data[i];
            sum += l;
            sumSq += l * l;
            n++;
        }
    }
    if (n === 0) return 0;
    const mean = sum / n;
    return sumSq / n - mean * mean;
}

/** Area-average downsampling of a gray image (used for the coarse / low-quality stereo). */
export function downsampleGray(image: GrayImage, width: number, height: number): GrayImage {
    if (image.width % width !== 0 || image.height % height !== 0) {
        return resampleArea(image.data, image.width, image.height, width, height);
    }
    const data = new Uint8Array(width * height);
    const sx = image.width / width;
    const sy = image.height / height;
    for (let y = 0; y < height; y++) {
        const y0 = Math.floor(y * sy);
        const y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
        for (let x = 0; x < width; x++) {
            const x0 = Math.floor(x * sx);
            const x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
            let sum = 0;
            for (let yy = y0; yy < y1; yy++) {
                for (let xx = x0; xx < x1; xx++) sum += image.data[yy * image.width + xx];
            }
            data[y * width + x] = Math.round(sum / ((y1 - y0) * (x1 - x0)));
        }
    }
    return { width, height, data };
}

/**
 * Exact box-filter downsampling for any ratio: every target pixel averages the source area it
 * covers, with partial weights for partly covered source pixels (separable: rows, then columns).
 */
function resampleArea(
    source: ArrayLike<number>,
    width: number,
    height: number,
    targetWidth: number,
    targetHeight: number,
): GrayImage {
    const columns = coverage(width, targetWidth);
    const rows = coverage(height, targetHeight);
    const temp = new Float32Array(targetWidth * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < targetWidth; x++) {
            const { first, weights } = columns[x];
            let sum = 0;
            for (let k = 0; k < weights.length; k++)
                sum += weights[k] * source[y * width + first + k];
            temp[y * targetWidth + x] = sum;
        }
    }
    const data = new Uint8Array(targetWidth * targetHeight);
    for (let y = 0; y < targetHeight; y++) {
        const { first, weights } = rows[y];
        for (let x = 0; x < targetWidth; x++) {
            let sum = 0;
            for (let k = 0; k < weights.length; k++) {
                sum += weights[k] * temp[(first + k) * targetWidth + x];
            }
            data[y * targetWidth + x] = Math.min(255, Math.round(sum));
        }
    }
    return { width: targetWidth, height: targetHeight, data };
}

/** Source pixels covered by each target pixel along one axis, with normalised weights. */
function coverage(source: number, target: number): { first: number; weights: number[] }[] {
    const scale = source / target;
    return Array.from({ length: target }, (_, t) => {
        const start = t * scale;
        const end = Math.min(source, (t + 1) * scale);
        const first = Math.floor(start);
        const weights: number[] = [];
        for (let i = first; i < end; i++) {
            weights.push((Math.min(end, i + 1) - Math.max(start, i)) / scale);
        }
        return { first, weights };
    });
}
