/** Phone depth sampled on a regular grid of normalised view coordinates (0 = no data). */
export interface DepthGrid {
    width: number;
    height: number;
    meters: Float32Array;
}

/** Bilinear upsampling of a depth grid to width × height, ignoring cells without data. */
export function upsampleDepth(grid: DepthGrid, width: number, height: number): Float32Array {
    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        const gy = Math.min(grid.height - 1, Math.max(0, ((y + 0.5) / height) * grid.height - 0.5));
        const y0 = Math.floor(gy);
        const y1 = Math.min(grid.height - 1, y0 + 1);
        const wy = gy - y0;
        for (let x = 0; x < width; x++) {
            const gx = Math.min(
                grid.width - 1,
                Math.max(0, ((x + 0.5) / width) * grid.width - 0.5),
            );
            const x0 = Math.floor(gx);
            const x1 = Math.min(grid.width - 1, x0 + 1);
            const wx = gx - x0;
            const samples = [
                [grid.meters[y0 * grid.width + x0], (1 - wx) * (1 - wy)],
                [grid.meters[y0 * grid.width + x1], wx * (1 - wy)],
                [grid.meters[y1 * grid.width + x0], (1 - wx) * wy],
                [grid.meters[y1 * grid.width + x1], wx * wy],
            ];
            let sum = 0;
            let weight = 0;
            for (const [d, w] of samples) {
                if (d > 0) {
                    sum += d * w;
                    weight += w;
                }
            }
            out[y * width + x] = weight > 0.5 ? sum / weight : 0;
        }
    }
    return out;
}

const halfTable = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
    const exponent = (h >> 10) & 31;
    const mantissa = h & 1023;
    const sign = h & 0x8000 ? -1 : 1;
    halfTable[h] =
        exponent === 0
            ? sign * Math.pow(2, -14) * (mantissa / 1024)
            : exponent === 31
              ? NaN
              : sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

/** IEEE half-precision bits → number. */
export function halfToFloat(bits: number): number {
    return halfTable[bits & 0xffff];
}

/**
 * Decodes one packed GPU depth sample (f16 depth | 6+6-bit octahedral normal | 4-bit cost) into
 * `out` = [depth, nx, ny, nz] in camera coordinates; false for an empty pixel.
 */
export function unpackDepthSample(value: number, out: Float64Array): boolean {
    if (value === 0) return false;
    const d = halfTable[value & 0xffff];
    if (!(d > 0)) return false;
    const ex = (((value >>> 16) & 63) / 63) * 2 - 1;
    const ey = (((value >>> 22) & 63) / 63) * 2 - 1;
    let nx = ex;
    let ny = ey;
    const nz = 1 - Math.abs(ex) - Math.abs(ey);
    if (nz < 0) {
        nx = (1 - Math.abs(ey)) * (ex >= 0 ? 1 : -1);
        ny = (1 - Math.abs(ex)) * (ey >= 0 ? 1 : -1);
    }
    const length = Math.hypot(nx, ny, nz) || 1;
    out[0] = d;
    out[1] = nx / length;
    out[2] = ny / length;
    out[3] = nz / length;
    return true;
}
