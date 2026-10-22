import { RgbaImage } from '../texture-types';
import { AtlasLayout } from './chart-packing';
import { Chart } from './charts';

/** Copies a chart's pixels from its source image into the atlas (gains + seam correction). */
export function bakeChart(
    chart: Chart,
    image: RgbaImage,
    gains: Float32Array,
    view: number,
    atlas: AtlasLayout,
    pixels: Uint8ClampedArray,
): void {
    const s = atlas.scale;
    const w = Math.ceil(chart.w * s) + 1;
    const h = Math.ceil(chart.h * s) + 1;
    // Correction raster: barycentric interpolation of the vertex offsets, then dilated.
    const correction = new Float32Array(w * h * 3);
    const covered = new Uint8Array(w * h);
    rasterizeCorrections(chart, s, w, h, correction, covered);
    dilate(correction, covered, w, h, 3);

    const g0 = gains[view * 3];
    const g1 = gains[view * 3 + 1];
    const g2 = gains[view * 3 + 2];
    for (let y = 0; y < h; y++) {
        const sy = Math.min(image.height - 1.001, Math.max(0, chart.y0 + (y + 0.5) / s - 0.5));
        const iy = Math.floor(sy);
        const wy = sy - iy;
        for (let x = 0; x < w; x++) {
            const ax = chart.ax + x;
            const ay = chart.ay + y;
            if (ax >= atlas.width || ay >= atlas.height) continue;
            const sx = Math.min(image.width - 1.001, Math.max(0, chart.x0 + (x + 0.5) / s - 0.5));
            const ix = Math.floor(sx);
            const wx = sx - ix;
            const i = (iy * image.width + ix) * 4;
            const j = i + image.width * 4;
            const o = (ay * atlas.width + ax) * 4;
            const c = (y * w + x) * 3;
            for (let ch = 0; ch < 3; ch++) {
                const top = image.data[i + ch] * (1 - wx) + image.data[i + 4 + ch] * wx;
                const bottom = image.data[j + ch] * (1 - wx) + image.data[j + 4 + ch] * wx;
                const gain = ch === 0 ? g0 : ch === 1 ? g1 : g2;
                pixels[o + ch] = (top * (1 - wy) + bottom * wy) * gain + correction[c + ch];
            }
        }
    }
}

function rasterizeCorrections(
    chart: Chart,
    s: number,
    w: number,
    h: number,
    out: Float32Array,
    covered: Uint8Array,
): void {
    const px = (i: number) => (chart.uv[i * 2] - chart.x0 + 0.5) * s;
    const py = (i: number) => (chart.uv[i * 2 + 1] - chart.y0 + 0.5) * s;
    for (let f = 0; f < chart.corners.length; f += 3) {
        const a = chart.corners[f];
        const b = chart.corners[f + 1];
        const c = chart.corners[f + 2];
        const ax = px(a);
        const ay = py(a);
        const bx = px(b);
        const by = py(b);
        const cx = px(c);
        const cy = py(c);
        const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (Math.abs(area) < 1e-9) continue;
        const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
        const x1 = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
        const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
        const y1 = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const qx = x + 0.5;
                const qy = y + 0.5;
                const l0 = ((bx - qx) * (cy - qy) - (by - qy) * (cx - qx)) / area;
                const l1 = ((cx - qx) * (ay - qy) - (cy - qy) * (ax - qx)) / area;
                const l2 = 1 - l0 - l1;
                if (l0 < -0.01 || l1 < -0.01 || l2 < -0.01) continue;
                const o = (y * w + x) * 3;
                for (let ch = 0; ch < 3; ch++) {
                    out[o + ch] =
                        l0 * chart.correction[a * 3 + ch] +
                        l1 * chart.correction[b * 3 + ch] +
                        l2 * chart.correction[c * 3 + ch];
                }
                covered[y * w + x] = 1;
            }
        }
    }
}

function dilate(
    values: Float32Array,
    covered: Uint8Array,
    w: number,
    h: number,
    steps: number,
): void {
    for (let step = 0; step < steps; step++) {
        const grown = covered.slice();
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = y * w + x;
                if (covered[i]) continue;
                let n = 0;
                let r = 0;
                let g = 0;
                let b = 0;
                for (const [dx, dy] of [
                    [1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                ]) {
                    const xx = x + dx;
                    const yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h || !covered[yy * w + xx]) continue;
                    const j = (yy * w + xx) * 3;
                    r += values[j];
                    g += values[j + 1];
                    b += values[j + 2];
                    n++;
                }
                if (n === 0) continue;
                values[i * 3] = r / n;
                values[i * 3 + 1] = g / n;
                values[i * 3 + 2] = b / n;
                grown[i] = 1;
            }
        }
        covered.set(grown);
    }
}
