import { INTERVALS, Octave, SIGMA0, UPSCALE } from './scale-space';

export interface SiftKeypoint {
    octave: number;
    /** Sub-pixel position in octave pixels. */
    ox: number;
    oy: number;
    /** Fractional scale index within the octave (1 … INTERVALS). */
    scale: number;
    /** Blur of the keypoint in octave pixels. */
    sigma: number;
    /** Position in pixels of the input image. */
    x: number;
    y: number;
    /** |DoG| at the refined extremum. */
    score: number;
}

/** Minimum |DoG| contrast (image values in [0, 1]; Lowe uses 0.04 for well-lit photos). */
export const CONTRAST = 0.015;
/** Principal-curvature ratio above which an extremum lies on an edge. */
export const EDGE_RATIO = 10;
export const BORDER = 5;
const REFINE_STEPS = 5;

/**
 * Scale-space extrema of the DoG pyramid, refined to sub-pixel and sub-scale accuracy by a
 * quadratic fit (Brown & Lowe), with low-contrast and edge responses rejected.
 */
export function detectKeypoints(octaves: Octave[]): SiftKeypoint[] {
    const prefilter = (0.5 * CONTRAST) / INTERVALS;
    const keypoints: SiftKeypoint[] = [];
    octaves.forEach((octave, o) => {
        const { width, height, dogs } = octave;
        for (let s = 1; s <= INTERVALS; s++) {
            const below = dogs[s - 1];
            const current = dogs[s];
            const above = dogs[s + 1];
            for (let y = BORDER; y < height - BORDER; y++) {
                for (let x = BORDER; x < width - BORDER; x++) {
                    const i = y * width + x;
                    const v = current[i];
                    if (Math.abs(v) <= prefilter) continue;
                    if (!isExtremum(v, i, width, below, current, above)) continue;
                    const keypoint = refine(octave, o, s, x, y);
                    if (keypoint) keypoints.push(keypoint);
                }
            }
        }
    });
    return distinctKeypoints(keypoints);
}

/** Keypoint at its position in the input image, from the octave-relative refined values. */
export function keypointAt(
    octave: number,
    ox: number,
    oy: number,
    scale: number,
    score: number,
): SiftKeypoint {
    const factor = (1 << octave) / UPSCALE;
    return {
        octave,
        ox,
        oy,
        scale,
        sigma: SIGMA0 * Math.pow(2, scale / INTERVALS),
        x: ox * factor,
        y: oy * factor,
        score,
    };
}

/**
 * Refinement can move two starting pixels onto the same extremum; duplicates would fail every
 * ratio test, so only the strongest per octave pixel is kept, in a deterministic order.
 */
export function distinctKeypoints(keypoints: SiftKeypoint[]): SiftKeypoint[] {
    const best = new Map<string, SiftKeypoint>();
    for (const keypoint of keypoints) {
        const key = `${keypoint.octave},${Math.round(keypoint.ox)},${Math.round(keypoint.oy)}`;
        const current = best.get(key);
        if (!current || keypoint.score > current.score) best.set(key, keypoint);
    }
    return [...best.values()].sort(
        (a, b) => b.score - a.score || a.octave - b.octave || a.oy - b.oy || a.ox - b.ox,
    );
}

function isExtremum(
    v: number,
    i: number,
    width: number,
    below: Float32Array,
    current: Float32Array,
    above: Float32Array,
): boolean {
    for (const layer of [below, current, above]) {
        for (let dy = -width; dy <= width; dy += width) {
            for (let dx = -1; dx <= 1; dx++) {
                const n = layer[i + dy + dx];
                if (layer === current && dy === 0 && dx === 0) continue;
                if (v > 0 ? n >= v : n <= v) return false;
            }
        }
    }
    return true;
}

function refine(
    octave: Octave,
    o: number,
    s0: number,
    x0: number,
    y0: number,
): SiftKeypoint | null {
    const { width, height, dogs } = octave;
    let x = x0;
    let y = y0;
    let s = s0;
    let offset: [number, number, number] = [0, 0, 0];
    let gradient: [number, number, number] = [0, 0, 0];
    let dxx = 0;
    let dyy = 0;
    let dxy = 0;
    let converged = false;
    for (let step = 0; step < REFINE_STEPS; step++) {
        const b = dogs[s - 1];
        const c = dogs[s];
        const a = dogs[s + 1];
        const i = y * width + x;
        const value = c[i];
        gradient = [
            0.5 * (c[i + 1] - c[i - 1]),
            0.5 * (c[i + width] - c[i - width]),
            0.5 * (a[i] - b[i]),
        ];
        dxx = c[i + 1] + c[i - 1] - 2 * value;
        dyy = c[i + width] + c[i - width] - 2 * value;
        const dss = a[i] + b[i] - 2 * value;
        dxy = 0.25 * (c[i + width + 1] - c[i + width - 1] - c[i - width + 1] + c[i - width - 1]);
        const dxs = 0.25 * (a[i + 1] - a[i - 1] - b[i + 1] + b[i - 1]);
        const dys = 0.25 * (a[i + width] - a[i - width] - b[i + width] + b[i - width]);
        const solved = solve3([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss], gradient);
        if (!solved) return null;
        offset = [-solved[0], -solved[1], -solved[2]];
        if (Math.abs(offset[0]) < 0.5 && Math.abs(offset[1]) < 0.5 && Math.abs(offset[2]) < 0.5) {
            converged = true;
            break;
        }
        if (offset.some((d) => Math.abs(d) > 4)) return null;
        x += Math.round(offset[0]);
        y += Math.round(offset[1]);
        s += Math.round(offset[2]);
        if (s < 1 || s > INTERVALS || x < BORDER || x >= width - BORDER) return null;
        if (y < BORDER || y >= height - BORDER) return null;
    }
    if (!converged) return null;
    const i = y * width + x;
    const contrast =
        dogs[s][i] +
        0.5 * (gradient[0] * offset[0] + gradient[1] * offset[1] + gradient[2] * offset[2]);
    if (Math.abs(contrast) * INTERVALS < CONTRAST) return null;
    const trace = dxx + dyy;
    const determinant = dxx * dyy - dxy * dxy;
    if (determinant <= 0 || trace * trace * EDGE_RATIO >= (EDGE_RATIO + 1) ** 2 * determinant) {
        return null;
    }
    return keypointAt(o, x + offset[0], y + offset[1], s + offset[2], Math.abs(contrast));
}

/** Solves the symmetric 3×3 system H·x = b by Cramer's rule. */
function solve3(h: number[], b: number[]): [number, number, number] | null {
    const [a, d, e, , bb, f, , , c] = h;
    const det = a * (bb * c - f * f) - d * (d * c - f * e) + e * (d * f - bb * e);
    if (Math.abs(det) < 1e-12) return null;
    const x =
        (b[0] * (bb * c - f * f) - d * (b[1] * c - f * b[2]) + e * (b[1] * f - bb * b[2])) / det;
    const y =
        (a * (b[1] * c - f * b[2]) - b[0] * (d * c - f * e) + e * (d * b[2] - b[1] * e)) / det;
    const z =
        (a * (bb * b[2] - b[1] * f) - d * (d * b[2] - b[1] * e) + b[0] * (d * f - bb * e)) / det;
    return [x, y, z];
}
