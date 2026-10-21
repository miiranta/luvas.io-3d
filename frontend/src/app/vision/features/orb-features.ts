import { seededRandom } from '../math/dense';
import { GrayImage, blurGray, resizeGray } from '../image/gray-image';

/** Keypoints and 256-bit descriptors of one image, stored as parallel arrays. */
export interface FeatureSet {
    count: number;
    /** Positions in pixels of the full-resolution (level 0) image. */
    x: Float32Array;
    y: Float32Array;
    level: Uint8Array;
    angle: Float32Array;
    /** 8 × u32 words per keypoint. */
    descriptors: Uint32Array;
}

export const DESCRIPTOR_WORDS = 8;

const LEVELS = 3;
const LEVEL_SCALE = 1.4;
const PATCH_RADIUS = 15;
const BORDER = 20; // a rotated pattern point reaches 13·√2 ≈ 18.4 px
const CELL = 24;
const HARRIS_K = 0.04;
const THRESHOLDS = [18, 10, 5];

const CIRCLE = [
    [0, -3],
    [1, -3],
    [2, -2],
    [3, -1],
    [3, 0],
    [3, 1],
    [2, 2],
    [1, 3],
    [0, 3],
    [-1, 3],
    [-2, 2],
    [-3, 1],
    [-3, 0],
    [-3, -1],
    [-2, -2],
    [-1, -3],
];

/** Gaussian BRIEF test pattern (σ = patch / 5), fixed seed so descriptors are comparable. */
const PATTERN: Int8Array = (() => {
    const random = seededRandom(0x5eed);
    const gaussian = () => {
        const u = Math.max(1e-9, random());
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
    };
    const pattern = new Int8Array(256 * 4);
    for (let i = 0; i < pattern.length; i++) {
        let value: number;
        do value = Math.round(gaussian() * (31 / 5));
        while (Math.abs(value) > 13);
        pattern[i] = value;
    }
    return pattern;
})();

/** Half-widths of the circular orientation patch per row. */
const UMAX: Int32Array = (() => {
    const umax = new Int32Array(PATCH_RADIUS + 1);
    for (let v = 0; v <= PATCH_RADIUS; v++) {
        umax[v] = Math.floor(Math.sqrt(PATCH_RADIUS * PATCH_RADIUS + 0.5 - v * v));
    }
    return umax;
})();

interface Candidate {
    x: number;
    y: number;
    score: number;
}

/** ORB-style features: FAST-9 corners ranked by Harris, spread on a grid, rBRIEF descriptors. */
export function extractFeatures(image: GrayImage, budget: number): FeatureSet {
    const levels: GrayImage[] = [image];
    for (let l = 1; l < LEVELS; l++) {
        const scale = Math.pow(LEVEL_SCALE, l);
        levels.push(
            resizeGray(image, Math.round(image.width / scale), Math.round(image.height / scale)),
        );
    }
    const areaWeights = levels.map((_, l) => Math.pow(LEVEL_SCALE, -2 * l));
    const areaTotal = areaWeights.reduce((sum, w) => sum + w, 0);

    const xs: number[] = [];
    const ys: number[] = [];
    const lv: number[] = [];
    const angles: number[] = [];
    const words: number[] = [];

    levels.forEach((level, l) => {
        const levelBudget = Math.round((budget * areaWeights[l]) / areaTotal);
        const blurred = blurGray(level);
        const picked = selectOnGrid(detectCorners(level, blurred, levelBudget), level, levelBudget);
        const scale = Math.pow(LEVEL_SCALE, l);
        for (const { x, y } of picked) {
            const angle = orientation(level, x, y);
            xs.push((x + 0.5) * scale - 0.5);
            ys.push((y + 0.5) * scale - 0.5);
            lv.push(l);
            angles.push(angle);
            describe(blurred, x, y, angle, words);
        }
    });

    return {
        count: xs.length,
        x: Float32Array.from(xs),
        y: Float32Array.from(ys),
        level: Uint8Array.from(lv),
        angle: Float32Array.from(angles),
        descriptors: Uint32Array.from(words),
    };
}

function detectCorners(image: GrayImage, blurred: GrayImage, budget: number): Candidate[] {
    const { width, height, data } = image;
    const offsets = CIRCLE.map(([dx, dy]) => dy * width + dx);
    let corners: Candidate[] = [];
    for (const threshold of THRESHOLDS) {
        corners = [];
        const score = new Float32Array(width * height);
        for (let y = BORDER; y < height - BORDER; y++) {
            for (let x = BORDER; x < width - BORDER; x++) {
                const i = y * width + x;
                if (isCorner(data, i, offsets, threshold)) score[i] = harris(blurred, x, y);
            }
        }
        for (let y = BORDER; y < height - BORDER; y++) {
            for (let x = BORDER; x < width - BORDER; x++) {
                const i = y * width + x;
                const s = score[i];
                if (s <= 0) continue;
                if (
                    s < score[i - 1] ||
                    s < score[i + 1] ||
                    s < score[i - width] ||
                    s < score[i + width] ||
                    s < score[i - width - 1] ||
                    s < score[i - width + 1] ||
                    s < score[i + width - 1] ||
                    s < score[i + width + 1]
                ) {
                    continue;
                }
                corners.push({ x, y, score: s });
            }
        }
        if (corners.length >= budget * 1.5) break;
    }
    return corners;
}

function isCorner(data: Uint8Array, i: number, offsets: number[], threshold: number): boolean {
    const center = data[i];
    const high = center + threshold;
    const low = center - threshold;
    let compassBright = 0;
    let compassDark = 0;
    for (let k = 0; k < 16; k += 4) {
        const value = data[i + offsets[k]];
        if (value > high) compassBright++;
        else if (value < low) compassDark++;
    }
    if (compassBright < 2 && compassDark < 2) return false;
    let bright = 0;
    let dark = 0;
    for (let k = 0; k < 16; k++) {
        const value = data[i + offsets[k]];
        if (value > high) bright |= 1 << k;
        else if (value < low) dark |= 1 << k;
    }
    return hasArc(bright) || hasArc(dark);
}

/** True when the circular 16-bit mask holds 9 consecutive set bits. */
function hasArc(mask: number): boolean {
    if (mask === 0) return false;
    let m = mask | (mask << 16);
    for (let k = 0; k < 8; k++) m &= m >>> 1;
    return (m & 0xffff) !== 0;
}

function harris(image: GrayImage, x: number, y: number): number {
    const { width, data } = image;
    let xx = 0;
    let yy = 0;
    let xy = 0;
    for (let dy = -3; dy <= 3; dy++) {
        let i = (y + dy) * width + x - 3;
        for (let dx = -3; dx <= 3; dx++, i++) {
            const gx = data[i + 1] - data[i - 1];
            const gy = data[i + width] - data[i - width];
            xx += gx * gx;
            yy += gy * gy;
            xy += gx * gy;
        }
    }
    const response = xx * yy - xy * xy - HARRIS_K * (xx + yy) * (xx + yy);
    return response > 0 ? response : 1e-6;
}

/** Strongest corners per grid cell first, then the best of the rest, up to the budget. */
function selectOnGrid(corners: Candidate[], image: GrayImage, budget: number): Candidate[] {
    corners.sort((a, b) => b.score - a.score);
    const columns = Math.ceil(image.width / CELL);
    const rows = Math.ceil(image.height / CELL);
    const perCell = Math.max(1, Math.ceil((budget / (columns * rows)) * 1.2));
    const filled = new Uint16Array(columns * rows);
    const chosen: Candidate[] = [];
    const rest: Candidate[] = [];
    for (const corner of corners) {
        const cell = Math.floor(corner.y / CELL) * columns + Math.floor(corner.x / CELL);
        if (filled[cell] < perCell && chosen.length < budget) {
            filled[cell]++;
            chosen.push(corner);
        } else {
            rest.push(corner);
        }
    }
    for (const corner of rest) {
        if (chosen.length >= budget) break;
        chosen.push(corner);
    }
    return chosen;
}

/** Intensity-centroid orientation (Rosin) over a circular patch. */
function orientation(image: GrayImage, x: number, y: number): number {
    const { width, data } = image;
    const center = y * width + x;
    let m01 = 0;
    let m10 = 0;
    for (let u = -PATCH_RADIUS; u <= PATCH_RADIUS; u++) m10 += u * data[center + u];
    for (let v = 1; v <= PATCH_RADIUS; v++) {
        let sum = 0;
        const d = UMAX[v];
        for (let u = -d; u <= d; u++) {
            const below = data[center + v * width + u];
            const above = data[center - v * width + u];
            sum += below - above;
            m10 += u * (below + above);
        }
        m01 += v * sum;
    }
    return Math.atan2(m01, m10);
}

function describe(image: GrayImage, x: number, y: number, angle: number, out: number[]): void {
    const { width, data } = image;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const center = y * width + x;
    const sample = (px: number, py: number) =>
        data[center + Math.round(px * s + py * c) * width + Math.round(px * c - py * s)];
    for (let word = 0; word < DESCRIPTOR_WORDS; word++) {
        let bits = 0;
        for (let bit = 0; bit < 32; bit++) {
            const p = (word * 32 + bit) * 4;
            if (sample(PATTERN[p], PATTERN[p + 1]) < sample(PATTERN[p + 2], PATTERN[p + 3])) {
                bits |= 1 << bit;
            }
        }
        out.push(bits >>> 0);
    }
}

export function hamming(a: Uint32Array, ia: number, b: Uint32Array, ib: number): number {
    let distance = 0;
    for (let w = 0; w < DESCRIPTOR_WORDS; w++) {
        let v = a[ia + w] ^ b[ib + w];
        v = v - ((v >>> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
        distance += (Math.imul((v + (v >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24) as number;
    }
    return distance;
}
