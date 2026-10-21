import { DESCRIPTOR_WORDS, FeatureSet, hamming } from '../features/orb-features';
import { Intrinsics, Pose, fundamentalMatrix, relativePose } from '../geometry/camera';
import { Mat3 } from '../math/mat3';
import { seededRandom, symmetricEigen } from '../math/dense';

export interface MatchOptions {
    /** Maximum distance (px) of a candidate from the predicted epipolar line. */
    band: number;
    minDepth: number;
    maxDepth: number;
}

/** Matched feature indices as interleaved pairs [a0, b0, a1, b1, …]. */
export type MatchList = Int32Array;

const GRID_CELL = 16;
const MAX_HAMMING = 72;
const RATIO = 0.82;

/**
 * Descriptor matching constrained by the known camera poses: a feature of A is only compared
 * with features of B lying inside a band around its epipolar segment (bounded by the depth
 * range), then filtered with Lowe's ratio test and a one-to-one check.
 */
export function guidedMatch(
    a: FeatureSet,
    b: FeatureSet,
    ka: Intrinsics,
    kb: Intrinsics,
    poseA: Pose,
    poseB: Pose,
    width: number,
    height: number,
    options: MatchOptions,
): MatchList {
    const columns = Math.ceil(width / GRID_CELL);
    const rows = Math.ceil(height / GRID_CELL);
    const cellStart = new Int32Array(columns * rows + 1);
    const cellOf = new Int32Array(b.count);
    for (let j = 0; j < b.count; j++) {
        const cx = Math.min(columns - 1, Math.max(0, Math.floor(b.x[j] / GRID_CELL)));
        const cy = Math.min(rows - 1, Math.max(0, Math.floor(b.y[j] / GRID_CELL)));
        cellOf[j] = cy * columns + cx;
        cellStart[cellOf[j] + 1]++;
    }
    for (let c = 0; c < columns * rows; c++) cellStart[c + 1] += cellStart[c];
    const fill = cellStart.slice(0, -1);
    const cellItems = new Int32Array(b.count);
    for (let j = 0; j < b.count; j++) cellItems[fill[cellOf[j]]++] = j;

    const F = fundamentalMatrix(ka, kb, poseA, poseB);
    const { R, t } = relativePose(poseA, poseB);
    const stamp = new Int32Array(columns * rows).fill(-1);
    const reach = Math.ceil(options.band / GRID_CELL);

    const bestFor = new Int32Array(a.count).fill(-1);
    const bestDistance = new Int32Array(a.count);

    for (let i = 0; i < a.count; i++) {
        const xa = a.x[i];
        const ya = a.y[i];
        // Epipolar line in B, normalised so that l·(x, y, 1) is a distance in pixels.
        let l0 = F[0] * xa + F[1] * ya + F[2];
        let l1 = F[3] * xa + F[4] * ya + F[5];
        let l2 = F[6] * xa + F[7] * ya + F[8];
        const ln = Math.hypot(l0, l1);
        if (ln < 1e-12) continue;
        l0 /= ln;
        l1 /= ln;
        l2 /= ln;

        // Ray of the feature in camera A, mapped to camera B: X_b(d) = d · (R·ray) + t.
        const rx = (xa - ka.cx) / ka.fx;
        const ry = (ya - ka.cy) / ka.fy;
        const qx = R[0] * rx + R[1] * ry + R[2];
        const qy = R[3] * rx + R[4] * ry + R[5];
        const qz = R[6] * rx + R[7] * ry + R[8];
        let near = options.minDepth;
        let far = options.maxDepth;
        // Keep the part of the ray in front of camera B (z_b ≥ 2 cm).
        if (Math.abs(qz) < 1e-12) {
            if (t[2] < 0.02) continue;
        } else {
            const limit = (0.02 - t[2]) / qz;
            if (qz > 0) near = Math.max(near, limit);
            else far = Math.min(far, limit);
        }
        if (near >= far) continue;
        const projectAt = (d: number): [number, number] => {
            const z = d * qz + t[2];
            return [(kb.fx * (d * qx + t[0])) / z + kb.cx, (kb.fy * (d * qy + t[1])) / z + kb.cy];
        };
        const [sx, sy] = projectAt(near);
        const [ex, ey] = projectAt(far);
        const segment = clipSegment(sx, sy, ex, ey, width, height, options.band);
        if (!segment) continue;

        let best = 1 << 30;
        let second = 1 << 30;
        let bestIndex = -1;
        const [x0, y0, x1, y1] = segment;
        const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (GRID_CELL / 2)));
        for (let s = 0; s <= steps; s++) {
            const px = x0 + ((x1 - x0) * s) / steps;
            const py = y0 + ((y1 - y0) * s) / steps;
            const cx = Math.floor(px / GRID_CELL);
            const cy = Math.floor(py / GRID_CELL);
            for (let gy = cy - reach; gy <= cy + reach; gy++) {
                if (gy < 0 || gy >= rows) continue;
                for (let gx = cx - reach; gx <= cx + reach; gx++) {
                    if (gx < 0 || gx >= columns) continue;
                    const cell = gy * columns + gx;
                    if (stamp[cell] === i) continue;
                    stamp[cell] = i;
                    for (let k = cellStart[cell]; k < cellStart[cell + 1]; k++) {
                        const j = cellItems[k];
                        if (Math.abs(a.level[i] - b.level[j]) > 1) continue;
                        if (Math.abs(l0 * b.x[j] + l1 * b.y[j] + l2) > options.band) continue;
                        const d = hamming(
                            a.descriptors,
                            i * DESCRIPTOR_WORDS,
                            b.descriptors,
                            j * DESCRIPTOR_WORDS,
                        );
                        if (d < best) {
                            second = best;
                            best = d;
                            bestIndex = j;
                        } else if (d < second) {
                            second = d;
                        }
                    }
                }
            }
        }
        if (bestIndex < 0 || best > MAX_HAMMING || best > RATIO * second) continue;
        bestFor[i] = bestIndex;
        bestDistance[i] = best;
    }

    // One-to-one: each feature of B keeps only its closest feature of A.
    const owner = new Int32Array(b.count).fill(-1);
    for (let i = 0; i < a.count; i++) {
        const j = bestFor[i];
        if (j < 0) continue;
        const current = owner[j];
        if (current < 0 || bestDistance[i] < bestDistance[current]) owner[j] = i;
    }
    const pairs: number[] = [];
    for (let j = 0; j < b.count; j++) if (owner[j] >= 0) pairs.push(owner[j], j);
    return Int32Array.from(pairs);
}

/** Liang–Barsky clip of a segment against the image expanded by a margin. */
function clipSegment(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    width: number,
    height: number,
    margin: number,
): [number, number, number, number] | null {
    if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
    const dx = x1 - x0;
    const dy = y1 - y0;
    let t0 = 0;
    let t1 = 1;
    const edges: [number, number][] = [
        [-dx, x0 + margin],
        [dx, width - 1 + margin - x0],
        [-dy, y0 + margin],
        [dy, height - 1 + margin - y0],
    ];
    for (const [p, q] of edges) {
        if (Math.abs(p) < 1e-12) {
            if (q < 0) return null;
            continue;
        }
        const r = q / p;
        if (p < 0) t0 = Math.max(t0, r);
        else t1 = Math.min(t1, r);
        if (t0 > t1) return null;
    }
    return [x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy];
}

/**
 * Robust re-estimation of F (normalised 8-point inside RANSAC) on already-guided matches;
 * returns the matches consistent with it. Used where the pose prior is loose (loop pairs).
 */
export function fundamentalInliers(
    a: FeatureSet,
    b: FeatureSet,
    matches: MatchList,
    threshold: number,
    seed = 1,
): MatchList {
    const n = matches.length / 2;
    if (n < 12) return new Int32Array(0);
    const pa = new Float64Array(n * 2);
    const pb = new Float64Array(n * 2);
    for (let m = 0; m < n; m++) {
        pa[m * 2] = a.x[matches[m * 2]];
        pa[m * 2 + 1] = a.y[matches[m * 2]];
        pb[m * 2] = b.x[matches[m * 2 + 1]];
        pb[m * 2 + 1] = b.y[matches[m * 2 + 1]];
    }
    const na = normaliser(pa);
    const nb = normaliser(pb);
    const random = seededRandom(seed);
    const limit = threshold * threshold;
    let bestCount = 0;
    let bestMask: Uint8Array | null = null;
    const sample = new Int32Array(8);
    for (let iteration = 0; iteration < 256; iteration++) {
        for (let s = 0; s < 8; s++) {
            let pick: number;
            do pick = Math.floor(random() * n);
            while (sample.subarray(0, s).includes(pick));
            sample[s] = pick;
        }
        const F = eightPoint(pa, pb, sample, na, nb);
        if (!F) continue;
        const mask = new Uint8Array(n);
        let count = 0;
        for (let m = 0; m < n; m++) {
            if (sampson(F, pa[m * 2], pa[m * 2 + 1], pb[m * 2], pb[m * 2 + 1]) < limit) {
                mask[m] = 1;
                count++;
            }
        }
        if (count > bestCount) {
            bestCount = count;
            bestMask = mask;
            if (count > n * 0.9) break;
        }
    }
    if (!bestMask || bestCount < 12) return new Int32Array(0);
    const kept: number[] = [];
    for (let m = 0; m < n; m++) if (bestMask[m]) kept.push(matches[m * 2], matches[m * 2 + 1]);
    return Int32Array.from(kept);
}

interface Normaliser {
    mx: number;
    my: number;
    s: number;
}

function normaliser(points: Float64Array): Normaliser {
    const n = points.length / 2;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
        mx += points[i * 2];
        my += points[i * 2 + 1];
    }
    mx /= n;
    my /= n;
    let spread = 0;
    for (let i = 0; i < n; i++) spread += Math.hypot(points[i * 2] - mx, points[i * 2 + 1] - my);
    return { mx, my, s: Math.SQRT2 / Math.max(1e-9, spread / n) };
}

function eightPoint(
    pa: Float64Array,
    pb: Float64Array,
    sample: Int32Array,
    na: Normaliser,
    nb: Normaliser,
): Mat3 | null {
    const ata = new Float64Array(81);
    const row = new Float64Array(9);
    for (const m of sample) {
        const x = (pa[m * 2] - na.mx) * na.s;
        const y = (pa[m * 2 + 1] - na.my) * na.s;
        const u = (pb[m * 2] - nb.mx) * nb.s;
        const v = (pb[m * 2 + 1] - nb.my) * nb.s;
        row.set([u * x, u * y, u, v * x, v * y, v, x, y, 1]);
        for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) ata[r * 9 + c] += row[r] * row[c];
    }
    const f = symmetricEigen(ata, 9).vectors.slice(0, 9);
    // Rank-2 projection: F ← F − (F v)(v)ᵀ with v the null direction of FᵀF.
    const ftf = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            ftf[r * 3 + c] = f[r] * f[c] + f[3 + r] * f[3 + c] + f[6 + r] * f[6 + c];
        }
    }
    const v = symmetricEigen(ftf, 3).vectors.slice(0, 3);
    const fv = [
        f[0] * v[0] + f[1] * v[1] + f[2] * v[2],
        f[3] * v[0] + f[4] * v[1] + f[5] * v[2],
        f[6] * v[0] + f[7] * v[1] + f[8] * v[2],
    ];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) f[r * 3 + c] -= fv[r] * v[c];
    // Undo the normalisation: F = Tbᵀ · F̂ · Ta.
    const ta = [na.s, 0, -na.s * na.mx, 0, na.s, -na.s * na.my, 0, 0, 1];
    const tb = [nb.s, 0, -nb.s * nb.mx, 0, nb.s, -nb.s * nb.my, 0, 0, 1];
    const out = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            let sum = 0;
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) sum += tb[i * 3 + r] * f[i * 3 + j] * ta[j * 3 + c];
            }
            out[r * 3 + c] = sum;
        }
    }
    return out.every(Number.isFinite) ? out : null;
}

function sampson(F: Mat3, xa: number, ya: number, xb: number, yb: number): number {
    const fx0 = F[0] * xa + F[1] * ya + F[2];
    const fx1 = F[3] * xa + F[4] * ya + F[5];
    const fx2 = F[6] * xa + F[7] * ya + F[8];
    const ft0 = F[0] * xb + F[3] * yb + F[6];
    const ft1 = F[1] * xb + F[4] * yb + F[7];
    const e = xb * fx0 + yb * fx1 + fx2;
    return (e * e) / (fx0 * fx0 + fx1 * fx1 + ft0 * ft0 + ft1 * ft1 + 1e-12);
}
