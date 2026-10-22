import { FeatureSet } from '../features/feature-set';
import { Mat3 } from '../math/mat3';
import { seededRandom, symmetricEigen } from '../math/dense';
import { MatchList } from './match-list';

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
