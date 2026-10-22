import { solveCholesky } from '../../math/dense';
import { FaceCandidates } from '../face-visibility';
import { TextureView } from '../texture-types';
import { sampleThumb } from '../thumbnail';

/** Per-view RGB gains from faces seen by several views (robust log-domain least squares). */
export function estimateGains(
    positions: Float32Array,
    indices: Uint32Array,
    candidates: FaceCandidates,
    views: TextureView[],
): Float32Array {
    const n = views.length;
    const faces = indices.length / 3;
    const stride = Math.max(1, Math.floor(faces / 30000));
    const samples: { view: number; log: Float64Array }[][] = [];
    const color = new Float64Array(3);
    for (let f = 0; f < faces; f += stride) {
        const a = indices[f * 3] * 3;
        const b = indices[f * 3 + 1] * 3;
        const c = indices[f * 3 + 2] * 3;
        const x = (positions[a] + positions[b] + positions[c]) / 3;
        const y = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3;
        const z = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3;
        const seen: { view: number; log: Float64Array }[] = [];
        for (let s = 0; s < candidates.slots; s++) {
            const view = candidates.view[f * candidates.slots + s];
            if (view < 0) break;
            if (!sampleThumb(views[view], x, y, z, color)) continue;
            if (
                Math.min(color[0], color[1], color[2]) < 8 ||
                Math.max(color[0], color[1], color[2]) > 248
            ) {
                continue;
            }
            seen.push({ view, log: Float64Array.from(color, (value) => Math.log(value)) });
        }
        if (seen.length >= 2) samples.push(seen);
    }
    const gains = new Float32Array(n * 3).fill(1);
    if (samples.length < 50) return gains;
    for (let channel = 0; channel < 3; channel++) {
        let x: Float64Array = new Float64Array(n);
        for (let round = 0; round < 3; round++) {
            const A = new Float64Array(n * n);
            const rhs = new Float64Array(n);
            for (const seen of samples) {
                for (let i = 0; i < seen.length; i++) {
                    for (let j = i + 1; j < seen.length; j++) {
                        const a = seen[i].view;
                        const b = seen[j].view;
                        // x_a − x_b ≈ log c_b − log c_a
                        const target = seen[j].log[channel] - seen[i].log[channel];
                        const residual = x[a] - x[b] - target;
                        const w = Math.abs(residual) <= 0.08 ? 1 : 0.08 / Math.abs(residual);
                        A[a * n + a] += w;
                        A[b * n + b] += w;
                        A[a * n + b] -= w;
                        A[b * n + a] -= w;
                        rhs[a] += w * target;
                        rhs[b] -= w * target;
                    }
                }
            }
            for (let i = 0; i < n; i++) A[i * n + i] += 0.5;
            const solved = solveCholesky(A, rhs, n);
            if (!solved) break;
            x = solved;
        }
        let mean = 0;
        for (const value of x) mean += value;
        mean /= n;
        for (let i = 0; i < n; i++)
            gains[i * 3 + channel] = Math.exp(Math.max(-0.7, Math.min(0.7, x[i] - mean)));
    }
    return gains;
}
