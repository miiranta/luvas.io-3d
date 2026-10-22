import { cameraCenter, opticalAxis, project } from '../geometry/camera';
import { quantile } from '../math/dense';
import { dot3 } from '../math/mat3';
import { SparseMap } from '../sfm/sparse-map';

const IDEAL_ANGLE = (10 * Math.PI) / 180;
const SIGMA_BELOW = (4 * Math.PI) / 180;
const SIGMA_ABOVE = (15 * Math.PI) / 180;

/**
 * Source views for multi-view stereo, scored by the sparse points they share with the
 * reference, each weighted by how close its triangulation angle is to ~10° (too small gives
 * no depth resolution, too large gives distorted patches). Falls back to pose geometry when
 * the reference has too few tracked points (low texture).
 */
export function selectSources(
    map: SparseMap,
    ref: number,
    count: number,
    allowed?: ReadonlySet<number>,
): number[] {
    const frame = map.frames[ref];
    const center = cameraCenter(frame.pose);
    const scores = new Float64Array(map.frames.length);
    const centers = map.frames.map((f) => cameraCenter(f.pose));
    for (let f = 0; f < frame.track.length; f++) {
        const id = frame.track[f];
        if (id < 0) continue;
        const track = map.tracks[id];
        if (!track.point) continue;
        const p = track.point;
        const a = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        const la = Math.hypot(a[0], a[1], a[2]);
        for (let o = 0; o < track.obs.length; o += 2) {
            const other = track.obs[o];
            if (other === ref || (allowed && !allowed.has(other))) continue;
            const c = centers[other];
            const b = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            const cos = dot3(a, b) / (la * Math.hypot(b[0], b[1], b[2]));
            const angle = Math.acos(Math.min(1, cos));
            if (angle < (1 * Math.PI) / 180) continue;
            const sigma = angle < IDEAL_ANGLE ? SIGMA_BELOW : SIGMA_ABOVE;
            scores[other] += Math.exp(-((angle - IDEAL_ANGLE) ** 2) / (2 * sigma * sigma));
        }
    }
    const ranked = [...scores.keys()]
        .filter((j) => j !== ref && scores[j] > 0.5)
        .sort((a, b) => scores[b] - scores[a]);
    if (ranked.length >= 2) return ranked.slice(0, count);

    // Pose-only fallback: similar viewing direction with a useful baseline.
    const axis = opticalAxis(frame.pose);
    const byPose = map.frames
        .map((other, j) => {
            const angle = Math.acos(Math.min(1, dot3(axis, opticalAxis(other.pose))));
            const baseline = map.baselineRatio(ref, j);
            return { j, angle, baseline };
        })
        .filter(
            ({ j, angle, baseline }) =>
                j !== ref &&
                (!allowed || allowed.has(j)) &&
                angle < (40 * Math.PI) / 180 &&
                baseline > 0.03,
        )
        .sort((a, b) => Math.abs(a.baseline - 0.15) - Math.abs(b.baseline - 0.15));
    return [...new Set([...ranked, ...byPose.map(({ j }) => j)])].slice(0, count);
}

/** Depth search interval of a view from its tracked points and/or the phone depth. */
export function depthRange(
    map: SparseMap,
    ref: number,
    prior: Float32Array | null,
): [number, number] {
    const frame = map.frames[ref];
    const depths: number[] = [];
    for (let f = 0; f < frame.track.length; f++) {
        const id = frame.track[f];
        if (id < 0) continue;
        const point = map.tracks[id].point;
        if (!point) continue;
        const z = project(frame.intrinsics, frame.pose, point)[2];
        if (z > 0) depths.push(z);
    }
    if (prior) {
        for (let i = 0; i < prior.length; i += 7) if (prior[i] > 0) depths.push(prior[i]);
    }
    if (depths.length < 10) return [0.08, 8];
    const near = quantile(depths, 0.02) * 0.6;
    const far = quantile(depths, 0.98) * 1.5;
    return [Math.max(0.05, near), Math.min(20, Math.max(far, near * 2))];
}
