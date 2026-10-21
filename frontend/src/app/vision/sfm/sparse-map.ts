import { FeatureSet } from '../features/orb-features';
import { Intrinsics, Pose, cameraCenter, opticalAxis, project } from '../geometry/camera';
import { Observation, reprojectionError, triangulate } from '../geometry/triangulate';
import { MatchList } from '../matching/guided-matcher';
import { dot3, norm3, sub3 } from '../math/mat3';
import { BundleProblem } from './bundle-adjuster';

/** A keyframe as seen by the sparse mapper (work resolution). */
export interface MapFrame {
    intrinsics: Intrinsics;
    width: number;
    height: number;
    /** Current estimate, refined by bundle adjustment. */
    pose: Pose;
    /** Pose reported by the phone; never modified. */
    prior: Pose;
    features: FeatureSet;
    /** Track id per feature, −1 when unmatched. */
    track: Int32Array;
    /** Packed RGB thumbnail used to colour points. */
    rgb: Uint8Array;
    thumbWidth: number;
    thumbHeight: number;
    /** Capture time in seconds. */
    time: number;
    /** Median scene depth seen by this frame (metres). */
    depthHint: number;
}

export interface Track {
    /** Interleaved (frame, feature) pairs. */
    obs: number[];
    point: Float64Array | null;
    color: [number, number, number];
    /** RMS reprojection error of the last triangulation (px). */
    error: number;
    alive: boolean;
}

const MIN_ANGLE = (1.5 * Math.PI) / 180;
const MAX_ERROR = 4;

export class SparseMap {
    readonly frames: MapFrame[] = [];
    readonly tracks: Track[] = [];
    readonly matched = new Set<string>();

    addFrame(frame: MapFrame): number {
        this.frames.push(frame);
        return this.frames.length - 1;
    }

    /**
     * Earlier frames likely to see the same surfaces as frame `index`, judged from the poses
     * alone: similar viewing direction and each frame's look-at point inside the other image.
     */
    covisible(index: number, limit: number, recent = 3): number[] {
        const frame = this.frames[index];
        const picked: number[] = [];
        for (let j = index - 1; j >= Math.max(0, index - recent); j--) picked.push(j);
        const others: { j: number; angle: number }[] = [];
        for (let j = 0; j < index - recent; j++) {
            const other = this.frames[j];
            const angle = Math.acos(
                Math.min(1, dot3(opticalAxis(frame.pose), opticalAxis(other.pose))),
            );
            if (angle > (55 * Math.PI) / 180) continue;
            if (!this.looksInto(frame, other) || !this.looksInto(other, frame)) continue;
            others.push({ j, angle });
        }
        others.sort((a, b) => a.angle - b.angle);
        // Spread the extra pairs over time instead of taking a run of consecutive frames.
        for (const { j } of others) {
            if (picked.length >= limit) break;
            if (picked.some((p) => Math.abs(p - j) < 2)) continue;
            picked.push(j);
        }
        return picked.filter((j) => !this.matched.has(pairKey(index, j)));
    }

    private looksInto(from: MapFrame, into: MapFrame): boolean {
        const center = cameraCenter(from.pose);
        const axis = opticalAxis(from.pose);
        const target = [
            center[0] + axis[0] * from.depthHint,
            center[1] + axis[1] * from.depthHint,
            center[2] + axis[2] * from.depthHint,
        ];
        const [u, v, z] = project(into.intrinsics, into.pose, target);
        const mx = into.width * 0.1;
        const my = into.height * 0.1;
        return z > 0 && u > -mx && u < into.width + mx && v > -my && v < into.height + my;
    }

    /** Merges matches between two frames into the track graph; returns the touched tracks. */
    link(i: number, j: number, matches: MatchList): Set<number> {
        this.matched.add(pairKey(i, j));
        const touched = new Set<number>();
        const fi = this.frames[i];
        const fj = this.frames[j];
        for (let m = 0; m < matches.length; m += 2) {
            const a = matches[m];
            const b = matches[m + 1];
            const ta = fi.track[a];
            const tb = fj.track[b];
            if (ta < 0 && tb < 0) {
                const id = this.tracks.length;
                this.tracks.push({
                    obs: [i, a, j, b],
                    point: null,
                    color: [128, 128, 128],
                    error: 0,
                    alive: true,
                });
                fi.track[a] = id;
                fj.track[b] = id;
                touched.add(id);
            } else if (ta >= 0 && tb < 0) {
                if (this.extend(ta, j, b)) touched.add(ta);
            } else if (ta < 0 && tb >= 0) {
                if (this.extend(tb, i, a)) touched.add(tb);
            } else if (ta !== tb) {
                const merged = this.merge(ta, tb);
                if (merged >= 0) touched.add(merged);
            }
        }
        return touched;
    }

    private hasFrame(track: Track, frame: number): boolean {
        for (let o = 0; o < track.obs.length; o += 2) if (track.obs[o] === frame) return true;
        return false;
    }

    private extend(id: number, frame: number, feature: number): boolean {
        const track = this.tracks[id];
        if (this.hasFrame(track, frame)) return false;
        track.obs.push(frame, feature);
        this.frames[frame].track[feature] = id;
        return true;
    }

    private merge(a: number, b: number): number {
        const [keep, drop] =
            this.tracks[a].obs.length >= this.tracks[b].obs.length ? [a, b] : [b, a];
        const target = this.tracks[keep];
        const source = this.tracks[drop];
        for (let o = 0; o < source.obs.length; o += 2) {
            if (this.hasFrame(target, source.obs[o])) return -1;
        }
        for (let o = 0; o < source.obs.length; o += 2) {
            target.obs.push(source.obs[o], source.obs[o + 1]);
            this.frames[source.obs[o]].track[source.obs[o + 1]] = keep;
        }
        source.obs = [];
        source.alive = false;
        source.point = null;
        return keep;
    }

    observationsOf(track: Track): Observation[] {
        const out: Observation[] = [];
        for (let o = 0; o < track.obs.length; o += 2) {
            const frame = this.frames[track.obs[o]];
            const f = track.obs[o + 1];
            out.push({
                k: frame.intrinsics,
                pose: frame.pose,
                u: frame.features.x[f],
                v: frame.features.y[f],
            });
        }
        return out;
    }

    /** (Re)triangulates tracks, dropping the worst observation once if it is an outlier. */
    triangulate(ids: Iterable<number>, maxError = MAX_ERROR): void {
        for (const id of ids) {
            const track = this.tracks[id];
            if (!track.alive) continue;
            let result = triangulate(this.observationsOf(track));
            if (result && result.worst > maxError && track.obs.length >= 6) {
                const observations = this.observationsOf(track);
                let worstIndex = 0;
                let worstError = 0;
                observations.forEach((o, index) => {
                    const error = result ? reprojectionError(result.point, o) : 0;
                    if (error > worstError) {
                        worstError = error;
                        worstIndex = index;
                    }
                });
                this.detach(id, worstIndex);
                result = triangulate(this.observationsOf(track));
            }
            if (!result || result.worst > maxError || result.angle < MIN_ANGLE) {
                track.point = null;
                continue;
            }
            if (!track.point) track.color = this.colorOf(track);
            track.point = new Float64Array(result.point);
            track.error = result.rms;
        }
    }

    private detach(id: number, observation: number): void {
        const track = this.tracks[id];
        const frame = track.obs[observation * 2];
        const feature = track.obs[observation * 2 + 1];
        this.frames[frame].track[feature] = -1;
        track.obs.splice(observation * 2, 2);
    }

    private colorOf(track: Track): [number, number, number] {
        const frame = this.frames[track.obs[0]];
        const f = track.obs[1];
        const sx = frame.thumbWidth / frame.width;
        const sy = frame.thumbHeight / frame.height;
        const x = Math.min(frame.thumbWidth - 1, Math.max(0, Math.round(frame.features.x[f] * sx)));
        const y = Math.min(
            frame.thumbHeight - 1,
            Math.max(0, Math.round(frame.features.y[f] * sy)),
        );
        const i = (y * frame.thumbWidth + x) * 3;
        return [frame.rgb[i], frame.rgb[i + 1], frame.rgb[i + 2]];
    }

    /** Valid 3-D points (with ≥ minObservations views). */
    points(minObservations = 2): { ids: number[]; positions: Float32Array; colors: Uint8Array } {
        const ids: number[] = [];
        for (let id = 0; id < this.tracks.length; id++) {
            const track = this.tracks[id];
            if (track.alive && track.point && track.obs.length / 2 >= minObservations) ids.push(id);
        }
        const positions = new Float32Array(ids.length * 3);
        const colors = new Uint8Array(ids.length * 3);
        ids.forEach((id, i) => {
            const track = this.tracks[id];
            positions.set(track.point as Float64Array, i * 3);
            colors.set(track.color, i * 3);
        });
        return { ids, positions, colors };
    }

    /** Median depth of the triangulated points seen by a frame (NaN when it sees none). */
    medianDepth(index: number): number {
        const frame = this.frames[index];
        const depths: number[] = [];
        for (let f = 0; f < frame.track.length; f++) {
            const id = frame.track[f];
            if (id < 0) continue;
            const point = this.tracks[id].point;
            if (!point) continue;
            const z = project(frame.intrinsics, frame.pose, point)[2];
            if (z > 0) depths.push(z);
        }
        if (depths.length < 8) return NaN;
        depths.sort((a, b) => a - b);
        return depths[depths.length >> 1];
    }

    /**
     * Bundle problem over the best-conditioned tracks: at least three views, then the ones
     * with the most views, capped so the Schur complement stays cheap on a phone.
     */
    bundleProblem(maxPoints: number): { problem: BundleProblem; trackIds: number[] } {
        const candidates = this.points(3).ids;
        candidates.sort((a, b) => this.tracks[b].obs.length - this.tracks[a].obs.length);
        const trackIds = candidates.slice(0, maxPoints);
        const points = new Float64Array(trackIds.length * 3);
        const camera: number[] = [];
        const point: number[] = [];
        const u: number[] = [];
        const v: number[] = [];
        trackIds.forEach((id, p) => {
            const track = this.tracks[id];
            points.set(track.point as Float64Array, p * 3);
            for (let o = 0; o < track.obs.length; o += 2) {
                const frame = this.frames[track.obs[o]];
                camera.push(track.obs[o]);
                point.push(p);
                u.push(frame.features.x[track.obs[o + 1]]);
                v.push(frame.features.y[track.obs[o + 1]]);
            }
        });
        return {
            trackIds,
            problem: {
                intrinsics: this.frames.map((frame) => frame.intrinsics),
                poses: this.frames.map((frame) => frame.pose),
                priors: this.frames.map((frame) => frame.prior),
                points,
                camera: Int32Array.from(camera),
                point: Int32Array.from(point),
                u: Float64Array.from(u),
                v: Float64Array.from(v),
            },
        };
    }

    /** Removes observations whose reprojection error exceeds the limit; returns how many. */
    pruneObservations(limit: number): number {
        let removed = 0;
        for (let id = 0; id < this.tracks.length; id++) {
            const track = this.tracks[id];
            if (!track.alive || !track.point) continue;
            const observations = this.observationsOf(track);
            for (let o = observations.length - 1; o >= 0; o--) {
                if (reprojectionError(track.point, observations[o]) > limit) {
                    this.detach(id, o);
                    removed++;
                }
            }
            if (track.obs.length < 4) track.point = null;
        }
        return removed;
    }

    /** Reprojection errors of every observation of every valid track (px). */
    reprojectionErrors(): Float64Array {
        const errors: number[] = [];
        for (const track of this.tracks) {
            if (!track.alive || !track.point) continue;
            for (const observation of this.observationsOf(track)) {
                errors.push(reprojectionError(track.point, observation));
            }
        }
        return Float64Array.from(errors);
    }

    /** Distance from a pair's centres relative to the scene depth (baseline ratio). */
    baselineRatio(i: number, j: number): number {
        const a = this.frames[i];
        const b = this.frames[j];
        const distance = norm3(sub3(cameraCenter(a.pose), cameraCenter(b.pose)));
        return distance / Math.max(0.05, Math.min(a.depthHint, b.depthHint));
    }
}

export function pairKey(i: number, j: number): string {
    return i < j ? `${i}:${j}` : `${j}:${i}`;
}
