import { cameraCenter } from '../../src/app/vision/geometry/camera';
import { GpuContext } from '../../src/app/vision/gpu/gpu-device';
import { QUALITY, Quality } from '../../src/app/core/models/quality';
import { reconstruct } from '../../src/app/vision/pipeline/reconstructor';
import {
    BOX_CENTER,
    BOX_HALF,
    BOX_SIZE,
    boxDistance,
    signedDistance,
    surfaceColor,
    surfaceDistance,
    surfaceSamples,
} from '../synthetic-scene';
import { Capture } from './capture';
import { check, log, percentile } from './report';
import { alignment } from './trajectory';

export interface Limits {
    /** Required drift reduction (mean centre error after / before), null = not checked. */
    drift: number | null;
    reprojection: number | null;
    dense: [number, number];
    mesh: [number, number];
    completeness: number;
    box: number;
    color: number;
}

export const LIMITS: Record<Quality, Limits> = {
    high: {
        drift: 1 / 3,
        reprojection: 0.8,
        dense: [1.5, 5],
        mesh: [1.5, 4],
        completeness: 0.85,
        box: 3,
        color: 14,
    },
    medium: {
        drift: 1 / 3,
        reprojection: 0.8,
        dense: [1.5, 5],
        mesh: [1.5, 4],
        completeness: 0.85,
        box: 3,
        color: 14,
    },
    low: {
        drift: 1 / 2,
        reprojection: 1,
        dense: [3, 8],
        mesh: [3, 8],
        completeness: 0.8,
        box: 6,
        color: 20,
    },
};

export interface Score {
    meshMedian: number;
    /** Median mesh error (mm) once the mesh is rigidly fitted to the truth: shape alone. */
    shape: number;
    /** p90 colour step across texture seams (0–255), NaN without an atlas. */
    seams: number;
    box: number;
    seconds: number;
    focal: number;
}

async function decodeJpeg(jpeg: Uint8Array): Promise<ImageData> {
    const bitmap = await createImageBitmap(new Blob([jpeg as Uint8Array<ArrayBuffer>]));
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height);
}

/** Reconstructs a capture at one quality and scores everything against the truth. */
export async function score(
    gpu: GpuContext,
    { session, truth, priors }: Capture,
    quality: Quality,
    scenario: string,
    limits: Limits,
    random: () => number,
): Promise<Score> {
    const label = `${scenario} · ${QUALITY[quality].label}`;
    const started = performance.now();
    const stages: string[] = [];
    const result = await reconstruct(
        session,
        gpu,
        {
            stage: (event) => {
                if (event.status === 'done') stages.push(event.stage);
            },
            cameras: () => undefined,
            sparse: () => undefined,
            dense: () => undefined,
            mesh: () => undefined,
        },
        quality,
    ).catch((error: unknown) => {
        const metrics = session.map.frames.slice(0, 3).map((f) => f.intrinsics.fx.toFixed(1));
        check(
            `${label}: reconstruction runs`,
            false,
            `${error instanceof Error ? error.message : error} after ${stages.join(' → ')}; fx ${metrics.join(', ')}`,
        );
        return null;
    });
    if (!result) {
        return {
            meshMedian: Infinity,
            shape: Infinity,
            seams: NaN,
            box: Infinity,
            seconds: 0,
            focal: NaN,
        };
    }
    const seconds = (performance.now() - started) / 1000;
    const m = result.metrics;
    log.push(
        `${label}: ${seconds.toFixed(1)} s — ${m.timings.map((t) => `${t.name} ${(t.ms / 1000).toFixed(1)}`).join(', ')}; ${m.keyframesUsed} keyframes, stereo ${m.stereoSize.join('×')}, focal ${(m.focalCorrection * 100).toFixed(2)} %, breaks ${m.tracking.breaks}`,
    );

    const priorError = truth.map((pose, i) =>
        Math.hypot(...pose.center.map((v, k) => v - priors[i].center[k])),
    );
    const afterError = truth.map((pose, i) => {
        const c = cameraCenter(session.map.frames[i].pose);
        return Math.hypot(c[0] - pose.center[0], c[1] - pose.center[1], c[2] - pose.center[2]);
    });
    const meanBefore = priorError.reduce((a, b) => a + b, 0) / priorError.length;
    const meanAfter = afterError.reduce((a, b) => a + b, 0) / afterError.length;
    if (limits.drift !== null) {
        check(
            `${label}: bundle adjustment removes the odometry drift`,
            meanAfter < meanBefore * limits.drift && meanAfter < 0.006,
            `camera centre error ${(meanBefore * 1000).toFixed(1)} → ${(meanAfter * 1000).toFixed(2)} mm, ${m.sparsePoints} sparse points`,
        );
    }
    if (limits.reprojection !== null) {
        check(
            `${label}: reprojection error is sub-pixel`,
            m.reprojection.after.median < limits.reprojection,
            `${m.reprojection.before.median.toFixed(2)} → ${m.reprojection.after.median.toFixed(2)} px (median), mean ${m.reprojection.after.mean.toFixed(2)} px`,
        );
    }

    const centres = session.map.frames.map((frame) => cameraCenter(frame.pose));
    const align = alignment(
        centres,
        truth.map((pose) => pose.center),
    );
    // Pose quality that matters for the model: centres after the rigid alignment, and the scale.
    const alignedCentres = centres.map(align);
    const rigidError =
        alignedCentres.reduce(
            (sum, c, i) =>
                sum +
                Math.hypot(
                    c[0] - truth[i].center[0],
                    c[1] - truth[i].center[1],
                    c[2] - truth[i].center[2],
                ),
            0,
        ) / alignedCentres.length;
    const spread = (points: number[][]) => {
        const mean = [0, 1, 2].map((k) => points.reduce((sum, p) => sum + p[k], 0) / points.length);
        return Math.sqrt(
            points.reduce(
                (sum, p) =>
                    sum + (p[0] - mean[0]) ** 2 + (p[1] - mean[1]) ** 2 + (p[2] - mean[2]) ** 2,
                0,
            ) / points.length,
        );
    };
    const scale = spread(centres) / spread(truth.map((pose) => pose.center));
    // Relative rotation between keyframes a few apart (what multi-view stereo depends on).
    const rotationErrors: number[] = [];
    const frames = session.map.frames;
    for (let i = 0; i + 3 < frames.length; i++) {
        const j = i + 3;
        const estimated = relative(frames[i].pose.R, frames[j].pose.R);
        const real = relative(truth[i].R, truth[j].R);
        let trace = 0;
        for (let r = 0; r < 3; r++)
            for (let c = 0; c < 3; c++) trace += estimated[r * 3 + c] * real[r * 3 + c];
        rotationErrors.push(
            (Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2))) * 180) / Math.PI,
        );
    }
    log.push(
        `${label}: aligned camera error ${(rigidError * 1000).toFixed(2)} mm, scale ${((scale - 1) * 100).toFixed(3)} %, relative rotation error median ${(percentile(rotationErrors, 0.5) * 1000).toFixed(1)} / p90 ${(percentile(rotationErrors, 0.9) * 1000).toFixed(1)} mdeg`,
    );
    const densePositions = result.dense.positions;
    const denseErrors: number[] = [];
    for (let i = 0; i < densePositions.length; i += 3 * 7) {
        denseErrors.push(
            surfaceDistance(
                align([densePositions[i], densePositions[i + 1], densePositions[i + 2]]),
            ),
        );
    }
    check(
        `${label}: dense cloud lies on the surface`,
        percentile(denseErrors, 0.5) * 1000 < limits.dense[0] &&
            percentile(denseErrors, 0.9) * 1000 < limits.dense[1],
        `${m.densePoints} points, error median ${(percentile(denseErrors, 0.5) * 1000).toFixed(2)} mm, p90 ${(percentile(denseErrors, 0.9) * 1000).toFixed(2)} mm, voxel ${(m.voxel * 1000).toFixed(2)} mm`,
    );

    // Completeness: visible surface samples with a mesh vertex nearby.
    const model = result.model;
    const cell = 0.004;
    const buckets = new Map<string, number[]>();
    const aligned: number[][] = [];
    for (let i = 0; i < model.positions.length; i += 3) {
        const p = align([model.positions[i], model.positions[i + 1], model.positions[i + 2]]);
        aligned.push(p);
        const key = `${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)},${Math.floor(p[2] / cell)}`;
        const list = buckets.get(key) ?? [];
        list.push(aligned.length - 1);
        buckets.set(key, list);
    }
    const near = (p: number[], radius: number) => {
        const [x, y, z] = p.map((v) => Math.floor(v / cell));
        for (let dx = -1; dx <= 1; dx++)
            for (let dy = -1; dy <= 1; dy++)
                for (let dz = -1; dz <= 1; dz++) {
                    for (const i of buckets.get(`${x + dx},${y + dy},${z + dz}`) ?? []) {
                        const q = aligned[i];
                        if (Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < radius) return true;
                    }
                }
        return false;
    };
    const samples = surfaceSamples(4000, random);
    // Vertices are about one voxel apart, so the radius cannot be smaller than that.
    const reach = Math.max(0.003, m.voxel * 1.2);
    const complete = samples.filter((p) => near(p, reach)).length / samples.length;
    const meshErrors: number[] = [];
    for (let i = 0; i < aligned.length; i += 5) meshErrors.push(surfaceDistance(aligned[i]));
    const meshMedian = percentile(meshErrors, 0.5) * 1000;
    const shape = shapeError(aligned);
    check(
        `${label}: mesh is accurate and complete`,
        meshMedian < limits.mesh[0] &&
            percentile(meshErrors, 0.9) * 1000 < limits.mesh[1] &&
            complete > limits.completeness,
        `${m.meshTriangles} triangles, error median ${meshMedian.toFixed(2)} mm, p90 ${(percentile(meshErrors, 0.9) * 1000).toFixed(2)} mm, completeness ${(complete * 100).toFixed(1)} % within ${(reach * 1000).toFixed(1)} mm; shape alone (mesh fitted to the truth) median ${shape.median.toFixed(3)} / p90 ${shape.p90.toFixed(3)} mm`,
    );

    // Metric size of the box, measured like a caliper: width and depth from the side faces at
    // mid-height (away from the fillet the fusion leaves where the box meets the table), height
    // from the top face.
    const tolerance = Math.max(0.0015, m.voxel * 0.6);
    const sides = aligned.filter(
        (p) => p[1] > 0.03 && p[1] < 0.09 && Math.abs(boxDistance(p)) < tolerance,
    );
    const top = aligned.filter(
        (p) => Math.abs(p[1] - BOX_SIZE[1]) < 2 * tolerance && Math.abs(boxDistance(p)) < tolerance,
    );
    // Distance between the medians of the two faces normal to axis k (vertices are assigned to
    // the face whose normalised box coordinate is largest, so other faces do not leak in).
    const span = (k: number) => {
        const other = 2 - k;
        const onFace = sides.filter(
            (p) =>
                Math.abs(p[k] - BOX_CENTER[k]) / BOX_HALF[k] >
                Math.abs(p[other] - BOX_CENTER[other]) / BOX_HALF[other],
        );
        const low = onFace.filter((p) => p[k] < BOX_CENTER[k]).map((p) => p[k]);
        const high = onFace.filter((p) => p[k] > BOX_CENTER[k]).map((p) => p[k]);
        return percentile(high, 0.5) - percentile(low, 0.5);
    };
    const size = [
        span(0),
        percentile(
            top.map((p) => p[1]),
            0.5,
        ),
        span(2),
    ];
    const sizeError = size.map((v, k) => Math.abs(v - BOX_SIZE[k]) * 1000);
    const box = Math.max(...sizeError);
    check(
        `${label}: box dimensions are metric`,
        box < limits.box,
        `measured ${size.map((v) => (v * 1000).toFixed(1)).join(' × ')} mm vs ${BOX_SIZE.map((v) => (v * 1000).toFixed(0)).join(' × ')} mm`,
    );

    // Texture: colour at each vertex (atlas texel or vertex colour) vs the true surface colour.
    const atlas = model.atlasJpeg ? await decodeJpeg(model.atlasJpeg) : null;
    // The absolute brightness of a scan is unknowable (exposure, vignetting): colours are
    // compared after one least-squares gain per channel.
    const pairs: { rgb: number[]; expected: [number, number, number] }[] = [];
    for (let v = 0; v < aligned.length; v += 3) {
        const p = aligned[v];
        if (surfaceDistance(p) > 0.002) continue;
        let rgb: number[];
        if (atlas && model.uvs) {
            const x = Math.min(atlas.width - 1, Math.floor(model.uvs[v * 2] * atlas.width));
            const y = Math.min(atlas.height - 1, Math.floor(model.uvs[v * 2 + 1] * atlas.height));
            const o = (y * atlas.width + x) * 4;
            rgb = [atlas.data[o], atlas.data[o + 1], atlas.data[o + 2]];
        } else {
            const colors = model.colors as Uint8Array;
            rgb = [colors[v * 4], colors[v * 4 + 1], colors[v * 4 + 2]];
        }
        pairs.push({ rgb, expected: surfaceColor(p) });
    }
    const gains = [0, 1, 2].map((c) => {
        let cross = 0;
        let own = 0;
        for (const { rgb, expected } of pairs) {
            cross += rgb[c] * expected[c];
            own += rgb[c] * rgb[c];
        }
        return own > 0 ? cross / own : 1;
    });
    const colorErrors = pairs.map(
        ({ rgb, expected }) =>
            [0, 1, 2].reduce((sum, c) => sum + Math.abs(rgb[c] * gains[c] - expected[c]), 0) / 3,
    );
    // Seams: the atlas duplicates the vertices on chart borders; their two colours should agree.
    const chartSides = new Map<string, number[][]>();
    for (let v = 0; v < aligned.length && atlas && model.uvs; v++) {
        const key = `${model.positions[v * 3]},${model.positions[v * 3 + 1]},${model.positions[v * 3 + 2]}`;
        const x = Math.min(atlas.width - 1, Math.floor(model.uvs[v * 2] * atlas.width));
        const y = Math.min(atlas.height - 1, Math.floor(model.uvs[v * 2 + 1] * atlas.height));
        const o = (y * atlas.width + x) * 4;
        const list = chartSides.get(key) ?? [];
        list.push([atlas.data[o], atlas.data[o + 1], atlas.data[o + 2]]);
        chartSides.set(key, list);
    }
    const steps: number[] = [];
    for (const list of chartSides.values()) {
        for (let i = 1; i < list.length; i++) {
            steps.push(
                (Math.abs(list[i][0] - list[0][0]) +
                    Math.abs(list[i][1] - list[0][1]) +
                    Math.abs(list[i][2] - list[0][2])) /
                    3,
            );
        }
    }
    const seams = steps.length
        ? `, seam step median ${percentile(steps, 0.5).toFixed(1)} / p90 ${percentile(steps, 0.9).toFixed(1)} over ${steps.length} seam vertices`
        : '';
    check(
        `${label}: texture matches the true surface colour`,
        percentile(colorErrors, 0.5) < limits.color,
        `${m.atlas ? `${m.atlas.join('×')} atlas, ${m.charts} charts` : 'vertex colours'}, ${m.untextured} untextured faces, colour error median ${percentile(colorErrors, 0.5).toFixed(1)} / p90 ${percentile(colorErrors, 0.9).toFixed(1)} (0–255, after a global gain of ${gains.map((g) => g.toFixed(2)).join('/')})${seams}`,
    );
    return {
        meshMedian,
        shape: shape.median,
        seams: steps.length ? percentile(steps, 0.9) : NaN,
        box,
        seconds,
        focal: m.focalCorrection,
    };
}

/** Rᵢ·Rⱼᵀ for row-major world → camera rotations: camera j → camera i. */
function relative(Ri: ArrayLike<number>, Rj: ArrayLike<number>): number[] {
    const out: number[] = [];
    for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++)
            out.push(
                Ri[r * 3] * Rj[c * 3] +
                    Ri[r * 3 + 1] * Rj[c * 3 + 1] +
                    Ri[r * 3 + 2] * Rj[c * 3 + 2],
            );
    return out;
}

/**
 * Mesh error once the mesh itself is rigidly fitted to the true surface (Gauss–Newton on the
 * signed distance), separating the shape from the placement given by the camera alignment.
 */
function shapeError(vertices: number[][]): { median: number; p90: number } {
    const sample = vertices.filter((_, i) => i % 11 === 0);
    let R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    let t = [0, 0, 0];
    const apply = (p: number[]) =>
        [0, 1, 2].map((r) => R[r * 3] * p[0] + R[r * 3 + 1] * p[1] + R[r * 3 + 2] * p[2] + t[r]);
    const h = 1e-5;
    for (let iteration = 0; iteration < 8; iteration++) {
        const A = new Array(36).fill(0);
        const b = new Array(6).fill(0);
        for (const v of sample) {
            const p = apply(v);
            const r = signedDistance(p);
            if (Math.abs(r) > 0.002) continue;
            const g = [0, 1, 2].map((k) => {
                const q = [...p];
                q[k] += h;
                return (signedDistance(q) - r) / h;
            });
            // d p / d ω = −[p]×, d p / d t = I.
            const J = [
                g[2] * p[1] - g[1] * p[2],
                g[0] * p[2] - g[2] * p[0],
                g[1] * p[0] - g[0] * p[1],
                g[0],
                g[1],
                g[2],
            ];
            for (let i = 0; i < 6; i++) {
                b[i] -= J[i] * r;
                for (let j = 0; j < 6; j++) A[i * 6 + j] += J[i] * J[j];
            }
        }
        const x = solve(A, b, 6);
        const [wx, wy, wz] = x;
        const dR = [1, -wz, wy, wz, 1, -wx, -wy, wx, 1];
        R = [0, 1, 2].flatMap((r) =>
            [0, 1, 2].map(
                (c) => dR[r * 3] * R[c] + dR[r * 3 + 1] * R[3 + c] + dR[r * 3 + 2] * R[6 + c],
            ),
        );
        t = [0, 1, 2].map(
            (r) => dR[r * 3] * t[0] + dR[r * 3 + 1] * t[1] + dR[r * 3 + 2] * t[2] + x[3 + r],
        );
    }
    const errors = vertices
        .filter((_, i) => i % 5 === 0)
        .map((v) => Math.abs(signedDistance(apply(v))) * 1000);
    return { median: percentile(errors, 0.5), p90: percentile(errors, 0.9) };
}

function solve(A: number[], b: number[], n: number): number[] {
    const M = A.map((v) => v);
    const x = [...b];
    for (let c = 0; c < n; c++) {
        let pivot = c;
        for (let r = c + 1; r < n; r++)
            if (Math.abs(M[r * n + c]) > Math.abs(M[pivot * n + c])) pivot = r;
        for (let k = 0; k < n; k++)
            [M[c * n + k], M[pivot * n + k]] = [M[pivot * n + k], M[c * n + k]];
        [x[c], x[pivot]] = [x[pivot], x[c]];
        for (let r = 0; r < n; r++) {
            if (r === c) continue;
            const f = M[r * n + c] / M[c * n + c];
            for (let k = 0; k < n; k++) M[r * n + k] -= f * M[c * n + k];
            x[r] -= f * x[c];
        }
    }
    return x.map((v, i) => v / M[i * n + i]);
}
