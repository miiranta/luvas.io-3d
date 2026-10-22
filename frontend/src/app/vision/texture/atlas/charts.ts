import { Intrinsics, Pose } from '../../geometry/camera';
import { FaceCandidates } from '../face-visibility';
import { TextureView } from '../texture-types';

/** Border (px) kept around every chart so bilinear filtering never reads a neighbour. */
const PAD = 2;

/** A connected set of faces textured from one view; its UVs are the faces' projection. */
export interface Chart {
    view: number;
    faces: number[];
    /** Global vertex id per local vertex. */
    vertices: number[];
    /** Local vertex index of each face corner (3 per face, same order as `faces`). */
    corners: Int32Array;
    /** Projected pixel position per local vertex (texture resolution). */
    uv: Float32Array;
    /** Seam-levelling colour offset per local vertex (RGB, 0–255 scale). */
    correction: Float32Array;
    x0: number;
    y0: number;
    w: number;
    h: number;
    /** Placement in the atlas. */
    ax: number;
    ay: number;
}

export function projectInto(
    k: Intrinsics,
    pose: Pose,
    p: Float32Array,
    v: number,
): [number, number, number] {
    const { R, t } = pose;
    const x = R[0] * p[v * 3] + R[1] * p[v * 3 + 1] + R[2] * p[v * 3 + 2] + t[0];
    const y = R[3] * p[v * 3] + R[4] * p[v * 3 + 1] + R[5] * p[v * 3 + 2] + t[1];
    const z = R[6] * p[v * 3] + R[7] * p[v * 3 + 1] + R[8] * p[v * 3 + 2] + t[2];
    return [(k.fx * x) / z + k.cx, (k.fy * y) / z + k.cy, z];
}

function faceFits(
    positions: Float32Array,
    indices: Uint32Array,
    f: number,
    view: TextureView,
): boolean {
    for (let k = 0; k < 3; k++) {
        const [u, v, z] = projectInto(view.intrinsics, view.pose, positions, indices[f * 3 + k]);
        if (z <= 0.01 || u < -1 || v < -1 || u > view.width || v > view.height) return false;
    }
    return true;
}

/** Faces whose view does not hold all three corners fall back to their next candidate. */
export function validateLabels(
    positions: Float32Array,
    indices: Uint32Array,
    labels: Int16Array,
    candidates: FaceCandidates,
    views: TextureView[],
): void {
    for (let f = 0; f < labels.length; f++) {
        if (labels[f] >= 0 && faceFits(positions, indices, f, views[labels[f]])) continue;
        let replacement = -1;
        for (let s = 0; s < candidates.slots; s++) {
            const view = candidates.view[f * candidates.slots + s];
            if (view >= 0 && faceFits(positions, indices, f, views[view])) {
                replacement = view;
                break;
            }
        }
        labels[f] = replacement;
    }
}

export function buildCharts(
    positions: Float32Array,
    indices: Uint32Array,
    labels: Int16Array,
    chartOf: Int32Array,
    count: number,
    views: TextureView[],
): Chart[] {
    const charts: Chart[] = [];
    const facesOf: number[][] = Array.from({ length: count }, () => []);
    for (let f = 0; f < labels.length; f++) facesOf[chartOf[f]].push(f);
    // One shared grey chart for every untextured face.
    const grey: number[] = [];
    for (let c = 0; c < count; c++) {
        const list = facesOf[c];
        const view = labels[list[0]];
        if (view < 0) {
            for (const f of list) grey.push(f);
            continue;
        }
        charts.push(makeChart(positions, indices, list, view, views[view]));
    }
    if (grey.length > 0) {
        const { vertices, corners } = localVertices(indices, grey);
        charts.push({
            view: -1,
            faces: grey,
            vertices,
            corners,
            uv: new Float32Array(vertices.length * 2),
            correction: new Float32Array(vertices.length * 3),
            x0: 0,
            y0: 0,
            w: 0,
            h: 0,
            ax: 0,
            ay: 0,
        });
    }
    return charts;
}

function makeChart(
    positions: Float32Array,
    indices: Uint32Array,
    faces: number[],
    view: number,
    texture: TextureView,
): Chart {
    const { vertices, corners } = localVertices(indices, faces);
    const uv = new Float32Array(vertices.length * 2);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    vertices.forEach((v, i) => {
        const [u, w] = projectInto(texture.intrinsics, texture.pose, positions, v);
        uv[i * 2] = u;
        uv[i * 2 + 1] = w;
        minX = Math.min(minX, u);
        minY = Math.min(minY, w);
        maxX = Math.max(maxX, u);
        maxY = Math.max(maxY, w);
    });
    const x0 = Math.floor(minX) - PAD;
    const y0 = Math.floor(minY) - PAD;
    return {
        view,
        faces,
        vertices,
        corners,
        uv,
        correction: new Float32Array(vertices.length * 3),
        x0,
        y0,
        w: Math.ceil(maxX) + PAD + 1 - x0,
        h: Math.ceil(maxY) + PAD + 1 - y0,
        ax: 0,
        ay: 0,
    };
}

function localVertices(
    indices: Uint32Array,
    faces: number[],
): { vertices: number[]; corners: Int32Array } {
    const vertices: number[] = [];
    const corners = new Int32Array(faces.length * 3);
    const seen = new Map<number, number>();
    faces.forEach((f, i) => {
        for (let k = 0; k < 3; k++) {
            const v = indices[f * 3 + k];
            let local = seen.get(v);
            if (local === undefined) {
                local = vertices.length;
                seen.set(v, local);
                vertices.push(v);
            }
            corners[i * 3 + k] = local;
        }
    });
    return { vertices, corners };
}
