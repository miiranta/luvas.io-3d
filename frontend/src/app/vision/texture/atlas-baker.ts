import { Intrinsics, Pose } from '../geometry/camera';
import { solveCholesky } from '../math/dense';
import { FaceCandidates } from './face-visibility';
import { connectedCharts } from './view-labeling';

export interface RgbaImage {
    width: number;
    height: number;
    data: Uint8Array | Uint8ClampedArray;
}

export interface TextureView {
    /** Intrinsics at the resolution of the full texture image. */
    intrinsics: Intrinsics;
    pose: Pose;
    width: number;
    height: number;
    /** Small RGB copy used for colour statistics. */
    thumb: { rgb: Uint8Array; width: number; height: number; intrinsics: Intrinsics };
}

export interface TexturedMesh {
    positions: Float32Array;
    uvs: Float32Array;
    indices: Uint32Array;
    atlas: { width: number; height: number; pixels: Uint8ClampedArray };
    charts: number;
    /** Faces no camera could texture (painted neutral grey). */
    untextured: number;
}

const PAD = 2;
const GREY = 150;

interface Chart {
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

/**
 * Builds the texture atlas from the per-face view labels:
 *  1. faces whose label does not fully project into the image fall back to their next view;
 *  2. connected same-label faces form charts whose UVs are simply their projection;
 *  3. per-view RGB gains equalise exposure (log-domain least squares over shared faces);
 *  4. seam levelling: at vertices shared by several charts the colours are pulled to their
 *     mean, and the offset is spread over each chart as a harmonic (Laplace) field;
 *  5. charts are shelf-packed and baked from the source images.
 */
export async function bakeAtlas(
    positions: Float32Array,
    indices: Uint32Array,
    labels: Int16Array,
    candidates: FaceCandidates,
    adjacency: Int32Array,
    views: TextureView[],
    loadImage: (view: number) => Promise<RgbaImage>,
    maxSize: number,
    progress: (fraction: number) => void,
): Promise<TexturedMesh> {
    const faces = indices.length / 3;
    validateLabels(positions, indices, labels, candidates, views);
    const gains = estimateGains(positions, indices, candidates, views);
    const { chartOf, count } = connectedCharts(labels, adjacency);
    const charts = buildCharts(positions, indices, labels, chartOf, count, views);
    levelSeams(positions, charts, views, gains);
    const atlas = packCharts(charts, maxSize);

    const pixels = new Uint8ClampedArray(atlas.width * atlas.height * 4);
    for (let i = 0; i < pixels.length; i += 4) {
        pixels[i] = pixels[i + 1] = pixels[i + 2] = GREY;
        pixels[i + 3] = 255;
    }
    const byView = new Map<number, Chart[]>();
    for (const chart of charts) {
        if (chart.view < 0) continue;
        const list = byView.get(chart.view) ?? [];
        list.push(chart);
        byView.set(chart.view, list);
    }
    let done = 0;
    for (const [view, list] of byView) {
        const image = await loadImage(view);
        for (const chart of list) bakeChart(chart, image, gains, view, atlas, pixels);
        progress(++done / byView.size);
    }

    // Output mesh: vertices are duplicated per chart because UVs differ across seams.
    let vertexCount = 0;
    for (const chart of charts) vertexCount += chart.vertices.length;
    const outPositions = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const outIndices = new Uint32Array(faces * 3);
    let base = 0;
    let faceCursor = 0;
    let untextured = 0;
    for (const chart of charts) {
        chart.vertices.forEach((v, i) => {
            outPositions.set(positions.subarray(v * 3, v * 3 + 3), (base + i) * 3);
            if (chart.view < 0) {
                uvs[(base + i) * 2] = (atlas.greyX + 0.5) / atlas.width;
                uvs[(base + i) * 2 + 1] = (atlas.greyY + 0.5) / atlas.height;
            } else {
                uvs[(base + i) * 2] =
                    (chart.ax + (chart.uv[i * 2] - chart.x0 + 0.5) * atlas.scale) / atlas.width;
                uvs[(base + i) * 2 + 1] =
                    (chart.ay + (chart.uv[i * 2 + 1] - chart.y0 + 0.5) * atlas.scale) /
                    atlas.height;
            }
        });
        for (let corner = 0; corner < chart.corners.length; corner++) {
            outIndices[faceCursor * 3 + corner] = base + chart.corners[corner];
        }
        faceCursor += chart.faces.length;
        if (chart.view < 0) untextured += chart.faces.length;
        base += chart.vertices.length;
    }
    return {
        positions: outPositions,
        uvs,
        indices: outIndices,
        atlas: { width: atlas.width, height: atlas.height, pixels },
        charts: charts.length,
        untextured,
    };
}

function projectInto(
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

function validateLabels(
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

function sampleThumb(
    view: TextureView,
    x: number,
    y: number,
    z: number,
    out: Float64Array,
): boolean {
    const t = view.thumb;
    const k = t.intrinsics;
    const { R } = view.pose;
    const tr = view.pose.t;
    const cx = R[0] * x + R[1] * y + R[2] * z + tr[0];
    const cy = R[3] * x + R[4] * y + R[5] * z + tr[1];
    const cz = R[6] * x + R[7] * y + R[8] * z + tr[2];
    if (cz <= 0.01) return false;
    const u = (k.fx * cx) / cz + k.cx;
    const v = (k.fy * cy) / cz + k.cy;
    if (u < 0 || v < 0 || u > t.width - 1.001 || v > t.height - 1.001) return false;
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const wx = u - x0;
    const wy = v - y0;
    for (let c = 0; c < 3; c++) {
        const i = (y0 * t.width + x0) * 3 + c;
        const top = t.rgb[i] * (1 - wx) + t.rgb[i + 3] * wx;
        const bottom = t.rgb[i + t.width * 3] * (1 - wx) + t.rgb[i + t.width * 3 + 3] * wx;
        out[c] = top * (1 - wy) + bottom * wy;
    }
    return true;
}

/** Per-view RGB gains from faces seen by several views (robust log-domain least squares). */
function estimateGains(
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

function buildCharts(
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

/**
 * Local seam levelling (after Waechter et al.): vertices on chart borders are given the mean of
 * the colours all their charts see; the difference becomes a correction that is interpolated
 * harmonically over the chart's interior vertices.
 */
function levelSeams(
    positions: Float32Array,
    charts: Chart[],
    views: TextureView[],
    gains: Float32Array,
): void {
    const vertexCount = positions.length / 3;
    const chartCount = new Uint8Array(vertexCount);
    for (const chart of charts) {
        if (chart.view < 0) continue;
        for (const v of chart.vertices) if (chartCount[v] < 255) chartCount[v]++;
    }
    const sums = new Float32Array(vertexCount * 3);
    const counts = new Uint8Array(vertexCount);
    const color = new Float64Array(3);
    const own = charts.map((chart) => new Float32Array(chart.vertices.length * 3).fill(NaN));
    charts.forEach((chart, c) => {
        if (chart.view < 0) return;
        const view = views[chart.view];
        chart.vertices.forEach((v, i) => {
            if (chartCount[v] < 2) return;
            if (
                !sampleThumb(
                    view,
                    positions[v * 3],
                    positions[v * 3 + 1],
                    positions[v * 3 + 2],
                    color,
                )
            )
                return;
            for (let ch = 0; ch < 3; ch++) {
                const value = color[ch] * gains[chart.view * 3 + ch];
                own[c][i * 3 + ch] = value;
                sums[v * 3 + ch] += value;
            }
            counts[v]++;
        });
    });
    charts.forEach((chart, c) => {
        if (chart.view < 0) return;
        const n = chart.vertices.length;
        const fixed = new Uint8Array(n);
        const correction = chart.correction;
        chart.vertices.forEach((v, i) => {
            if (counts[v] < 2 || Number.isNaN(own[c][i * 3])) return;
            fixed[i] = 1;
            for (let ch = 0; ch < 3; ch++) {
                correction[i * 3 + ch] = sums[v * 3 + ch] / counts[v] - own[c][i * 3 + ch];
            }
        });
        if (!fixed.some(Boolean)) return;
        // Chart-local vertex graph from the faces' edges.
        const neighbours: number[][] = Array.from({ length: n }, () => []);
        const corners = chart.corners;
        for (let f = 0; f < corners.length; f += 3) {
            for (let k = 0; k < 3; k++) {
                const a = corners[f + k];
                const b = corners[f + ((k + 1) % 3)];
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
        }
        const next = new Float32Array(correction);
        for (let iteration = 0; iteration < 60; iteration++) {
            for (let i = 0; i < n; i++) {
                if (fixed[i] || neighbours[i].length === 0) continue;
                let r = 0;
                let g = 0;
                let b = 0;
                for (const j of neighbours[i]) {
                    r += correction[j * 3];
                    g += correction[j * 3 + 1];
                    b += correction[j * 3 + 2];
                }
                const m = neighbours[i].length;
                next[i * 3] = r / m;
                next[i * 3 + 1] = g / m;
                next[i * 3 + 2] = b / m;
            }
            correction.set(next);
        }
    });
}

interface AtlasLayout {
    width: number;
    height: number;
    scale: number;
    greyX: number;
    greyY: number;
}

/** Shelf packing (tallest first); shrinks every chart uniformly when the atlas would overflow. */
function packCharts(charts: Chart[], maxSize: number): AtlasLayout {
    const textured = charts.filter((chart) => chart.view >= 0);
    const order = [...textured].sort((a, b) => b.h - a.h);
    let area = 16;
    for (const chart of textured) area += chart.w * chart.h;
    let widest = 0;
    for (const chart of textured) widest = Math.max(widest, chart.w);
    let scale = Math.min(1, maxSize / Math.sqrt(area * 1.12), (maxSize - 8) / widest);
    for (let attempt = 0; attempt < 12; attempt++) {
        const width = Math.min(
            maxSize,
            Math.max(64, Math.ceil(Math.sqrt(area * 1.12) * scale), Math.ceil(widest * scale) + 5),
        );
        let x = 4;
        let y = 0;
        let shelf = 4;
        let fits = true;
        for (const chart of order) {
            const w = Math.ceil(chart.w * scale) + 1;
            const h = Math.ceil(chart.h * scale) + 1;
            if (x + w > width) {
                x = 0;
                y += shelf;
                shelf = 0;
            }
            chart.ax = x;
            chart.ay = y;
            x += w;
            shelf = Math.max(shelf, h);
            if (y + shelf > maxSize) {
                fits = false;
                break;
            }
        }
        if (fits) {
            return { width, height: Math.max(4, y + shelf), scale, greyX: 1, greyY: 1 };
        }
        scale *= 0.9;
    }
    throw new Error('texture atlas does not fit');
}

function bakeChart(
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
