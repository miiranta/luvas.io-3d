import { FaceCandidates } from './face-visibility';
import { connectedCharts } from './view-labeling';
import { bakeChart } from './atlas/chart-baking';
import { packCharts } from './atlas/chart-packing';
import { Chart, buildCharts, validateLabels } from './atlas/charts';
import { estimateGains } from './atlas/color-gains';
import { levelSeams } from './atlas/seam-levelling';
import { RgbaImage, TextureView, TexturedMesh } from './texture-types';

export type { RgbaImage, TextureView, TexturedMesh } from './texture-types';

const GREY = 150;

/**
 * Builds the texture atlas from the per-face view labels:
 *  1. faces whose label does not fully project into the image fall back to their next view;
 *  2. connected same-label faces form charts whose UVs are simply their projection;
 *  3. per-view RGB gains equalise exposure (log-domain least squares over shared faces);
 *  4. seam levelling (optional): at vertices shared by several charts the colours are pulled to their
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
    seams = true,
): Promise<TexturedMesh> {
    const faces = indices.length / 3;
    validateLabels(positions, indices, labels, candidates, views);
    const gains = estimateGains(positions, indices, candidates, views);
    const { chartOf, count } = connectedCharts(labels, adjacency);
    const charts = buildCharts(positions, indices, labels, chartOf, count, views);
    if (seams) levelSeams(positions, charts, views, gains);
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
