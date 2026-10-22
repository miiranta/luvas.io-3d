import { TextureView } from '../texture-types';
import { sampleThumb } from '../thumbnail';
import { Chart } from './charts';

/**
 * Local seam levelling (after Waechter et al.): vertices on chart borders are given the mean of
 * the colours all their charts see; the difference becomes a correction that is interpolated
 * harmonically over the chart's interior vertices.
 */
export function levelSeams(
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
