import { FaceCandidates } from './face-visibility';
import { labelFacesMrf, labellingEnergy } from './mrf-labeling';
import { absorbSmallCharts, connectedCharts, faceAdjacency, labelFaces } from './view-labeling';

/** A w × h grid of quads split into triangles (2 faces per cell). */
function grid(w: number, h: number): Uint32Array {
    const indices: number[] = [];
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const a = y * (w + 1) + x;
            const b = a + 1;
            const c = a + w + 1;
            const d = c + 1;
            indices.push(a, b, d, a, d, c);
        }
    }
    return Uint32Array.from(indices);
}

function candidates(
    faces: number,
    score: (face: number, view: number) => number,
    views: number,
): FaceCandidates {
    const slots = 4;
    const out: FaceCandidates = {
        slots,
        view: new Int16Array(faces * slots).fill(-1),
        score: new Float32Array(faces * slots),
    };
    for (let f = 0; f < faces; f++) {
        const ranked = Array.from({ length: views }, (_, v) => ({ v, s: score(f, v) }))
            .filter(({ s }) => s > 0)
            .sort((a, b) => b.s - a.s)
            .slice(0, slots);
        ranked.forEach(({ v, s }, i) => {
            out.view[f * slots + i] = v;
            out.score[f * slots + i] = s;
        });
    }
    return out;
}

describe('face adjacency', () => {
    it('links every interior edge both ways', () => {
        const indices = grid(4, 3);
        const adjacency = faceAdjacency(indices, 5 * 4);
        for (let f = 0; f < indices.length / 3; f++) {
            for (let k = 0; k < 3; k++) {
                const g = adjacency[f * 3 + k];
                if (g >= 0) expect([0, 1, 2].some((j) => adjacency[g * 3 + j] === f)).toBe(true);
            }
        }
        // 4 × 3 cells: 24 faces × 3 edge slots, minus the 14 border edges.
        const linked = Array.from(adjacency).filter((g) => g >= 0).length;
        expect(linked).toBe(24 * 3 - 14);
    });
});

describe('view labelling', () => {
    it('grows one chart when a single view is nearly as good everywhere', () => {
        const indices = grid(10, 10);
        const adjacency = faceAdjacency(indices, 11 * 11);
        // View 0 is the best for every face; view 1 is the best for a few scattered faces only
        // by a small margin, which must not fragment the chart.
        const scores = candidates(200, (f, v) => (v === 0 ? 1 : f % 7 === 0 ? 1.1 : 0.5), 2);
        const labels = labelFaces(scores, adjacency);
        absorbSmallCharts(labels, scores, adjacency, 12);
        const charts = connectedCharts(labels, adjacency);
        expect(charts.count).toBeLessThanOrEqual(2);
    });

    it('splits where the best view changes', () => {
        const indices = grid(10, 4);
        const adjacency = faceAdjacency(indices, 11 * 5);
        // Left half seen well by view 0 only, right half by view 1 only.
        const scores = candidates(80, (f, v) => (f % 20 < 10 === (v === 0) ? 1 : 0.1), 2);
        const labels = labelFaces(scores, adjacency);
        const charts = connectedCharts(labels, adjacency);
        expect(charts.count).toBe(2);
        expect(labels[0]).toBe(0);
        expect(labels[19]).toBe(1);
    });
});

describe('MRF view selection', () => {
    it('ignores small score differences instead of fragmenting the texture', () => {
        const indices = grid(12, 12);
        const adjacency = faceAdjacency(indices, 13 * 13);
        // Three views with noisy, nearly equal scores: the best view per face changes all the time.
        let seed = 7;
        const noise = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 0.2;
        const scores = candidates(288, () => 0.9 + noise(), 3);
        const labels = labelFacesMrf(scores, adjacency, 1);
        const greedy = labelFaces(scores, adjacency);
        expect(connectedCharts(labels, adjacency).count).toBeLessThanOrEqual(3);
        const mrf = labellingEnergy(labels, scores, adjacency, 1);
        const charts = labellingEnergy(greedy, scores, adjacency, 1);
        expect(mrf.data + mrf.seams).toBeLessThanOrEqual(charts.data + charts.seams);
    });

    it('still switches view where one is clearly better', () => {
        const indices = grid(10, 4);
        const adjacency = faceAdjacency(indices, 11 * 5);
        const scores = candidates(80, (f, v) => (f % 20 < 10 === (v === 0) ? 1 : 0.1), 2);
        const labels = labelFacesMrf(scores, adjacency, 1);
        expect(connectedCharts(labels, adjacency).count).toBe(2);
        expect(labels[0]).toBe(0);
        expect(labels[19]).toBe(1);
    });

    it('beats chart growing where views overlap in strips', () => {
        const indices = grid(16, 8);
        const adjacency = faceAdjacency(indices, 17 * 9);
        // Four views each best on a vertical strip, all of them seeing everything a bit worse:
        // chart growing seeds charts strip by strip, the MRF can trade strips against seams.
        const scores = candidates(
            256,
            (f, v) => {
                const column = Math.floor((f % 32) / 2);
                return 1 - 0.12 * Math.abs(Math.floor(column / 4) - v) + 0.03 * ((f * 7 + v) % 3);
            },
            4,
        );
        const mrf = labellingEnergy(labelFacesMrf(scores, adjacency, 1), scores, adjacency, 1);
        const greedy = labellingEnergy(labelFaces(scores, adjacency), scores, adjacency, 1);
        expect(mrf.data + mrf.seams).toBeLessThan(greedy.data + greedy.seams);
    });

    it('leaves faces no view sees unlabelled', () => {
        const indices = grid(4, 4);
        const adjacency = faceAdjacency(indices, 5 * 5);
        const scores = candidates(32, (f, v) => (f < 4 ? 0 : v === 0 ? 1 : 0.5), 2);
        const labels = labelFacesMrf(scores, adjacency, 1);
        expect(Array.from(labels.subarray(0, 4))).toEqual([-1, -1, -1, -1]);
        expect(labels[10]).toBe(0);
    });
});
