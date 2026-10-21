import { FaceCandidates } from './face-visibility';

/** Neighbour face across each edge (v0v1, v1v2, v2v0), −1 on borders. */
export function faceAdjacency(indices: Uint32Array, vertexCount: number): Int32Array {
    const faces = indices.length / 3;
    const offsets = new Int32Array(vertexCount + 1);
    for (const v of indices) offsets[v + 1]++;
    for (let v = 0; v < vertexCount; v++) offsets[v + 1] += offsets[v];
    const cursor = offsets.slice(0, -1);
    const incident = new Int32Array(indices.length);
    for (let f = 0; f < faces; f++) {
        for (let k = 0; k < 3; k++) incident[cursor[indices[f * 3 + k]]++] = f;
    }
    const adjacency = new Int32Array(faces * 3).fill(-1);
    for (let f = 0; f < faces; f++) {
        for (let k = 0; k < 3; k++) {
            const a = indices[f * 3 + k];
            const b = indices[f * 3 + ((k + 1) % 3)];
            for (let i = offsets[a]; i < offsets[a + 1]; i++) {
                const g = incident[i];
                if (g === f) continue;
                const g0 = indices[g * 3];
                const g1 = indices[g * 3 + 1];
                const g2 = indices[g * 3 + 2];
                if (g0 === b || g1 === b || g2 === b) {
                    adjacency[f * 3 + k] = g;
                    break;
                }
            }
        }
    }
    return adjacency;
}

/**
 * Picks one source view per face by growing charts: starting from the face with the highest
 * score, a chart keeps absorbing neighbours while its view still scores at least `ratio` of the
 * neighbour's own best view. Charts come out large (few seams) and every face is textured from
 * a view nearly as good as its best one.
 */
export function labelFaces(
    candidates: FaceCandidates,
    adjacency: Int32Array,
    ratio = 0.6,
): Int16Array {
    const faces = adjacency.length / 3;
    const { slots, view, score } = candidates;
    const labels = new Int16Array(faces).fill(-1);
    const order = [];
    for (let f = 0; f < faces; f++) if (score[f * slots] > 0) order.push(f);
    order.sort((a, b) => score[b * slots] - score[a * slots]);
    const queue = new Int32Array(faces);
    for (const seed of order) {
        if (labels[seed] >= 0) continue;
        const label = view[seed * slots];
        labels[seed] = label;
        let head = 0;
        let tail = 0;
        queue[tail++] = seed;
        while (head < tail) {
            const f = queue[head++];
            for (let k = 0; k < 3; k++) {
                const g = adjacency[f * 3 + k];
                if (g < 0 || labels[g] >= 0) continue;
                const best = score[g * slots];
                for (let s = 0; s < slots; s++) {
                    if (view[g * slots + s] !== label) continue;
                    if (score[g * slots + s] >= ratio * best) {
                        labels[g] = label;
                        queue[tail++] = g;
                    }
                    break;
                }
            }
        }
    }

    // Faces seen by no camera borrow the most common label around them.
    for (let round = 0; round < 8; round++) {
        let filled = 0;
        for (let f = 0; f < faces; f++) {
            if (labels[f] >= 0) continue;
            const around: number[] = [];
            for (let k = 0; k < 3; k++) {
                const g = adjacency[f * 3 + k];
                if (g >= 0 && labels[g] >= 0) around.push(labels[g]);
            }
            if (around.length === 0) continue;
            labels[f] = mostCommon(around);
            filled++;
        }
        if (filled === 0) break;
    }
    return labels;
}

/** Re-labels faces of tiny charts to a neighbouring chart's view when that view sees them. */
export function absorbSmallCharts(
    labels: Int16Array,
    candidates: FaceCandidates,
    adjacency: Int32Array,
    minFaces: number,
): void {
    const { chartOf, sizes } = connectedCharts(labels, adjacency);
    const { slots, view } = candidates;
    for (let f = 0; f < labels.length; f++) {
        if (sizes[chartOf[f]] >= minFaces) continue;
        let target = -1;
        let targetSize = 0;
        for (let k = 0; k < 3; k++) {
            const g = adjacency[f * 3 + k];
            if (g < 0 || labels[g] === labels[f] || sizes[chartOf[g]] < targetSize) continue;
            const label = labels[g];
            let seen = false;
            for (let s = 0; s < slots; s++) if (view[f * slots + s] === label) seen = true;
            if (seen) {
                target = label;
                targetSize = sizes[chartOf[g]];
            }
        }
        if (target >= 0) labels[f] = target;
    }
}

/** Connected components of faces sharing an edge and a label. */
export function connectedCharts(
    labels: Int16Array,
    adjacency: Int32Array,
): { chartOf: Int32Array; sizes: Int32Array; count: number } {
    const faces = labels.length;
    const chartOf = new Int32Array(faces).fill(-1);
    const sizes: number[] = [];
    const stack = new Int32Array(faces);
    for (let seed = 0; seed < faces; seed++) {
        if (chartOf[seed] >= 0) continue;
        const chart = sizes.length;
        let top = 0;
        let size = 0;
        stack[top++] = seed;
        chartOf[seed] = chart;
        while (top > 0) {
            const f = stack[--top];
            size++;
            for (let k = 0; k < 3; k++) {
                const g = adjacency[f * 3 + k];
                if (g < 0 || chartOf[g] >= 0 || labels[g] !== labels[f]) continue;
                chartOf[g] = chart;
                stack[top++] = g;
            }
        }
        sizes.push(size);
    }
    return { chartOf, sizes: Int32Array.from(sizes), count: sizes.length };
}

function mostCommon(values: number[]): number {
    let best = values[0];
    let bestCount = 0;
    for (const value of values) {
        let count = 0;
        for (const other of values) if (other === value) count++;
        if (count > bestCount) {
            bestCount = count;
            best = value;
        }
    }
    return best;
}
