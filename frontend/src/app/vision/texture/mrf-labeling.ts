import { FaceCandidates } from './face-visibility';
import { labelFaces } from './view-labeling';

/**
 * One source view per face as a Markov random field (Lempitsky & Ivanov 2007, Waechter et al.
 * 2014): each face pays for using a worse view than its best one, each edge between faces with
 * different views pays `smoothness` (a seam). Minimised over every face's candidate views by
 * min-sum loopy belief propagation (Potts messages cost O(candidates) per edge), then polished
 * with iterated conditional modes, which never increases the energy. Loopy BP can stall with
 * domain walls between nearly tied views, so the greedy chart-growing labelling is polished too
 * and the lower-energy of the two is returned: never worse than chart growing.
 *
 * Data term: 1 − score / best score of the face, so 0 for its best view and close to 1 for a
 * view that sees it barely (score = projected area × sharpness, a proxy of Waechter's gradient
 * integral). Faces seen by no view keep −1.
 */
export function labelFacesMrf(
    candidates: FaceCandidates,
    adjacency: Int32Array,
    smoothness: number,
    sweeps = 12,
): Int16Array {
    const { slots, view, score } = candidates;
    const faces = adjacency.length / 3;
    const data = new Float32Array(faces * slots).fill(Infinity);
    for (let f = 0; f < faces; f++) {
        const best = score[f * slots];
        if (!(best > 0)) continue;
        for (let s = 0; s < slots; s++) {
            const value = score[f * slots + s];
            if (value > 0 && view[f * slots + s] >= 0) data[f * slots + s] = 1 - value / best;
        }
    }
    // Which edge of the neighbour points back at the face.
    const reverse = new Int8Array(faces * 3).fill(-1);
    for (let f = 0; f < faces; f++) {
        for (let k = 0; k < 3; k++) {
            const g = adjacency[f * 3 + k];
            if (g < 0) continue;
            for (let j = 0; j < 3; j++) if (adjacency[g * 3 + j] === f) reverse[f * 3 + k] = j;
        }
    }
    // messages[(f·3 + k)·slots + s]: from the neighbour across edge k of f, over f's slots.
    const messages = new Float32Array(faces * 3 * slots);
    const h = new Float32Array(slots);
    const outgoing = new Float32Array(slots);
    const send = (f: number, k: number) => {
        const g = adjacency[f * 3 + k];
        const back = reverse[f * 3 + k];
        if (g < 0 || back < 0 || !(data[g * slots] < Infinity)) return;
        // h(s) = data + everything f heard except from g.
        let lowest = Infinity;
        for (let s = 0; s < slots; s++) {
            let value = data[f * slots + s];
            for (let j = 0; j < 3; j++) if (j !== k) value += messages[(f * 3 + j) * slots + s];
            h[s] = value;
            if (value < lowest) lowest = value;
        }
        if (!(lowest < Infinity)) return;
        // Potts: g taking label l costs h(l) if f can take l too, else the cheapest + a seam.
        let floor = Infinity;
        for (let t = 0; t < slots; t++) {
            const label = view[g * slots + t];
            let value = lowest + smoothness;
            for (let s = 0; s < slots; s++) {
                if (view[f * slots + s] === label && h[s] < value) value = h[s];
            }
            outgoing[t] = value;
            if (value < floor) floor = value;
        }
        const target = (g * 3 + back) * slots;
        for (let t = 0; t < slots; t++) {
            // Normalised and damped (plain loopy BP oscillates on meshes).
            messages[target + t] = 0.5 * messages[target + t] + 0.5 * (outgoing[t] - floor);
        }
    };
    for (let sweep = 0; sweep < sweeps; sweep++) {
        if (sweep % 2 === 0) {
            for (let f = 0; f < faces; f++) for (let k = 0; k < 3; k++) send(f, k);
        } else {
            for (let f = faces - 1; f >= 0; f--) for (let k = 0; k < 3; k++) send(f, k);
        }
    }

    const labels = new Int16Array(faces).fill(-1);
    for (let f = 0; f < faces; f++) {
        let best = Infinity;
        for (let s = 0; s < slots; s++) {
            let value = data[f * slots + s];
            for (let k = 0; k < 3; k++) value += messages[(f * 3 + k) * slots + s];
            if (value < best) {
                best = value;
                labels[f] = view[f * slots + s];
            }
        }
    }
    polish(labels, data, candidates, adjacency, smoothness);
    const greedy = labelFaces(candidates, adjacency);
    for (let f = 0; f < faces; f++) if (!(data[f * slots] < Infinity)) greedy[f] = -1;
    polish(greedy, data, candidates, adjacency, smoothness);
    const energy = (l: Int16Array) => {
        const { data, seams } = labellingEnergy(l, candidates, adjacency, smoothness);
        return data + seams;
    };
    return energy(labels) <= energy(greedy) ? labels : greedy;
}

/** Iterated conditional modes: each face takes the label of least local energy, until stable. */
function polish(
    labels: Int16Array,
    data: Float32Array,
    candidates: FaceCandidates,
    adjacency: Int32Array,
    smoothness: number,
): void {
    const { slots, view } = candidates;
    for (let round = 0; round < 6; round++) {
        let changed = 0;
        for (let f = 0; f < labels.length; f++) {
            if (labels[f] < 0) continue;
            let best = labels[f];
            let bestCost = Infinity;
            for (let s = 0; s < slots; s++) {
                const label = view[f * slots + s];
                if (!(data[f * slots + s] < Infinity)) continue;
                let cost = data[f * slots + s];
                for (let k = 0; k < 3; k++) {
                    const g = adjacency[f * 3 + k];
                    if (g >= 0 && labels[g] >= 0 && labels[g] !== label) cost += smoothness;
                }
                if (cost < bestCost - 1e-6) {
                    bestCost = cost;
                    best = label;
                }
            }
            if (best !== labels[f]) {
                labels[f] = best;
                changed++;
            }
        }
        if (changed === 0) break;
    }
}

/** MRF energy of a labelling (for comparisons). */
export function labellingEnergy(
    labels: Int16Array,
    candidates: FaceCandidates,
    adjacency: Int32Array,
    smoothness: number,
): { data: number; seams: number } {
    const { slots, view, score } = candidates;
    let data = 0;
    let seams = 0;
    for (let f = 0; f < labels.length; f++) {
        const best = score[f * slots];
        if (labels[f] < 0 || !(best > 0)) continue;
        for (let s = 0; s < slots; s++) {
            if (view[f * slots + s] === labels[f]) data += 1 - score[f * slots + s] / best;
        }
        for (let k = 0; k < 3; k++) {
            const g = adjacency[f * 3 + k];
            if (g > f && labels[g] >= 0 && labels[g] !== labels[f]) seams++;
        }
    }
    return { data, seams: seams * smoothness };
}
