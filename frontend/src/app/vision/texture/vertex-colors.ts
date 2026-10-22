import { FaceCandidates } from './face-visibility';
import { TextureView } from './texture-types';
import { sampleThumb } from './thumbnail';

const GREY = 150;

/**
 * Fast colouring without a texture: every vertex takes the two best views among the faces
 * around it and averages their colour (weighted by the view score), sampled from the small
 * keyframe thumbnails. RGBA8 per vertex.
 */
export function vertexColors(
    positions: Float32Array,
    indices: Uint32Array,
    candidates: FaceCandidates,
    views: TextureView[],
): Uint8Array {
    const vertices = positions.length / 3;
    const best = new Int32Array(vertices).fill(-1);
    const bestScore = new Float32Array(vertices);
    const second = new Int32Array(vertices).fill(-1);
    const secondScore = new Float32Array(vertices);
    const { slots, view, score } = candidates;
    for (let f = 0; f < indices.length / 3; f++) {
        for (let s = 0; s < Math.min(2, slots); s++) {
            const v = view[f * slots + s];
            if (v < 0) break;
            const sc = score[f * slots + s];
            for (let k = 0; k < 3; k++) {
                const vertex = indices[f * 3 + k];
                if (v === best[vertex]) {
                    bestScore[vertex] = Math.max(bestScore[vertex], sc);
                } else if (sc > bestScore[vertex]) {
                    second[vertex] = best[vertex];
                    secondScore[vertex] = bestScore[vertex];
                    best[vertex] = v;
                    bestScore[vertex] = sc;
                } else if (v !== second[vertex] && sc > secondScore[vertex]) {
                    second[vertex] = v;
                    secondScore[vertex] = sc;
                }
            }
        }
    }
    const colors = new Uint8Array(vertices * 4);
    const sample = new Float64Array(3);
    for (let i = 0; i < vertices; i++) {
        let r = 0;
        let g = 0;
        let b = 0;
        let weight = 0;
        for (const [v, w] of [
            [best[i], bestScore[i]],
            [second[i], secondScore[i]],
        ]) {
            if (
                v < 0 ||
                !sampleThumb(
                    views[v],
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2],
                    sample,
                )
            ) {
                continue;
            }
            r += sample[0] * w;
            g += sample[1] * w;
            b += sample[2] * w;
            weight += w;
        }
        colors[i * 4] = weight > 0 ? r / weight : GREY;
        colors[i * 4 + 1] = weight > 0 ? g / weight : GREY;
        colors[i * 4 + 2] = weight > 0 ? b / weight : GREY;
        colors[i * 4 + 3] = 255;
    }
    return colors;
}
