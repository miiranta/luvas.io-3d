import { vertexColors } from '../vertex-colors';
import { TextureMethod, visibleCandidates } from './texture-method';

/** Fastest: no atlas, each vertex coloured from its two best views' thumbnails. */
export const vertexColours: TextureMethod = {
    async selectViews(input, progress) {
        const candidates = await visibleCandidates(input, progress);
        return { candidates, adjacency: null, labels: null };
    },
    async bake(input, selection) {
        const { mesh } = input;
        return {
            model: {
                positions: mesh.positions,
                indices: mesh.indices,
                uvs: null,
                atlasJpeg: null,
                colors: vertexColors(
                    mesh.positions,
                    mesh.indices,
                    selection.candidates,
                    input.views,
                ),
            },
            charts: 0,
            untextured: 0,
            atlas: null,
        };
    },
};
