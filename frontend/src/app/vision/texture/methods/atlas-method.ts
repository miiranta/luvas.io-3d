import { bakeAtlas } from '../atlas-baker';
import { labelFacesMrf } from '../mrf-labeling';
import { absorbSmallCharts, faceAdjacency, fillUnseen, labelFaces } from '../view-labeling';
import { TextureMethod, visibleCandidates } from './texture-method';

const ATLAS_MAX = 4096;
const SMALL_CHART = 12;
const JPEG_QUALITY = 0.9;

export interface AtlasOptions {
    /** Level the colours across seams (besides the per-view exposure gains). */
    seams: boolean;
    /**
     * How each face gets its view: greedy chart growing (optionally merging tiny charts), or an
     * MRF trading view quality against seams with this seam cost (in units of a face's loss).
     */
    selection: { kind: 'charts'; mergeSmall: boolean } | { kind: 'mrf'; smoothness: number };
}

/**
 * Texture atlas: one view per face (chart growing or MRF), charts packed into an atlas with
 * per-view exposure gains; optionally seams levelled.
 */
export function atlasMethod(options: AtlasOptions): TextureMethod {
    return {
        async selectViews(input, progress) {
            const candidates = await visibleCandidates(input, (f) => progress(f * 0.9));
            const { mesh } = input;
            const adjacency = faceAdjacency(mesh.indices, mesh.positions.length / 3);
            const { selection } = options;
            let labels: Int16Array;
            if (selection.kind === 'mrf') {
                progress(0.9, 'optimising seams');
                labels = labelFacesMrf(candidates, adjacency, selection.smoothness);
                fillUnseen(labels, adjacency);
            } else {
                progress(0.9, 'growing charts');
                labels = labelFaces(candidates, adjacency);
                if (selection.mergeSmall) {
                    absorbSmallCharts(labels, candidates, adjacency, SMALL_CHART);
                    absorbSmallCharts(labels, candidates, adjacency, SMALL_CHART);
                }
            }
            return { candidates, adjacency, labels };
        },
        async bake(input, selection, progress) {
            if (!selection.labels || !selection.adjacency)
                throw new Error('atlas needs face labels');
            const baked = await bakeAtlas(
                input.mesh.positions,
                input.mesh.indices,
                selection.labels,
                selection.candidates,
                selection.adjacency,
                input.views,
                (view) => input.image(view),
                ATLAS_MAX,
                (fraction) => progress(fraction * 0.9),
                options.seams,
            );
            progress(0.9, 'encoding JPEG');
            const jpeg = await input.jpeg(
                { width: baked.atlas.width, height: baked.atlas.height, data: baked.atlas.pixels },
                JPEG_QUALITY,
            );
            return {
                model: {
                    positions: baked.positions,
                    indices: baked.indices,
                    uvs: baked.uvs,
                    atlasJpeg: jpeg,
                    colors: null,
                },
                charts: baked.charts,
                untextured: baked.untextured,
                atlas: [baked.atlas.width, baked.atlas.height],
            };
        },
    };
}
