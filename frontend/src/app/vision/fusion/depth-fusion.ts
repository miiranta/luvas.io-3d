import { Intrinsics, Pose, fromCamera } from '../geometry/camera';
import { applyTransposed3 } from '../math/mat3';
import { DenseStereo } from '../mvs/dense-stereo';
import { unpackDepthSample } from '../mvs/depth-maps';
import { PointAccumulator, PointCloud } from './point-accumulator';

/** One stereo view as the fusion sees it. */
export interface FusionView {
    intrinsics: Intrinsics;
    pose: Pose;
    /** Layers checked by the consistency filter and how many must agree. */
    checks: number[];
    needed: number;
    /** Small RGB image (any size) used to colour the points. */
    thumb: { width: number; height: number; rgb: Uint8Array };
}

/**
 * Runs the multi-view consistency filter on every depth map and merges the surviving samples
 * (position, normal, colour) on a voxel grid of `cell` metres, growing the cell while the cloud
 * exceeds `cellBudget` cells. The final cell size is a good voxel size for meshing.
 */
export async function fuseDepthMaps(
    stereo: DenseStereo,
    views: FusionView[],
    cell: number,
    cellBudget: number,
    progress: (fraction: number, detail: string) => void,
): Promise<{ cloud: PointCloud; cell: number }> {
    const accumulator = new PointAccumulator(cell);
    for (let layer = 0; layer < views.length; layer++) {
        progress(layer / views.length, `${layer + 1}/${views.length}`);
        const view = views[layer];
        const packed = await stereo.filteredMap(layer, view.checks, view.needed);
        addDepthMap(accumulator, packed, stereo.width, stereo.height, view);
        while (accumulator.size > cellBudget) accumulator.coarsen();
    }
    return { cloud: accumulator.cloud(), cell: accumulator.cell };
}

/** Decodes one packed depth map into world-space, coloured, oriented samples. */
function addDepthMap(
    accumulator: PointAccumulator,
    packed: Uint32Array,
    width: number,
    height: number,
    { intrinsics: k, pose, thumb }: FusionView,
): void {
    const sx = thumb.width / width;
    const sy = thumb.height / height;
    const sample = new Float64Array(4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (!unpackDepthSample(packed[y * width + x], sample)) continue;
            const d = sample[0];
            const world = fromCamera(pose, [((x - k.cx) / k.fx) * d, ((y - k.cy) / k.fy) * d, d]);
            const normal = applyTransposed3(pose.R, [sample[1], sample[2], sample[3]]);
            const tx = Math.min(thumb.width - 1, Math.floor((x + 0.5) * sx));
            const ty = Math.min(thumb.height - 1, Math.floor((y + 0.5) * sy));
            const i = (ty * thumb.width + tx) * 3;
            accumulator.add(
                world[0],
                world[1],
                world[2],
                normal[0],
                normal[1],
                normal[2],
                thumb.rgb[i],
                thumb.rgb[i + 1],
                thumb.rgb[i + 2],
            );
        }
    }
}
