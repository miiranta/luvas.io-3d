import { Intrinsics, Pose } from '../geometry/camera';
import { robustBounds } from '../math/bounds';
import { DenseStereo } from '../mvs/dense-stereo';
import { Mesh, removeSmallComponents } from './mesh';
import { extractSurface } from './surface-nets';
import { BRICK, BrickGrid, TsdfVolume } from './tsdf-volume';

/** One depth map to integrate. */
export interface MeshingView {
    intrinsics: Intrinsics;
    pose: Pose;
    checks: number[];
    needed: number;
    /** Farthest depth worth integrating (m). */
    maxDepth: number;
    /** Phone depth at the stereo resolution (fills stereo holes at low weight), or null. */
    prior(): Float32Array | null;
}

/** 16 k bricks × 512 voxels × 8 B = 64 MB of GPU memory (and the same once read back). */
const BRICK_BUDGET = 16_000;
const GRID_CELL_BUDGET = 8_000_000;
const MIN_WEIGHT = 0.6;

/**
 * TSDF fusion of the filtered depth maps on the GPU, in bricks allocated only around the dense
 * cloud (`surface`, which also bounds the volume) and the `extra` points (phone depth: blank walls
 * stereo cannot see), then surface nets and removal of small floating parts.
 * The voxel grows when the volume would exceed the memory budget; the used voxel is returned.
 */
export async function meshDepthMaps(
    device: GPUDevice,
    stereo: DenseStereo,
    views: MeshingView[],
    surface: Float32Array,
    extra: Float32Array,
    voxel: number,
    /** Typical scene depth for confidence-weighted integration (see IntegrationView), 0 = off. */
    confidence: number,
    progress: (fraction: number, detail: string) => void,
): Promise<{ mesh: Mesh; voxel: number }> {
    progress(-1, 'allocating volume');
    const bounds = robustBounds(surface);
    let grid: BrickGrid;
    let layout: ReturnType<BrickGrid['build']>;
    for (;;) {
        const margin = voxel * BRICK * 2;
        const origin: [number, number, number] = [
            bounds.min[0] - margin,
            bounds.min[1] - margin,
            bounds.min[2] - margin,
        ];
        grid = new BrickGrid(origin, voxel, [
            bounds.max[0] - bounds.min[0] + 2 * margin,
            bounds.max[1] - bounds.min[1] + 2 * margin,
            bounds.max[2] - bounds.min[2] + 2 * margin,
        ]);
        if (grid.cells > GRID_CELL_BUDGET) {
            voxel *= 1.4;
            continue;
        }
        for (const points of [surface, extra]) {
            for (let i = 0; i < points.length; i += 3) {
                grid.mark(points[i], points[i + 1], points[i + 2], voxel * 4);
            }
        }
        layout = grid.build();
        if (layout.count <= BRICK_BUDGET) break;
        voxel *= 1.3;
    }
    const tsdf = new TsdfVolume(device, grid, layout, stereo.filtered, stereo.width, stereo.height);
    for (let layer = 0; layer < views.length; layer++) {
        progress((0.8 * layer) / views.length, `integrating ${layer + 1}/${views.length}`);
        const view = views[layer];
        await stereo.filter(layer, view.checks, view.needed);
        await tsdf.integrate({
            intrinsics: view.intrinsics,
            pose: view.pose,
            maxDepth: view.maxDepth,
            prior: view.prior(),
            confidence,
        });
    }
    progress(0.85, 'extracting surface');
    const volume = await tsdf.read();
    tsdf.destroy();
    const mesh = extractSurface(
        { origin: grid.origin, voxel, dims: grid.dims, lookup: layout.lookup, ...volume },
        MIN_WEIGHT,
    );
    return { mesh: removeSmallComponents(mesh, 0.01, 200), voxel };
}
