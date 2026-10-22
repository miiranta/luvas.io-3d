import { Mesh, computeNormals } from './mesh';
import { BRICK } from './tsdf-volume';

export interface SdfBricks {
    origin: [number, number, number];
    voxel: number;
    /** Grid size in bricks. */
    dims: [number, number, number];
    lookup: Int32Array;
    tsdf: Float32Array;
    weight: Float32Array;
}

const BRICK_VOXELS = BRICK * BRICK * BRICK;

/** Brick-local voxel index → cube corner offsets used for the 12 edges of a cell. */
const CORNERS = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
const EDGES = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/**
 * Naive surface nets on a sparse brick TSDF: one vertex per cell whose corners change sign
 * (placed at the mean of the edge crossings), one quad per sign-changing voxel edge. Cells with
 * a corner observed less than `minWeight` are skipped, so unobserved space never makes faces.
 */
export function extractSurface(volume: SdfBricks, minWeight: number): Mesh {
    const [bx, by, bz] = volume.dims;
    const extent = [bx * BRICK, by * BRICK, bz * BRICK];
    const bricks = volume.tsdf.length / BRICK_VOXELS;

    const slot = (x: number, y: number, z: number): number => {
        if (x < 0 || y < 0 || z < 0 || x >= extent[0] || y >= extent[1] || z >= extent[2])
            return -1;
        const brick = volume.lookup[((z >> 3) * by + (y >> 3)) * bx + (x >> 3)];
        if (brick < 0) return -1;
        return brick * BRICK_VOXELS + (x & 7) + ((y & 7) << 3) + ((z & 7) << 6);
    };

    const cellVertex = new Int32Array(bricks * BRICK_VOXELS).fill(-1);
    const positions: number[] = [];
    const values = new Float32Array(8);
    const origin = volume.origin;
    const voxel = volume.voxel;

    // Brick coordinates of every pool entry, recovered from the lookup table.
    const brickCoord = new Int32Array(bricks * 3);
    for (let i = 0; i < volume.lookup.length; i++) {
        const b = volume.lookup[i];
        if (b < 0) continue;
        brickCoord[b * 3] = i % bx;
        brickCoord[b * 3 + 1] = Math.floor(i / bx) % by;
        brickCoord[b * 3 + 2] = Math.floor(i / (bx * by));
    }

    const sampleCell = (x: number, y: number, z: number): boolean => {
        let inside = 0;
        for (let c = 0; c < 8; c++) {
            const s = slot(x + CORNERS[c][0], y + CORNERS[c][1], z + CORNERS[c][2]);
            if (s < 0 || volume.weight[s] < minWeight) return false;
            values[c] = volume.tsdf[s];
            if (values[c] < 0) inside++;
        }
        return inside > 0 && inside < 8;
    };

    // Pass 1: vertices.
    for (let b = 0; b < bricks; b++) {
        const x0 = brickCoord[b * 3] * BRICK;
        const y0 = brickCoord[b * 3 + 1] * BRICK;
        const z0 = brickCoord[b * 3 + 2] * BRICK;
        for (let local = 0; local < BRICK_VOXELS; local++) {
            const x = x0 + (local & 7);
            const y = y0 + ((local >> 3) & 7);
            const z = z0 + (local >> 6);
            if (!sampleCell(x, y, z)) continue;
            let sx = 0;
            let sy = 0;
            let sz = 0;
            let n = 0;
            for (const [a, c] of EDGES) {
                const va = values[a];
                const vc = values[c];
                if (va < 0 === vc < 0) continue;
                const t = va / (va - vc);
                sx += CORNERS[a][0] + (CORNERS[c][0] - CORNERS[a][0]) * t;
                sy += CORNERS[a][1] + (CORNERS[c][1] - CORNERS[a][1]) * t;
                sz += CORNERS[a][2] + (CORNERS[c][2] - CORNERS[a][2]) * t;
                n++;
            }
            cellVertex[b * BRICK_VOXELS + local] = positions.length / 3;
            positions.push(
                origin[0] + (x + 0.5 + sx / n) * voxel,
                origin[1] + (y + 0.5 + sy / n) * voxel,
                origin[2] + (z + 0.5 + sz / n) * voxel,
            );
        }
    }

    // Pass 2: one quad per sign-changing edge, wound so the normal points to free space.
    const indices: number[] = [];
    const vertexAt = (x: number, y: number, z: number): number => {
        const s = slot(x, y, z);
        return s < 0 ? -1 : cellVertex[s];
    };
    const axes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ];
    for (let b = 0; b < bricks; b++) {
        const x0 = brickCoord[b * 3] * BRICK;
        const y0 = brickCoord[b * 3 + 1] * BRICK;
        const z0 = brickCoord[b * 3 + 2] * BRICK;
        for (let local = 0; local < BRICK_VOXELS; local++) {
            const x = x0 + (local & 7);
            const y = y0 + ((local >> 3) & 7);
            const z = z0 + (local >> 6);
            const s0 = b * BRICK_VOXELS + local;
            if (volume.weight[s0] < minWeight) continue;
            const v0 = volume.tsdf[s0];
            for (let a = 0; a < 3; a++) {
                const [ax, ay, az] = axes[a];
                const s1 = slot(x + ax, y + ay, z + az);
                if (s1 < 0 || volume.weight[s1] < minWeight) continue;
                const v1 = volume.tsdf[s1];
                if (v0 < 0 === v1 < 0) continue;
                const [ux, uy, uz] = axes[(a + 1) % 3];
                const [wx, wy, wz] = axes[(a + 2) % 3];
                const q0 = vertexAt(x - ux - wx, y - uy - wy, z - uz - wz);
                const q1 = vertexAt(x - wx, y - wy, z - wz);
                const q2 = vertexAt(x, y, z);
                const q3 = vertexAt(x - ux, y - uy, z - uz);
                if (q0 < 0 || q1 < 0 || q2 < 0 || q3 < 0) continue;
                const quad = v0 < 0 ? [q0, q1, q2, q3] : [q0, q3, q2, q1];
                // Split along the shorter diagonal.
                if (
                    distance2(positions, quad[0], quad[2]) <= distance2(positions, quad[1], quad[3])
                ) {
                    indices.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
                } else {
                    indices.push(quad[0], quad[1], quad[3], quad[1], quad[2], quad[3]);
                }
            }
        }
    }
    const mesh = {
        positions: Float32Array.from(positions),
        normals: new Float32Array(positions.length),
        indices: Uint32Array.from(indices),
    };
    computeNormals(mesh);
    return mesh;
}

function distance2(p: number[], a: number, b: number): number {
    const dx = p[a * 3] - p[b * 3];
    const dy = p[a * 3 + 1] - p[b * 3 + 1];
    const dz = p[a * 3 + 2] - p[b * 3 + 2];
    return dx * dx + dy * dy + dz * dz;
}
