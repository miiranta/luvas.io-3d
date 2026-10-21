import { PointAccumulator } from './point-accumulator';
import { extractSurface, removeSmallComponents } from './surface-nets';
import { BRICK, BrickGrid } from './tsdf-volume';

/** Sparse brick TSDF of a sphere (truncated at ±4 voxels), as the GPU fusion would produce. */
function sphereVolume(radius: number, voxel: number) {
    const extent = radius * 2 + voxel * BRICK * 2;
    const origin: [number, number, number] = [-extent / 2, -extent / 2, -extent / 2];
    const grid = new BrickGrid(origin, voxel, [extent, extent, extent]);
    const mu = voxel * 4;
    for (let a = 0; a < 2000; a++) {
        const z = (a / 1999) * 2 - 1;
        const phi = a * 2.399963;
        const r = Math.sqrt(1 - z * z);
        grid.mark(Math.cos(phi) * r * radius, Math.sin(phi) * r * radius, z * radius, mu);
    }
    const layout = grid.build();
    const tsdf = new Float32Array(layout.count * 512).fill(1);
    const weight = new Float32Array(layout.count * 512);
    for (let b = 0; b < layout.count; b++) {
        for (let local = 0; local < 512; local++) {
            const v = [local & 7, (local >> 3) & 7, local >> 6];
            const p = [0, 1, 2].map(
                (k) => origin[k] + (layout.coords[b * 4 + k] * BRICK + v[k] + 0.5) * voxel,
            );
            const sdf = Math.hypot(p[0], p[1], p[2]) - radius;
            tsdf[b * 512 + local] = Math.max(-1, Math.min(1, sdf / mu));
            weight[b * 512 + local] = sdf > -mu ? 2 : 0;
        }
    }
    return { origin, voxel, dims: grid.dims, lookup: layout.lookup, tsdf, weight };
}

describe('surface nets', () => {
    const radius = 0.05;
    const voxel = 0.002;
    const mesh = extractSurface(sphereVolume(radius, voxel), 0.5);

    it('places vertices on the zero level set', () => {
        expect(mesh.indices.length / 3).toBeGreaterThan(5000);
        let worst = 0;
        for (let i = 0; i < mesh.positions.length; i += 3) {
            const r = Math.hypot(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]);
            worst = Math.max(worst, Math.abs(r - radius));
        }
        expect(worst).toBeLessThan(voxel * 0.35);
    });

    it('builds a closed surface with outward normals', () => {
        const edges = new Map<string, number>();
        for (let f = 0; f < mesh.indices.length; f += 3) {
            for (let k = 0; k < 3; k++) {
                const a = mesh.indices[f + k];
                const b = mesh.indices[f + ((k + 1) % 3)];
                const key = a < b ? `${a}:${b}` : `${b}:${a}`;
                edges.set(key, (edges.get(key) ?? 0) + 1);
            }
        }
        expect([...edges.values()].every((n) => n === 2)).toBe(true);
        let outward = 0;
        for (let i = 0; i < mesh.positions.length; i += 3) {
            const dot =
                mesh.normals[i] * mesh.positions[i] +
                mesh.normals[i + 1] * mesh.positions[i + 1] +
                mesh.normals[i + 2] * mesh.positions[i + 2];
            if (dot > 0) outward++;
        }
        expect(outward / (mesh.positions.length / 3)).toBeGreaterThan(0.99);
    });

    it('keeps the main component and drops floaters', () => {
        const floater = {
            positions: Float32Array.from([...mesh.positions, 1, 1, 1, 1.001, 1, 1, 1, 1.001, 1]),
            normals: new Float32Array(mesh.normals.length + 9),
            indices: Uint32Array.from([
                ...mesh.indices,
                ...[0, 1, 2].map((k) => mesh.positions.length / 3 + k),
            ]),
        };
        const cleaned = removeSmallComponents(floater, 0.01, 10);
        expect(cleaned.indices.length).toBe(mesh.indices.length);
    });
});

describe('point accumulator', () => {
    it('averages samples per cell and merges cells when coarsened', () => {
        const accumulator = new PointAccumulator(0.01);
        accumulator.add(0.001, 0.001, 0.001, 0, 0, 1, 100, 0, 0);
        accumulator.add(0.003, 0.003, 0.003, 0, 0, 1, 200, 0, 0);
        accumulator.add(0.015, 0.001, 0.001, 1, 0, 0, 0, 50, 0);
        let cloud = accumulator.cloud();
        expect(cloud.positions.length / 3).toBe(2);
        const first = [...cloud.support].indexOf(2);
        expect(cloud.positions[first * 3]).toBeCloseTo(0.002, 6);
        expect(cloud.colors[first * 3]).toBe(150);
        accumulator.coarsen(2);
        cloud = accumulator.cloud();
        expect(cloud.positions.length / 3).toBe(1);
        expect(cloud.support[0]).toBe(3);
    });

    it('grows past its initial capacity', () => {
        const accumulator = new PointAccumulator(0.001);
        for (let i = 0; i < 100000; i++) accumulator.add(i * 0.002, 0, 0, 0, 1, 0, 1, 2, 3);
        expect(accumulator.size).toBe(100000);
    });
});
