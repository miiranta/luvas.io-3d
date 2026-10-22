/** Indexed triangle mesh with per-vertex normals. */
export interface Mesh {
    positions: Float32Array;
    normals: Float32Array;
    indices: Uint32Array;
}

/** Area-weighted vertex normals. */
export function computeNormals(mesh: Mesh): void {
    const { positions: p, indices, normals } = mesh;
    normals.fill(0);
    for (let f = 0; f < indices.length; f += 3) {
        const a = indices[f] * 3;
        const b = indices[f + 1] * 3;
        const c = indices[f + 2] * 3;
        const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
        const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
        const nx = e1[1] * e2[2] - e1[2] * e2[1];
        const ny = e1[2] * e2[0] - e1[0] * e2[2];
        const nz = e1[0] * e2[1] - e1[1] * e2[0];
        for (const v of [a, b, c]) {
            normals[v] += nx;
            normals[v + 1] += ny;
            normals[v + 2] += nz;
        }
    }
    for (let v = 0; v < normals.length; v += 3) {
        const length = Math.hypot(normals[v], normals[v + 1], normals[v + 2]) || 1;
        normals[v] /= length;
        normals[v + 1] /= length;
        normals[v + 2] /= length;
    }
}

/**
 * Drops connected components smaller than `fraction` of the faces (and below `minFaces`),
 * which removes floating noise left by isolated wrong depths. Re-indexes the vertices.
 */
export function removeSmallComponents(mesh: Mesh, fraction: number, minFaces: number): Mesh {
    const vertices = mesh.positions.length / 3;
    const parent = new Int32Array(vertices);
    for (let i = 0; i < vertices; i++) parent[i] = i;
    const find = (i: number): number => {
        while (parent[i] !== i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    const { indices } = mesh;
    for (let f = 0; f < indices.length; f += 3) {
        const a = find(indices[f]);
        const b = find(indices[f + 1]);
        const c = find(indices[f + 2]);
        parent[b] = a;
        parent[find(c)] = a;
    }
    const faces = new Map<number, number>();
    for (let f = 0; f < indices.length; f += 3) {
        const root = find(indices[f]);
        faces.set(root, (faces.get(root) ?? 0) + 1);
    }
    const limit = Math.max(minFaces, (indices.length / 3) * fraction);
    const remap = new Int32Array(vertices).fill(-1);
    const positions: number[] = [];
    const normals: number[] = [];
    const kept: number[] = [];
    for (let f = 0; f < indices.length; f += 3) {
        if ((faces.get(find(indices[f])) ?? 0) < limit) continue;
        for (let k = 0; k < 3; k++) {
            const v = indices[f + k];
            if (remap[v] < 0) {
                remap[v] = positions.length / 3;
                positions.push(
                    mesh.positions[v * 3],
                    mesh.positions[v * 3 + 1],
                    mesh.positions[v * 3 + 2],
                );
                normals.push(mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]);
            }
            kept.push(remap[v]);
        }
    }
    return {
        positions: Float32Array.from(positions),
        normals: Float32Array.from(normals),
        indices: Uint32Array.from(kept),
    };
}

/** Total area of the triangles (m² for metric positions). */
export function surfaceArea(mesh: Mesh): number {
    const p = mesh.positions;
    let area = 0;
    for (let f = 0; f < mesh.indices.length; f += 3) {
        const a = mesh.indices[f] * 3;
        const b = mesh.indices[f + 1] * 3;
        const c = mesh.indices[f + 2] * 3;
        const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
        const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
        area +=
            0.5 *
            Math.hypot(
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            );
    }
    return area;
}
