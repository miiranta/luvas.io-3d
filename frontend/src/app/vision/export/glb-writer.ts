export interface GlbMesh {
    name: string;
    /** 0 = points, 1 = lines, 4 = triangles. */
    mode: 0 | 1 | 4;
    positions: Float32Array;
    uvs?: Float32Array;
    /** RGBA8 per vertex. */
    colors?: Uint8Array;
    indices?: Uint32Array;
    /** Encoded image (JPEG/PNG) used as unlit base colour; requires `uvs`. */
    texture?: { bytes: Uint8Array; mimeType: string };
}

interface View {
    buffer: 0;
    byteOffset: number;
    byteLength: number;
    target?: number;
}

/** Minimal glTF 2.0 binary writer: one node per mesh, unlit materials (baked lighting). */
export function writeGlb(meshes: GlbMesh[]): ArrayBuffer {
    const chunks: Uint8Array[] = [];
    let offset = 0;
    const bufferViews: View[] = [];
    const accessors: Record<string, unknown>[] = [];
    const materials: Record<string, unknown>[] = [];
    const textures: Record<string, unknown>[] = [];
    const images: Record<string, unknown>[] = [];
    const gltfMeshes: Record<string, unknown>[] = [];

    const pushView = (bytes: Uint8Array, target?: number): number => {
        const padding = (4 - (offset % 4)) % 4;
        if (padding) {
            chunks.push(new Uint8Array(padding));
            offset += padding;
        }
        bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.byteLength, target });
        chunks.push(bytes);
        offset += bytes.byteLength;
        return bufferViews.length - 1;
    };
    const asBytes = (array: ArrayBufferView) =>
        new Uint8Array(array.buffer, array.byteOffset, array.byteLength);

    for (const mesh of meshes) {
        const attributes: Record<string, number> = {};
        const count = mesh.positions.length / 3;
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < count; i++) {
            for (let a = 0; a < 3; a++) {
                min[a] = Math.min(min[a], mesh.positions[i * 3 + a]);
                max[a] = Math.max(max[a], mesh.positions[i * 3 + a]);
            }
        }
        accessors.push({
            bufferView: pushView(asBytes(mesh.positions), 34962),
            componentType: 5126,
            count,
            type: 'VEC3',
            min: count ? min : [0, 0, 0],
            max: count ? max : [0, 0, 0],
        });
        attributes['POSITION'] = accessors.length - 1;
        if (mesh.uvs) {
            accessors.push({
                bufferView: pushView(asBytes(mesh.uvs), 34962),
                componentType: 5126,
                count,
                type: 'VEC2',
            });
            attributes['TEXCOORD_0'] = accessors.length - 1;
        }
        if (mesh.colors) {
            accessors.push({
                bufferView: pushView(asBytes(mesh.colors), 34962),
                componentType: 5121,
                normalized: true,
                count,
                type: 'VEC4',
            });
            attributes['COLOR_0'] = accessors.length - 1;
        }
        const primitive: Record<string, unknown> = { attributes, mode: mesh.mode };
        if (mesh.indices) {
            accessors.push({
                bufferView: pushView(asBytes(mesh.indices), 34963),
                componentType: 5125,
                count: mesh.indices.length,
                type: 'SCALAR',
            });
            primitive['indices'] = accessors.length - 1;
        }
        const material: Record<string, unknown> = {
            name: mesh.name,
            pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 1 },
            extensions: { KHR_materials_unlit: {} },
            doubleSided: mesh.mode !== 4,
        };
        if (mesh.texture) {
            images.push({
                bufferView: pushView(mesh.texture.bytes),
                mimeType: mesh.texture.mimeType,
            });
            textures.push({ source: images.length - 1, sampler: 0 });
            (material['pbrMetallicRoughness'] as Record<string, unknown>)['baseColorTexture'] = {
                index: textures.length - 1,
            };
        }
        materials.push(material);
        primitive['material'] = materials.length - 1;
        gltfMeshes.push({ name: mesh.name, primitives: [primitive] });
    }

    const json: Record<string, unknown> = {
        asset: { version: '2.0', generator: 'luvas.io scan 3d' },
        extensionsUsed: ['KHR_materials_unlit'],
        scene: 0,
        scenes: [{ nodes: meshes.map((_, i) => i) }],
        nodes: meshes.map((mesh, i) => ({ name: mesh.name, mesh: i })),
        meshes: gltfMeshes,
        materials,
        accessors,
        bufferViews,
        buffers: [{ byteLength: offset }],
    };
    if (images.length) {
        json['images'] = images;
        json['textures'] = textures;
        json['samplers'] = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];
    }

    const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
    const jsonLength = Math.ceil(jsonBytes.length / 4) * 4;
    const binLength = Math.ceil(offset / 4) * 4;
    const total = 12 + 8 + jsonLength + 8 + binLength;
    const out = new Uint8Array(total);
    const view = new DataView(out.buffer);
    view.setUint32(0, 0x46546c67, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, total, true);
    view.setUint32(12, jsonLength, true);
    view.setUint32(16, 0x4e4f534a, true);
    out.set(jsonBytes, 20);
    out.fill(0x20, 20 + jsonBytes.length, 20 + jsonLength);
    const binStart = 20 + jsonLength;
    view.setUint32(binStart, binLength, true);
    view.setUint32(binStart + 4, 0x004e4942, true);
    let cursor = binStart + 8;
    for (const chunk of chunks) {
        out.set(chunk, cursor);
        cursor += chunk.byteLength;
    }
    return out.buffer;
}
