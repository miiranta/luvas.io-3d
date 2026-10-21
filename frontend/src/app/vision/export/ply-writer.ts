/** Binary little-endian PLY point cloud: xyz, optional normals, rgb. */
export function writePly(
    positions: Float32Array,
    colors: Uint8Array,
    normals?: Float32Array,
    comment = 'luvas.io scan 3d',
): ArrayBuffer {
    const count = positions.length / 3;
    const header =
        [
            'ply',
            'format binary_little_endian 1.0',
            `comment ${comment}`,
            `element vertex ${count}`,
            'property float x',
            'property float y',
            'property float z',
            ...(normals ? ['property float nx', 'property float ny', 'property float nz'] : []),
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header',
        ].join('\n') + '\n';
    const headerBytes = new TextEncoder().encode(header);
    const stride = (normals ? 24 : 12) + 3;
    const out = new Uint8Array(headerBytes.length + count * stride);
    out.set(headerBytes, 0);
    const view = new DataView(out.buffer);
    let o = headerBytes.length;
    for (let i = 0; i < count; i++) {
        view.setFloat32(o, positions[i * 3], true);
        view.setFloat32(o + 4, positions[i * 3 + 1], true);
        view.setFloat32(o + 8, positions[i * 3 + 2], true);
        o += 12;
        if (normals) {
            view.setFloat32(o, normals[i * 3], true);
            view.setFloat32(o + 4, normals[i * 3 + 1], true);
            view.setFloat32(o + 8, normals[i * 3 + 2], true);
            o += 12;
        }
        out[o] = colors[i * 3];
        out[o + 1] = colors[i * 3 + 1];
        out[o + 2] = colors[i * 3 + 2];
        o += 3;
    }
    return out.buffer;
}
