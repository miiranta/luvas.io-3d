import { CAMERA_STRIDE } from '../../core/models/scan-results';
import { cameraLines } from './camera-lines';
import { writeGlb } from './glb-writer';
import { writePly } from './ply-writer';

describe('GLB writer', () => {
    it('writes a valid binary glTF with aligned chunks', () => {
        const glb = writeGlb([
            {
                name: 'mesh',
                mode: 4,
                positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
                uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
                indices: new Uint32Array([0, 1, 2]),
                texture: {
                    bytes: new Uint8Array([0xff, 0xd8, 0xff, 0xd9, 1]),
                    mimeType: 'image/jpeg',
                },
            },
            {
                name: 'sparse',
                mode: 0,
                positions: new Float32Array([0.5, 0.5, 0.5]),
                colors: new Uint8Array([1, 2, 3, 255]),
            },
        ]);
        const view = new DataView(glb);
        expect(view.getUint32(0, true)).toBe(0x46546c67);
        expect(view.getUint32(8, true)).toBe(glb.byteLength);
        const jsonLength = view.getUint32(12, true);
        expect(jsonLength % 4).toBe(0);
        const json = JSON.parse(new TextDecoder().decode(new Uint8Array(glb, 20, jsonLength)));
        expect(json.meshes.map((m: { name: string }) => m.name)).toEqual(['mesh', 'sparse']);
        expect(json.accessors[0].max).toEqual([1, 1, 0]);
        expect(json.images[0].mimeType).toBe('image/jpeg');
        for (const bufferView of json.bufferViews) expect(bufferView.byteOffset % 4).toBe(0);
        const binLength = view.getUint32(20 + jsonLength, true);
        expect(20 + jsonLength + 8 + binLength).toBe(glb.byteLength);
        expect(binLength).toBeGreaterThanOrEqual(json.buffers[0].byteLength);
    });
});

describe('PLY writer', () => {
    it('writes the header and one fixed-size record per point', () => {
        const ply = writePly(
            new Float32Array([1, 2, 3, 4, 5, 6]),
            new Uint8Array([9, 8, 7, 6, 5, 4]),
            new Float32Array(6),
        );
        const text = new TextDecoder().decode(ply);
        const header = text.slice(0, text.indexOf('end_header\n') + 'end_header\n'.length);
        expect(header).toContain('element vertex 2');
        expect(header).toContain('property float nx');
        expect(ply.byteLength - new TextEncoder().encode(header).length).toBe(2 * 27);
        const view = new DataView(ply, new TextEncoder().encode(header).length);
        expect(view.getFloat32(27 + 8, true)).toBe(6);
        expect(view.getUint8(27 + 24)).toBe(6);
    });
});

describe('camera lines', () => {
    it('draws a frustum per camera plus the path between them', () => {
        const cameras = new Float32Array(2 * CAMERA_STRIDE);
        for (const c of [0, 1]) {
            cameras.set([1, 0, 0, 0, 1, 0, 0, 0, 1], c * CAMERA_STRIDE);
            cameras.set([c, 0, 0, 0.5, 0.4], c * CAMERA_STRIDE + 9);
        }
        const lines = cameraLines(cameras, 0.1);
        expect(lines.positions.length / 6).toBe(2 * 8 + 1);
        // First segment: apex → corner (−tx, −ty) at depth 0.1 along +z.
        expect(Array.from(lines.positions.slice(3, 6)).map((v) => +v.toFixed(5))).toEqual([
            -0.05, -0.04, 0.1,
        ]);
    });
});
