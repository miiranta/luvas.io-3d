import { CAMERA_STRIDE } from '../../core/models/scan-results';

/**
 * Line segments drawing each camera as a small frustum (apex at the centre, image rectangle at
 * `depth` metres) plus the trajectory through the centres. Colours run from the first camera
 * (pink) to the last (yellow), matching the app palette. RGBA8 colours.
 */
export function cameraLines(
    cameras: Float32Array,
    depth: number,
): { positions: Float32Array; colors: Uint8Array } {
    const count = cameras.length / CAMERA_STRIDE;
    const segments = count * 8 + Math.max(0, count - 1);
    const positions = new Float32Array(segments * 6);
    const colors = new Uint8Array(segments * 8);
    let s = 0;
    const push = (a: number[], b: number[], color: number[]) => {
        positions.set(a, s * 6);
        positions.set(b, s * 6 + 3);
        colors.set(color, s * 8);
        colors.set(color, s * 8 + 4);
        s++;
    };
    const centre = (c: number) => [
        cameras[c * CAMERA_STRIDE + 9],
        cameras[c * CAMERA_STRIDE + 10],
        cameras[c * CAMERA_STRIDE + 11],
    ];
    for (let c = 0; c < count; c++) {
        const o = c * CAMERA_STRIDE;
        const R = cameras.subarray(o, o + 9);
        const tx = cameras[o + 12] * depth;
        const ty = cameras[o + 13] * depth;
        const apex = centre(c);
        // Camera → world: X = C + Rᵀ · x_cam.
        const corner = (x: number, y: number) => [
            apex[0] + R[0] * x + R[3] * y + R[6] * depth,
            apex[1] + R[1] * x + R[4] * y + R[7] * depth,
            apex[2] + R[2] * x + R[5] * y + R[8] * depth,
        ];
        const corners = [corner(-tx, -ty), corner(tx, -ty), corner(tx, ty), corner(-tx, ty)];
        const t = count > 1 ? c / (count - 1) : 0;
        const color = [
            Math.round(255 * (1 - t) + 255 * t),
            Math.round(93 * (1 - t) + 201 * t),
            Math.round(143 * (1 - t) + 60 * t),
            255,
        ];
        for (let k = 0; k < 4; k++) {
            push(apex, corners[k], color);
            push(corners[k], corners[(k + 1) % 4], color);
        }
        if (c > 0) push(centre(c - 1), apex, [255, 253, 245, 255]);
    }
    return { positions, colors };
}
