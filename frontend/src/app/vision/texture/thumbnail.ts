import { TextureView } from './texture-types';

/** Bilinear RGB sample of a world point in a view's thumbnail; false when outside. */
export function sampleThumb(
    view: TextureView,
    x: number,
    y: number,
    z: number,
    out: Float64Array,
): boolean {
    const t = view.thumb;
    const k = t.intrinsics;
    const { R } = view.pose;
    const tr = view.pose.t;
    const cx = R[0] * x + R[1] * y + R[2] * z + tr[0];
    const cy = R[3] * x + R[4] * y + R[5] * z + tr[1];
    const cz = R[6] * x + R[7] * y + R[8] * z + tr[2];
    if (cz <= 0.01) return false;
    const u = (k.fx * cx) / cz + k.cx;
    const v = (k.fy * cy) / cz + k.cy;
    if (u < 0 || v < 0 || u > t.width - 1.001 || v > t.height - 1.001) return false;
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const wx = u - x0;
    const wy = v - y0;
    for (let c = 0; c < 3; c++) {
        const i = (y0 * t.width + x0) * 3 + c;
        const top = t.rgb[i] * (1 - wx) + t.rgb[i + 3] * wx;
        const bottom = t.rgb[i + t.width * 3] * (1 - wx) + t.rgb[i + t.width * 3 + 3] * wx;
        out[c] = top * (1 - wy) + bottom * wy;
    }
    return true;
}
