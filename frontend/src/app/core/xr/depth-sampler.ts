import { KeyframeMessage } from '../models/worker-protocol';

const DEPTH_GRID = 64;

/** Phone depth on a regular grid (64 on the long side) of normalised view coordinates; 0 = none. */
export function sampleDepth(
    frame: XRFrame,
    view: XRView,
    width: number,
    height: number,
): KeyframeMessage['depth'] {
    let info: XRCPUDepthInformation | null = null;
    try {
        info = frame.getDepthInformation(view);
    } catch {
        return null;
    }
    if (!info) return null;
    const long = Math.max(width, height);
    const gw = Math.round((DEPTH_GRID * width) / long);
    const gh = Math.round((DEPTH_GRID * height) / long);
    const meters = new Float32Array(gw * gh);
    for (let y = 0; y < gh; y++) {
        for (let x = 0; x < gw; x++) {
            try {
                const d = info.getDepthInMeters((x + 0.5) / gw, (y + 0.5) / gh);
                meters[y * gw + x] = d > 0.05 && d < 8 ? d : 0;
            } catch {
                meters[y * gw + x] = 0;
            }
        }
    }
    return { width: gw, height: gh, meters: meters.buffer };
}
