import { Pose, fromCamera, scaleIntrinsics } from '../../geometry/camera';
import { PointAccumulator } from '../../fusion/point-accumulator';
import { StoredKeyframe } from './keyframe';

const PREVIEW_CELL = 0.015;
const PREVIEW_LIMIT = 60000;

/** Coarse coloured cloud of the phone depth, for the live AR overlay. */
export class DepthPreview {
    private readonly points = new PointAccumulator(PREVIEW_CELL);

    /** Back-projects every other depth sample of the keyframe from `pose`. */
    add(keyframe: StoredKeyframe, pose: Pose): void {
        const grid = keyframe.depth;
        if (!grid) return;
        const k = scaleIntrinsics(
            keyframe.work.intrinsics,
            grid.width / keyframe.work.width,
            grid.height / keyframe.work.height,
        );
        const t = keyframe.thumb;
        for (let y = 0; y < grid.height; y += 2) {
            for (let x = 0; x < grid.width; x += 2) {
                const d = grid.meters[y * grid.width + x];
                if (!(d > 0.05 && d < 6)) continue;
                const world = fromCamera(pose, [
                    ((x - k.cx) / k.fx) * d,
                    ((y - k.cy) / k.fy) * d,
                    d,
                ]);
                const tx = Math.min(
                    t.width - 1,
                    Math.round(((x + 0.5) / grid.width) * t.width - 0.5),
                );
                const ty = Math.min(
                    t.height - 1,
                    Math.round(((y + 0.5) / grid.height) * t.height - 0.5),
                );
                const i = (ty * t.width + tx) * 3;
                this.points.add(
                    world[0],
                    world[1],
                    world[2],
                    0,
                    0,
                    0,
                    t.rgb[i],
                    t.rgb[i + 1],
                    t.rgb[i + 2],
                );
            }
        }
        while (this.points.size > PREVIEW_LIMIT) this.points.coarsen();
    }

    cloud(): { positions: Float32Array; colors: Uint8Array } {
        return this.points.cloud();
    }
}
