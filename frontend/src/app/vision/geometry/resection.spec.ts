import { seededRandom } from '../math/dense';
import { cross3, expSO3, multiply3, normalize3, sub3 } from '../math/mat3';
import { Intrinsics, Pose, cameraCenter, correctPose, project, worldCorrection } from './camera';
import { resect } from './resection';

const K: Intrinsics = { fx: 500, fy: 500, cx: 179.5, cy: 319.5 };

function lookAt(center: number[], target: number[]): Pose {
    const z = normalize3(sub3(target, center));
    const x = normalize3(cross3(z, [0, 1, 0]));
    const y = cross3(z, x);
    const R = new Float64Array([...x, ...y, ...z]);
    const t = new Float64Array(
        [0, 1, 2].map(
            (r) => -(R[r * 3] * center[0] + R[r * 3 + 1] * center[1] + R[r * 3 + 2] * center[2]),
        ),
    );
    return { R, t };
}

describe('resection', () => {
    it('recovers a camera pose that jumped by centimetres and degrees, ignoring outliers', () => {
        const random = seededRandom(4);
        const truth = lookAt([0.3, 0.2, 0.3], [0, 0.05, 0]);
        const points: number[] = [];
        const uv: number[] = [];
        while (points.length < 3 * 200) {
            const p = [(random() - 0.5) * 0.3, random() * 0.12, (random() - 0.5) * 0.3];
            const [u, v, z] = project(K, truth, p);
            if (z <= 0 || u < 0 || v < 0 || u > 359 || v > 639) continue;
            points.push(...p);
            const outlier = points.length / 3 <= 20;
            uv.push(
                u + (outlier ? random() * 80 : (random() - 0.5) * 0.6),
                v + (outlier ? -40 : (random() - 0.5) * 0.6),
            );
        }
        // The phone pose: 4 cm and 1.5° away from the truth.
        const jump = {
            R: expSO3([0, (1.5 * Math.PI) / 180, 0]),
            t: new Float64Array([0.03, 0.01, -0.025]),
        };
        const start = correctPose(truth, jump);
        const result = resect(K, start, Float64Array.from(points), Float64Array.from(uv));
        expect(result).not.toBeNull();
        const error = Math.hypot(...sub3(cameraCenter(result!.pose), cameraCenter(truth)));
        expect(error).toBeLessThan(0.002);
        expect(result!.rms).toBeLessThan(0.6);
        expect(result!.inliers).toBeGreaterThan(170);
    });

    it('turns a re-anchored frame into a correction that moves the rest of its segment', () => {
        const before = lookAt([0.3, 0.2, 0.3], [0, 0, 0]);
        const after = lookAt([0.32, 0.21, 0.28], [0, 0, 0]);
        const G = worldCorrection(before, after);
        const moved = correctPose(before, G);
        expect(Math.hypot(...sub3(cameraCenter(moved), cameraCenter(after)))).toBeLessThan(1e-9);
        // A later frame of the same segment keeps its relative pose to the anchor.
        const later = {
            R: multiply3(expSO3([0, 0.1, 0]), before.R),
            t: new Float64Array([0.01, 0, 0.02]),
        };
        const laterMoved = correctPose(later, G);
        const relative = (a: Pose, b: Pose) => sub3(cameraCenter(a), cameraCenter(b));
        const d0 = Math.hypot(...relative(later, before));
        const d1 = Math.hypot(...relative(laterMoved, moved));
        expect(d1).toBeCloseTo(d0, 9);
    });
});
