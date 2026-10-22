import { Pose } from '../geometry/camera';
import { selectKeyframes } from './keyframe-selection';

/** Camera at x on a line, looking down +z (world → camera: identity rotation). */
function at(x: number): Pose {
    return { R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: new Float64Array([-x, 0, 0]) };
}

describe('keyframe selection', () => {
    it('keeps everything within budget', () => {
        const frames = [0, 0.1, 0.2].map((x) => ({ pose: at(x), depthHint: 1, sharpness: 1 }));
        expect(selectKeyframes(frames, 5)).toEqual([0, 1, 2]);
    });

    it('drops redundant and blurry views first and keeps the spread', () => {
        // Evenly spaced views, plus a near-duplicate of view 3 and a blurry neighbour of view 6.
        const xs = [0, 0.1, 0.2, 0.3, 0.301, 0.4, 0.5, 0.505, 0.6, 0.7];
        const sharpness = xs.map((_, i) => (i === 7 ? 0.3 : 1));
        const frames = xs.map((x, i) => ({ pose: at(x), depthHint: 1, sharpness: sharpness[i] }));
        const kept = selectKeyframes(frames, 8);
        expect(kept.length).toBe(8);
        expect(kept).toEqual([...kept].sort((a, b) => a - b));
        expect(kept).toContain(0);
        expect(kept).toContain(9);
        expect(kept.includes(3) && kept.includes(4)).toBe(false);
        expect(kept).not.toContain(7);
    });
});
