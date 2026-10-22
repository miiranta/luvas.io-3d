import { seededRandom } from '../math/dense';
import { GrayImage, blurGray, sharpness } from '../image/gray-image';
import { DESCRIPTOR_WORDS, extractFeatures, hamming } from './orb-features';
import { FeatureSet } from './feature-set';
import { extractSift, siftDistance } from './sift-features';

/**
 * Random blobs on a gradient: plenty of corners, no repeated structure. The scene is shifted by
 * (dx, dy) and rotated by `angle` about the image centre; sampled 4× per pixel so that sub-pixel
 * shifts are rendered faithfully.
 */
function texture(width: number, height: number, dx = 0, dy = 0, angle = 0): GrayImage {
    const random = seededRandom(9);
    const blobs = Array.from({ length: 260 }, () => ({
        x: random() * (width + 80) - 40,
        y: random() * (height + 80) - 40,
        r: 3 + random() * 14,
        v: random() * 200 - 100,
    }));
    const data = new Uint8Array(width * height);
    const c = Math.cos(-angle);
    const s = Math.sin(-angle);
    const sample = (px: number, py: number) => {
        // Scene coordinates of the image point: undo the rotation, then the shift.
        const qx = px - width / 2;
        const qy = py - height / 2;
        const x = c * qx - s * qy + width / 2 - dx;
        const y = s * qx + c * qy + height / 2 - dy;
        let value = 110 + (x / width) * 30;
        for (const b of blobs) {
            const d2 = (x - b.x) ** 2 + (y - b.y) ** 2;
            if (d2 < b.r * b.r) value += b.v * (1 - d2 / (b.r * b.r));
        }
        return value;
    };
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let value = 0;
            for (const [ox, oy] of [
                [-0.25, -0.25],
                [0.25, -0.25],
                [-0.25, 0.25],
                [0.25, 0.25],
            ]) {
                value += sample(x + ox, y + oy) / 4;
            }
            data[y * width + x] = Math.max(0, Math.min(255, Math.round(value)));
        }
    }
    return { width, height, data };
}

describe('ORB features', () => {
    it('spreads the requested number of keypoints over the image', () => {
        const features = extractFeatures(texture(320, 240), 600);
        expect(features.count).toBeGreaterThan(400);
        expect(features.count).toBeLessThanOrEqual(600);
        const left = Array.from(features.x).filter((x) => x < 160).length;
        expect(left / features.count).toBeGreaterThan(0.3);
        expect(left / features.count).toBeLessThan(0.7);
    });

    it('gives matching descriptors for the same corner in a shifted image', () => {
        const a = extractFeatures(texture(320, 240), 500);
        const b = extractFeatures(texture(320, 240, 7, -4), 500);
        let found = 0;
        let close = 0;
        for (let i = 0; i < a.count; i++) {
            for (let j = 0; j < b.count; j++) {
                if (Math.abs(b.x[j] - a.x[i] - 7) > 0.6 || Math.abs(b.y[j] - a.y[i] + 4) > 0.6)
                    continue;
                found++;
                const d = hamming(
                    a.descriptors,
                    i * DESCRIPTOR_WORDS,
                    b.descriptors,
                    j * DESCRIPTOR_WORDS,
                );
                if (d < 40) close++;
                break;
            }
        }
        expect(found).toBeGreaterThan(150);
        expect(close / found).toBeGreaterThan(0.9);
    });
});

/** For each feature of a, the feature of b at the expected position (within `radius`), or −1. */
function counterpart(
    a: FeatureSet,
    b: FeatureSet,
    i: number,
    expected: (x: number, y: number) => [number, number],
    radius: number,
): { j: number; error: number } {
    const [ex, ey] = expected(a.x[i], a.y[i]);
    let best = -1;
    let error = radius;
    for (let j = 0; j < b.count; j++) {
        const e = Math.hypot(b.x[j] - ex, b.y[j] - ey);
        if (e < error) {
            error = e;
            best = j;
        }
    }
    return { j: best, error };
}

describe('SIFT features', () => {
    const words = 32;

    it('locates keypoints to sub-pixel accuracy under a fractional shift', () => {
        const a = extractSift(texture(320, 240), 600);
        const b = extractSift(texture(320, 240, 3.4, -1.7), 600);
        expect(a.count).toBeGreaterThan(120);
        const errors: number[] = [];
        for (let i = 0; i < a.count; i++) {
            const { j, error } = counterpart(a, b, i, (x, y) => [x + 3.4, y - 1.7], 1.5);
            if (j >= 0) errors.push(error);
        }
        errors.sort((p, q) => p - q);
        expect(errors.length).toBeGreaterThan(100);
        expect(errors[Math.floor(errors.length / 2)]).toBeLessThan(0.12);
    });

    it('gives close descriptors for the same keypoint after a rotation', () => {
        const angle = (25 * Math.PI) / 180;
        const a = extractSift(texture(320, 240), 600);
        const b = extractSift(texture(320, 240, 0, 0, angle), 600);
        const rotate = (x: number, y: number): [number, number] => {
            const qx = x - 160;
            const qy = y - 120;
            return [
                Math.cos(angle) * qx - Math.sin(angle) * qy + 160,
                Math.sin(angle) * qx + Math.cos(angle) * qy + 120,
            ];
        };
        let found = 0;
        let nearest = 0;
        for (let i = 0; i < a.count; i++) {
            const { j } = counterpart(a, b, i, rotate, 1);
            if (j < 0) continue;
            found++;
            // The true counterpart must be the closest descriptor in b.
            const own = siftDistance(a.descriptors, i * words, b.descriptors, j * words);
            let closer = 0;
            for (let k = 0; k < b.count && closer === 0; k++) {
                if (
                    k !== j &&
                    siftDistance(a.descriptors, i * words, b.descriptors, k * words) < own
                )
                    closer++;
            }
            if (closer === 0) nearest++;
        }
        expect(found).toBeGreaterThan(100);
        expect(nearest / found).toBeGreaterThan(0.85);
    });
});

describe('sharpness', () => {
    it('drops when the image is blurred', () => {
        const image = texture(200, 150);
        const blurred = blurGray(blurGray(blurGray(image)));
        expect(sharpness(blurred)).toBeLessThan(sharpness(image) * 0.5);
    });
});
