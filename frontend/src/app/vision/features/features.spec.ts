import { seededRandom } from '../math/dense';
import { GrayImage, blurGray, sharpness } from '../image/gray-image';
import { DESCRIPTOR_WORDS, extractFeatures, hamming } from './orb-features';

/** Random blobs on a gradient: plenty of corners, no repeated structure. */
function texture(width: number, height: number, dx = 0, dy = 0): GrayImage {
    const random = seededRandom(9);
    const blobs = Array.from({ length: 260 }, () => ({
        x: random() * (width + 80) - 40,
        y: random() * (height + 80) - 40,
        r: 3 + random() * 14,
        v: random() * 200 - 100,
    }));
    const data = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let value = 110 + (x / width) * 30;
            for (const b of blobs) {
                const d2 = (x - dx - b.x) ** 2 + (y - dy - b.y) ** 2;
                if (d2 < b.r * b.r) value += b.v * (1 - d2 / (b.r * b.r));
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

describe('sharpness', () => {
    it('drops when the image is blurred', () => {
        const image = texture(200, 150);
        const blurred = blurGray(blurGray(blurGray(image)));
        expect(sharpness(blurred)).toBeLessThan(sharpness(image) * 0.5);
    });
});
