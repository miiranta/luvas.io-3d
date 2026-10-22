import { downsampleGray, rgbaToGray } from './gray-image';

describe('gray resampling', () => {
    // A horizontal ramp: any resampling error shows up as a shift of the values.
    const width = 1280;
    const height = 8;
    const ramp = (x: number) => 20 + x * 0.15;

    it('keeps pixel centres aligned for non-integer ratios (as scaleIntrinsics assumes)', () => {
        const rgba = new Uint8Array(width * height * 4);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++)
                rgba.fill(Math.round(ramp(x)), (y * width + x) * 4, (y * width + x) * 4 + 3);
        }
        const gray = rgbaToGray(rgba, width, height, 960, 6);
        const scale = width / 960;
        let worst = 0;
        for (let x = 2; x < 958; x++) {
            const expected = ramp((x + 0.5) * scale - 0.5);
            worst = Math.max(worst, Math.abs(gray.data[2 * 960 + x] - expected));
        }
        // Rounding of the input and output to bytes only.
        expect(worst).toBeLessThanOrEqual(1);
    });

    it('averages exact blocks for integer ratios', () => {
        const data = new Uint8Array(8 * 4).map((_, i) => (i % 8) * 10);
        const half = downsampleGray({ width: 8, height: 4, data }, 4, 2);
        expect(Array.from(half.data.subarray(0, 4))).toEqual([5, 25, 45, 65]);
    });
});
