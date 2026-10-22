import { quantile } from './dense';

export interface Bounds {
    min: [number, number, number];
    max: [number, number, number];
}

/**
 * Axis-aligned box of xyz points ignoring the extreme `tail` fraction on each side (outliers),
 * widened by `pad` of its size. Subsamples to ~50 k points.
 */
export function robustBounds(positions: Float32Array, tail = 0.002, pad = 0.02): Bounds {
    const min: number[] = [];
    const max: number[] = [];
    const count = positions.length / 3;
    const stride = Math.max(1, Math.floor(count / 50000));
    for (let a = 0; a < 3; a++) {
        const values: number[] = [];
        for (let i = 0; i < count; i += stride) values.push(positions[i * 3 + a]);
        const low = quantile(values, tail);
        const high = quantile(values, 1 - tail);
        const margin = (high - low) * pad;
        min.push(low - margin);
        max.push(high + margin);
    }
    return { min: min as Bounds['min'], max: max as Bounds['max'] };
}

/** Exact axis-aligned extent (size along x, y, z) of xyz points. */
export function extent(positions: Float32Array): [number, number, number] {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let a = 0; a < 3; a++) {
            min[a] = Math.min(min[a], positions[i + a]);
            max[a] = Math.max(max[a], positions[i + a]);
        }
    }
    return [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
}
