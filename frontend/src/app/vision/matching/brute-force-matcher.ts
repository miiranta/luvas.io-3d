import { FeatureSet } from '../features/feature-set';
import { MatchList } from './match-list';

/**
 * Unguided matching (every descriptor against every other, Lowe ratio + mutual best). Used as
 * the fallback when guided matching fails because the phone pose itself is wrong, e.g. after a
 * tracking jump; follow it with `fundamentalInliers`.
 */
export function bruteForceMatch(a: FeatureSet, b: FeatureSet, ratio = 0.8): MatchList {
    const { distance, words, maxDistance } = a.metric;
    const bestOfA = new Int32Array(a.count).fill(-1);
    const bestOfB = new Int32Array(b.count).fill(-1);
    const distanceOfB = new Int32Array(b.count).fill(1 << 30);
    for (let i = 0; i < a.count; i++) {
        let best = 1 << 30;
        let second = 1 << 30;
        let index = -1;
        for (let j = 0; j < b.count; j++) {
            if (Math.abs(a.level[i] - b.level[j]) > 1) continue;
            const d = distance(a.descriptors, i * words, b.descriptors, j * words);
            if (d < best) {
                second = best;
                best = d;
                index = j;
            } else if (d < second) {
                second = d;
            }
            if (d < distanceOfB[j]) {
                distanceOfB[j] = d;
                bestOfB[j] = i;
            }
        }
        if (index >= 0 && best <= maxDistance && best <= ratio * second) bestOfA[i] = index;
    }
    const pairs: number[] = [];
    for (let i = 0; i < a.count; i++) {
        const j = bestOfA[i];
        if (j >= 0 && bestOfB[j] === i) pairs.push(i, j);
    }
    return Int32Array.from(pairs);
}
