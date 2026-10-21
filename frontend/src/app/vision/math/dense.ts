/**
 * Solves A·x = b for a symmetric positive-definite n×n matrix (row-major) by Cholesky
 * factorisation. A is overwritten with its factor; returns null when A is not SPD.
 */
export function solveCholesky(a: Float64Array, b: Float64Array, n: number): Float64Array | null {
    for (let j = 0; j < n; j++) {
        const rowJ = j * n;
        let diagonal = a[rowJ + j];
        for (let k = 0; k < j; k++) diagonal -= a[rowJ + k] * a[rowJ + k];
        if (!(diagonal > 1e-300)) return null;
        const pivot = Math.sqrt(diagonal);
        a[rowJ + j] = pivot;
        const inverse = 1 / pivot;
        for (let i = j + 1; i < n; i++) {
            const rowI = i * n;
            let sum = a[rowI + j];
            for (let k = 0; k < j; k++) sum -= a[rowI + k] * a[rowJ + k];
            a[rowI + j] = sum * inverse;
        }
    }
    const x = new Float64Array(b);
    for (let i = 0; i < n; i++) {
        let sum = x[i];
        for (let k = 0; k < i; k++) sum -= a[i * n + k] * x[k];
        x[i] = sum / a[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
        let sum = x[i];
        for (let k = i + 1; k < n; k++) sum -= a[k * n + i] * x[k];
        x[i] = sum / a[i * n + i];
    }
    return x;
}

/**
 * Eigen-decomposition of a symmetric n×n matrix by cyclic Jacobi rotations.
 * Returns eigenvalues (ascending) and the matching eigenvectors as rows.
 */
export function symmetricEigen(
    matrix: ArrayLike<number>,
    n: number,
): { values: Float64Array; vectors: Float64Array } {
    const a = Float64Array.from(matrix);
    const v = new Float64Array(n * n);
    for (let i = 0; i < n; i++) v[i * n + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
        let off = 0;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
        if (off < 1e-24) break;
        for (let p = 0; p < n; p++) {
            for (let q = p + 1; q < n; q++) {
                const apq = a[p * n + q];
                if (Math.abs(apq) < 1e-300) continue;
                const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
                const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                const c = 1 / Math.sqrt(t * t + 1);
                const s = t * c;
                for (let k = 0; k < n; k++) {
                    const akp = a[k * n + p];
                    const akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (let k = 0; k < n; k++) {
                    const apk = a[p * n + k];
                    const aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (let k = 0; k < n; k++) {
                    const vpk = v[p * n + k];
                    const vqk = v[q * n + k];
                    v[p * n + k] = c * vpk - s * vqk;
                    v[q * n + k] = s * vpk + c * vqk;
                }
            }
        }
    }
    const order = Array.from({ length: n }, (_, i) => i).sort(
        (i, j) => a[i * n + i] - a[j * n + j],
    );
    const values = new Float64Array(n);
    const vectors = new Float64Array(n * n);
    order.forEach((source, target) => {
        values[target] = a[source * n + source];
        vectors.set(v.subarray(source * n, source * n + n), target * n);
    });
    return { values, vectors };
}

/** Deterministic PRNG (mulberry32) so captures and tests are reproducible. */
export function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function median(values: ArrayLike<number>): number {
    return quantile(values, 0.5);
}

export function quantile(values: ArrayLike<number>, q: number): number {
    if (values.length === 0) return NaN;
    const sorted = Float64Array.from(values).sort();
    const position = Math.min(sorted.length - 1, Math.max(0, q * (sorted.length - 1)));
    const low = Math.floor(position);
    const high = Math.min(sorted.length - 1, low + 1);
    return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
