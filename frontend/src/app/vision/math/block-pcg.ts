/** A diagonal block of the system: rows/columns [offset, offset + size). */
export interface Block {
    offset: number;
    size: number;
}

/**
 * Conjugate gradients on a dense symmetric positive-definite n×n system (row-major),
 * preconditioned by the inverses of the given diagonal blocks (block Jacobi).
 */
export function blockPcg(
    S: Float64Array,
    b: Float64Array,
    n: number,
    blocks: Block[],
    maxIterations = 300,
    tolerance = 1e-10,
): Float64Array {
    const inverses = blocks.map(({ offset, size }) => {
        const block = new Float64Array(size * size);
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) block[r * size + c] = S[(offset + r) * n + offset + c];
        }
        return invertSmall(block, size);
    });
    const applyM = (v: Float64Array, out: Float64Array) => {
        blocks.forEach(({ offset, size }, k) => {
            const inverse = inverses[k];
            for (let r = 0; r < size; r++) {
                let sum = 0;
                for (let c = 0; c < size; c++) sum += inverse[r * size + c] * v[offset + c];
                out[offset + r] = sum;
            }
        });
    };
    const x = new Float64Array(n);
    const r = new Float64Array(b);
    const z = new Float64Array(n);
    applyM(r, z);
    const p = new Float64Array(z);
    const q = new Float64Array(n);
    let rz = dot(r, z);
    const target = tolerance * dot(b, b);
    for (let iteration = 0; iteration < maxIterations && dot(r, r) > target; iteration++) {
        for (let i = 0; i < n; i++) {
            let sum = 0;
            const row = i * n;
            for (let j = 0; j < n; j++) sum += S[row + j] * p[j];
            q[i] = sum;
        }
        const alpha = rz / (dot(p, q) || 1e-300);
        for (let i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        applyM(r, z);
        const next = dot(r, z);
        const beta = next / (rz || 1e-300);
        rz = next;
        for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    return x;
}

/** Gauss–Jordan inverse with partial pivoting; falls back to the inverse diagonal if singular. */
export function invertSmall(block: Float64Array, size: number): Float64Array {
    const a = new Float64Array(block);
    const inv = new Float64Array(size * size);
    for (let i = 0; i < size; i++) inv[i * size + i] = 1;
    for (let col = 0; col < size; col++) {
        let pivot = col;
        for (let row = col + 1; row < size; row++) {
            if (Math.abs(a[row * size + col]) > Math.abs(a[pivot * size + col])) pivot = row;
        }
        const value = a[pivot * size + col];
        if (Math.abs(value) < 1e-30) {
            const diagonal = new Float64Array(size * size);
            for (let i = 0; i < size; i++) {
                diagonal[i * size + i] = 1 / Math.max(1e-30, block[i * size + i]);
            }
            return diagonal;
        }
        for (let k = 0; k < size; k++) {
            [a[col * size + k], a[pivot * size + k]] = [a[pivot * size + k], a[col * size + k]];
            [inv[col * size + k], inv[pivot * size + k]] = [
                inv[pivot * size + k],
                inv[col * size + k],
            ];
        }
        for (let k = 0; k < size; k++) {
            a[col * size + k] /= value;
            inv[col * size + k] /= value;
        }
        for (let row = 0; row < size; row++) {
            if (row === col) continue;
            const factor = a[row * size + col];
            if (factor === 0) continue;
            for (let k = 0; k < size; k++) {
                a[row * size + k] -= factor * a[col * size + k];
                inv[row * size + k] -= factor * inv[col * size + k];
            }
        }
    }
    return inv;
}

function dot(a: Float64Array, b: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}
