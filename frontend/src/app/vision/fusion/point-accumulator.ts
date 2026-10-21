export interface PointCloud {
    positions: Float32Array;
    normals: Float32Array;
    colors: Uint8Array;
    /** Number of depth samples merged into each point. */
    support: Uint16Array;
}

const EMPTY = 0x7fffffff;

/**
 * Merges oriented, coloured samples on a voxel grid (open-addressing hash on the integer cell),
 * averaging position, normal and colour per cell. Memory is bounded by the number of occupied
 * cells; `coarsen` grows the cell size when that number exceeds the budget.
 */
export class PointAccumulator {
    private capacity = 1 << 16;
    private keys = new Int32Array(this.capacity * 3).fill(EMPTY);
    private sums = new Float32Array(this.capacity * 9);
    private counts = new Uint32Array(this.capacity);
    private used = 0;

    constructor(public cell: number) {}

    get size(): number {
        return this.used;
    }

    add(
        x: number,
        y: number,
        z: number,
        nx: number,
        ny: number,
        nz: number,
        r: number,
        g: number,
        b: number,
    ): void {
        const slot = this.slotOf(
            Math.floor(x / this.cell),
            Math.floor(y / this.cell),
            Math.floor(z / this.cell),
        );
        const o = slot * 9;
        this.sums[o] += x;
        this.sums[o + 1] += y;
        this.sums[o + 2] += z;
        this.sums[o + 3] += nx;
        this.sums[o + 4] += ny;
        this.sums[o + 5] += nz;
        this.sums[o + 6] += r;
        this.sums[o + 7] += g;
        this.sums[o + 8] += b;
        this.counts[slot]++;
    }

    private slotOf(ix: number, iy: number, iz: number): number {
        if (this.used * 10 > this.capacity * 6) this.grow();
        const mask = this.capacity - 1;
        let slot = hash(ix, iy, iz) & mask;
        for (;;) {
            const k = slot * 3;
            if (this.keys[k] === EMPTY) {
                this.keys[k] = ix;
                this.keys[k + 1] = iy;
                this.keys[k + 2] = iz;
                this.used++;
                return slot;
            }
            if (this.keys[k] === ix && this.keys[k + 1] === iy && this.keys[k + 2] === iz)
                return slot;
            slot = (slot + 1) & mask;
        }
    }

    private grow(): void {
        this.rebuild(this.capacity * 2, this.cell);
    }

    /**
     * Grows the cell size (1.4× by default), re-binning every cell by its mean position and
     * merging the sums of cells that fall together.
     */
    coarsen(factor = 1.4): void {
        this.rebuild(this.capacity, this.cell * factor);
    }

    private rebuild(capacity: number, cell: number): void {
        const keys = this.keys;
        const sums = this.sums;
        const counts = this.counts;
        const old = this.capacity;
        this.capacity = capacity;
        this.cell = cell;
        this.keys = new Int32Array(capacity * 3).fill(EMPTY);
        this.sums = new Float32Array(capacity * 9);
        this.counts = new Uint32Array(capacity);
        this.used = 0;
        for (let s = 0; s < old; s++) {
            if (keys[s * 3] === EMPTY) continue;
            const n = counts[s];
            const slot = this.slotOf(
                Math.floor(sums[s * 9] / n / cell),
                Math.floor(sums[s * 9 + 1] / n / cell),
                Math.floor(sums[s * 9 + 2] / n / cell),
            );
            for (let i = 0; i < 9; i++) this.sums[slot * 9 + i] += sums[s * 9 + i];
            this.counts[slot] += n;
        }
    }

    /** Averaged points with at least `minSupport` samples. */
    cloud(minSupport = 1): PointCloud {
        let total = 0;
        for (let s = 0; s < this.capacity; s++) {
            if (this.keys[s * 3] !== EMPTY && this.counts[s] >= minSupport) total++;
        }
        const positions = new Float32Array(total * 3);
        const normals = new Float32Array(total * 3);
        const colors = new Uint8Array(total * 3);
        const support = new Uint16Array(total);
        let p = 0;
        for (let s = 0; s < this.capacity; s++) {
            if (this.keys[s * 3] === EMPTY || this.counts[s] < minSupport) continue;
            const n = this.counts[s];
            const o = s * 9;
            positions[p * 3] = this.sums[o] / n;
            positions[p * 3 + 1] = this.sums[o + 1] / n;
            positions[p * 3 + 2] = this.sums[o + 2] / n;
            const length = Math.hypot(this.sums[o + 3], this.sums[o + 4], this.sums[o + 5]) || 1;
            normals[p * 3] = this.sums[o + 3] / length;
            normals[p * 3 + 1] = this.sums[o + 4] / length;
            normals[p * 3 + 2] = this.sums[o + 5] / length;
            colors[p * 3] = this.sums[o + 6] / n;
            colors[p * 3 + 1] = this.sums[o + 7] / n;
            colors[p * 3 + 2] = this.sums[o + 8] / n;
            support[p] = Math.min(65535, n);
            p++;
        }
        return { positions, normals, colors, support };
    }
}

function hash(x: number, y: number, z: number): number {
    return (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
}
