import { Mat4, invert, lookAt, multiply, perspective } from './mat4';

const FOV = (50 * Math.PI) / 180;

/** Turntable camera around a target with +y up (ARCore's world is gravity-aligned). */
export class OrbitCamera {
    target = [0, 0, 0];
    distance = 1;
    yaw = 0.6;
    pitch = 0.45;
    private radius = 1;

    get fov(): number {
        return FOV;
    }

    eye(): number[] {
        const c = Math.cos(this.pitch);
        return [
            this.target[0] + this.distance * c * Math.sin(this.yaw),
            this.target[1] + this.distance * Math.sin(this.pitch),
            this.target[2] + this.distance * c * Math.cos(this.yaw),
        ];
    }

    viewProjection(aspect: number): Mat4 {
        const near = Math.max(0.002, this.distance * 0.01);
        const far = this.distance + this.radius * 4 + 1;
        return multiply(
            perspective(FOV, aspect, near, far),
            lookAt(this.eye(), this.target, [0, 1, 0]),
        );
    }

    /** Frames points robustly (2nd–98th percentile box). */
    frame(positions: Float32Array): void {
        const count = positions.length / 3;
        if (count === 0) return;
        const stride = Math.max(1, Math.floor(count / 20000));
        const axes: number[][] = [[], [], []];
        for (let i = 0; i < count; i += stride) {
            for (let a = 0; a < 3; a++) axes[a].push(positions[i * 3 + a]);
        }
        const low: number[] = [];
        const high: number[] = [];
        for (const values of axes) {
            values.sort((a, b) => a - b);
            low.push(values[Math.floor(values.length * 0.02)]);
            high.push(values[Math.min(values.length - 1, Math.floor(values.length * 0.98))]);
        }
        this.target = [0, 1, 2].map((a) => (low[a] + high[a]) / 2);
        this.radius = Math.max(
            0.05,
            Math.hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]) / 2,
        );
        this.distance = (this.radius / Math.sin(FOV / 2)) * 1.05;
    }

    rotate(dx: number, dy: number): void {
        this.yaw -= dx * 0.008;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + dy * 0.008));
    }

    zoom(factor: number): void {
        this.distance = Math.max(
            this.radius * 0.05,
            Math.min(this.radius * 20, this.distance * factor),
        );
    }

    /** Moves the target in the view plane by a screen offset (pixels of a `height`-px view). */
    pan(dx: number, dy: number, height: number): void {
        const scale = (2 * this.distance * Math.tan(FOV / 2)) / height;
        const right = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
        const up = [
            -Math.sin(this.pitch) * Math.sin(this.yaw),
            Math.cos(this.pitch),
            -Math.sin(this.pitch) * Math.cos(this.yaw),
        ];
        for (let a = 0; a < 3; a++) this.target[a] += (-dx * right[a] + dy * up[a]) * scale;
    }

    /** World ray through a pixel (normalised device coordinates in [−1, 1]). */
    ray(
        ndcX: number,
        ndcY: number,
        aspect: number,
    ): { origin: number[]; direction: number[] } | null {
        const inverse = invert(this.viewProjection(aspect));
        if (!inverse) return null;
        const unproject = (z: number) => {
            const v = [ndcX, ndcY, z, 1];
            const out = [0, 0, 0, 0];
            for (let r = 0; r < 4; r++) {
                out[r] =
                    inverse[r] * v[0] +
                    inverse[4 + r] * v[1] +
                    inverse[8 + r] * v[2] +
                    inverse[12 + r] * v[3];
            }
            return [out[0] / out[3], out[1] / out[3], out[2] / out[3]];
        };
        const origin = unproject(-1);
        const end = unproject(1);
        const d = [end[0] - origin[0], end[1] - origin[1], end[2] - origin[2]];
        const length = Math.hypot(d[0], d[1], d[2]);
        return { origin, direction: [d[0] / length, d[1] / length, d[2] / length] };
    }
}

/** Nearest ray–triangle hit (Möller–Trumbore), brute force over the mesh. */
export function pickMesh(
    positions: Float32Array,
    indices: Uint32Array,
    origin: number[],
    direction: number[],
): number[] | null {
    let best = Infinity;
    const [ox, oy, oz] = origin;
    const [dx, dy, dz] = direction;
    for (let f = 0; f < indices.length; f += 3) {
        const a = indices[f] * 3;
        const b = indices[f + 1] * 3;
        const c = indices[f + 2] * 3;
        const e1x = positions[b] - positions[a];
        const e1y = positions[b + 1] - positions[a + 1];
        const e1z = positions[b + 2] - positions[a + 2];
        const e2x = positions[c] - positions[a];
        const e2y = positions[c + 1] - positions[a + 1];
        const e2z = positions[c + 2] - positions[a + 2];
        const px = dy * e2z - dz * e2y;
        const py = dz * e2x - dx * e2z;
        const pz = dx * e2y - dy * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-12) continue;
        const inv = 1 / det;
        const tx = ox - positions[a];
        const ty = oy - positions[a + 1];
        const tz = oz - positions[a + 2];
        const u = (tx * px + ty * py + tz * pz) * inv;
        if (u < 0 || u > 1) continue;
        const qx = ty * e1z - tz * e1y;
        const qy = tz * e1x - tx * e1z;
        const qz = tx * e1y - ty * e1x;
        const v = (dx * qx + dy * qy + dz * qz) * inv;
        if (v < 0 || u + v > 1) continue;
        const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
        if (t > 1e-6 && t < best) best = t;
    }
    return Number.isFinite(best) ? [ox + dx * best, oy + dy * best, oz + dz * best] : null;
}
