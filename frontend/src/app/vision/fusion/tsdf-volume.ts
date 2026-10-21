import { Intrinsics, Pose, project } from '../geometry/camera';
import { BufferUsage, readBuffer, storageBuffer } from '../gpu/gpu-device';
import { PACKING_WGSL } from '../mvs/patchmatch.wgsl';

export const BRICK = 8;
const BRICK_VOXELS = BRICK * BRICK * BRICK;
const MAX_WEIGHT = 96;
const PRIOR_WEIGHT = 0.12;
/** Bricks per submit (≈ 1 M voxel updates): short enough for phone GPUs. */
const CHUNK_BRICKS = 2048;

/** Sparse set of 8³ voxel bricks around the observed surface, on a bounded grid. */
export class BrickGrid {
    readonly dims: [number, number, number];
    private readonly marks: Uint8Array;

    constructor(
        readonly origin: [number, number, number],
        readonly voxel: number,
        extent: [number, number, number],
    ) {
        const side = voxel * BRICK;
        this.dims = [
            Math.max(1, Math.ceil(extent[0] / side)),
            Math.max(1, Math.ceil(extent[1] / side)),
            Math.max(1, Math.ceil(extent[2] / side)),
        ];
        this.marks = new Uint8Array(this.dims[0] * this.dims[1] * this.dims[2]);
    }

    get cells(): number {
        return this.marks.length;
    }

    /** Marks every brick touched by the cube of half-size `radius` around a point. */
    mark(x: number, y: number, z: number, radius: number): void {
        const side = this.voxel * BRICK;
        const [ox, oy, oz] = this.origin;
        const [dx, dy, dz] = this.dims;
        const x0 = Math.max(0, Math.floor((x - radius - ox) / side));
        const x1 = Math.min(dx - 1, Math.floor((x + radius - ox) / side));
        const y0 = Math.max(0, Math.floor((y - radius - oy) / side));
        const y1 = Math.min(dy - 1, Math.floor((y + radius - oy) / side));
        const z0 = Math.max(0, Math.floor((z - radius - oz) / side));
        const z1 = Math.min(dz - 1, Math.floor((z + radius - oz) / side));
        for (let bz = z0; bz <= z1; bz++) {
            for (let by = y0; by <= y1; by++) {
                for (let bx = x0; bx <= x1; bx++) this.marks[(bz * dy + by) * dx + bx] = 1;
            }
        }
    }

    /** Pool layout: brick coordinates (xyz, pad) and the grid → pool lookup. */
    build(): { coords: Int32Array; lookup: Int32Array; count: number } {
        const lookup = new Int32Array(this.marks.length).fill(-1);
        const coords: number[] = [];
        const [dx, dy] = this.dims;
        for (let i = 0; i < this.marks.length; i++) {
            if (!this.marks[i]) continue;
            lookup[i] = coords.length / 4;
            coords.push(i % dx, Math.floor(i / dx) % dy, Math.floor(i / (dx * dy)), 0);
        }
        return { coords: Int32Array.from(coords), lookup, count: coords.length / 4 };
    }
}

export interface IntegrationView {
    intrinsics: Intrinsics;
    pose: Pose;
    /** Farthest depth worth integrating for this view (m). */
    maxDepth: number;
    /** Phone depth at work resolution (0 = none), used where stereo left holes. */
    prior: Float32Array | null;
}

const TSDF_WGSL = /* wgsl */ `
struct Integrate {
    k: vec4<f32>,
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    t: vec4<f32>,
    origin: vec4<f32>,
    dims: vec4<u32>,
    band: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Integrate;
@group(0) @binding(1) var<storage, read> visible: array<u32>;
@group(0) @binding(2) var<storage, read> coords: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read_write> tsdf: array<f32>;
@group(0) @binding(4) var<storage, read_write> weight: array<f32>;
@group(0) @binding(5) var<storage, read> depth: array<u32>;
@group(0) @binding(6) var<storage, read> prior: array<f32>;

${PACKING_WGSL}

@compute @workgroup_size(64)
fn integrate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let slot = wg.x + wg.y * 65535u;
    if (slot >= u.dims.z) {
        return;
    }
    let brick = visible[slot];
    let base = coords[brick].xyz * 8;
    let r = transpose(mat3x3<f32>(u.r0.xyz, u.r1.xyz, u.r2.xyz));
    let mu = u.band.x;
    for (var k = 0u; k < 8u; k++) {
        let local = li + k * 64u;
        let v = vec3<i32>(i32(local % 8u), i32((local / 8u) % 8u), i32(local / 64u));
        let world = u.origin.xyz + (vec3<f32>(base + v) + 0.5) * u.origin.w;
        let cam = r * world + u.t.xyz;
        if (cam.z <= 0.01) {
            continue;
        }
        let pixel = vec2<i32>(round(vec2<f32>(u.k.x * cam.x / cam.z + u.k.z, u.k.y * cam.y / cam.z + u.k.w)));
        if (any(pixel < vec2<i32>(0)) || pixel.x >= i32(u.dims.x) || pixel.y >= i32(u.dims.y)) {
            continue;
        }
        let i = u32(pixel.y) * u.dims.x + u32(pixel.x);
        let packed = depth[i];
        var d = unpack_depth(packed);
        var w = 0.0;
        if (d > 0.0) {
            w = max(0.2, abs(dot(unpack_normal(packed), normalize(cam))));
        } else {
            d = prior[i];
            w = u.band.y;
        }
        if (!(d > 0.0)) {
            continue;
        }
        let sdf = d - cam.z;
        if (sdf < -mu) {
            continue;
        }
        let observed = min(1.0, sdf / mu);
        let index = brick * 512u + local;
        let w0 = weight[index];
        tsdf[index] = (tsdf[index] * w0 + observed * w) / (w0 + w);
        weight[index] = min(w0 + w, u.band.z);
    }
}
`;

/** Truncated signed distance volume on the GPU, integrated one depth map at a time. */
export class TsdfVolume {
    readonly truncation: number;
    private readonly tsdf: GPUBuffer;
    private readonly weight: GPUBuffer;
    private readonly visible: GPUBuffer;
    private readonly prior: GPUBuffer;
    private readonly coords: GPUBuffer;
    private readonly uniforms: GPUBuffer;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroup: GPUBindGroup;
    private readonly zeros: Float32Array;
    private readonly centers: Float64Array;

    constructor(
        private readonly device: GPUDevice,
        readonly grid: BrickGrid,
        readonly layout: { coords: Int32Array; lookup: Int32Array; count: number },
        depth: GPUBuffer,
        readonly width: number,
        readonly height: number,
    ) {
        this.truncation = grid.voxel * 4;
        const voxels = layout.count * BRICK_VOXELS;
        this.tsdf = storageBuffer(device, voxels * 4, 'tsdf');
        this.weight = storageBuffer(device, voxels * 4, 'tsdf weight');
        this.visible = storageBuffer(device, layout.count * 4, 'visible bricks');
        this.prior = storageBuffer(device, width * height * 4, 'tsdf prior');
        this.zeros = new Float32Array(width * height);
        this.coords = storageBuffer(device, layout.coords.byteLength, 'brick coords');
        device.queue.writeBuffer(this.coords, 0, layout.coords as Int32Array<ArrayBuffer>);
        device.queue.writeBuffer(this.tsdf, 0, new Float32Array(voxels).fill(1));
        this.uniforms = device.createBuffer({
            size: 128,
            usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        });
        const module = device.createShaderModule({ label: 'tsdf', code: TSDF_WGSL });
        this.pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'integrate' },
        });
        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniforms } },
                { binding: 1, resource: { buffer: this.visible } },
                { binding: 2, resource: { buffer: this.coords } },
                { binding: 3, resource: { buffer: this.tsdf } },
                { binding: 4, resource: { buffer: this.weight } },
                { binding: 5, resource: { buffer: depth } },
                { binding: 6, resource: { buffer: this.prior } },
            ],
        });
        const side = grid.voxel * BRICK;
        this.centers = new Float64Array(layout.count * 3);
        for (let b = 0; b < layout.count; b++) {
            for (let a = 0; a < 3; a++) {
                this.centers[b * 3 + a] = grid.origin[a] + (layout.coords[b * 4 + a] + 0.5) * side;
            }
        }
    }

    /** Bricks whose centre projects into the view (with a margin of the brick's radius). */
    private visibleBricks(view: IntegrationView): Uint32Array {
        const out: number[] = [];
        const k = view.intrinsics;
        const radius = this.grid.voxel * BRICK * 0.87 + this.truncation;
        for (let b = 0; b < this.layout.count; b++) {
            const [u, v, z] = project(k, view.pose, this.centers.subarray(b * 3, b * 3 + 3));
            if (z < 0.02 - radius || z > view.maxDepth + radius) continue;
            const margin = (k.fx * radius) / Math.max(0.02, z);
            if (u < -margin || v < -margin || u > this.width + margin || v > this.height + margin) {
                continue;
            }
            out.push(b);
        }
        return Uint32Array.from(out);
    }

    /**
     * Integrates the depth map currently held in the bound depth buffer. The visible bricks go
     * in chunks, one short submit each, so a phone GPU is never monopolised for long.
     */
    async integrate(view: IntegrationView): Promise<void> {
        const visible = this.visibleBricks(view);
        if (visible.length === 0) return;
        this.device.queue.writeBuffer(this.prior, 0, view.prior ?? this.zeros);
        const buffer = new ArrayBuffer(128);
        const f = new Float32Array(buffer);
        const u = new Uint32Array(buffer);
        const k = view.intrinsics;
        const { R, t } = view.pose;
        f.set([k.fx, k.fy, k.cx, k.cy], 0);
        f.set([R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0], 4);
        f.set([t[0], t[1], t[2], 0], 16);
        f.set([...this.grid.origin, this.grid.voxel], 20);
        f.set([this.truncation, PRIOR_WEIGHT, MAX_WEIGHT, 0], 28);
        for (let start = 0; start < visible.length; start += CHUNK_BRICKS) {
            const chunk = visible.subarray(start, start + CHUNK_BRICKS);
            this.device.queue.writeBuffer(this.visible, 0, chunk);
            u.set([this.width, this.height, chunk.length, 0], 24);
            this.device.queue.writeBuffer(this.uniforms, 0, buffer);
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(chunk.length);
            pass.end();
            this.device.queue.submit([encoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
        }
    }

    async read(): Promise<{ tsdf: Float32Array; weight: Float32Array }> {
        const bytes = this.layout.count * BRICK_VOXELS * 4;
        const [tsdf, weight] = await Promise.all([
            readBuffer(this.device, this.tsdf, bytes),
            readBuffer(this.device, this.weight, bytes),
        ]);
        return { tsdf: new Float32Array(tsdf), weight: new Float32Array(weight) };
    }

    destroy(): void {
        for (const buffer of [
            this.tsdf,
            this.weight,
            this.visible,
            this.prior,
            this.coords,
            this.uniforms,
        ]) {
            buffer.destroy();
        }
    }
}
