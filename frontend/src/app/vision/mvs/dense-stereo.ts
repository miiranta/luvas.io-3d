import { Intrinsics, Pose, relativePose, scaleIntrinsics } from '../geometry/camera';
import {
    BufferUsage,
    ShaderStage,
    TextureUsage,
    readBuffer,
    storageBuffer,
} from '../gpu/gpu-device';
import { PATCHMATCH_WGSL } from './patchmatch.wgsl';

export interface StereoView {
    intrinsics: Intrinsics;
    pose: Pose;
}

export interface SolveOptions {
    sources: number[];
    minDepth: number;
    maxDepth: number;
    /** Depth guess per pixel (0 = none), e.g. from the phone's depth sensor. */
    prior: Float32Array | null;
    /** Second pass: adds the geometric-consistency term and starts from the first pass. */
    geometric: boolean;
    iterations: number;
    /** Start from the coarse level's planes instead of random ones (coarse-to-fine). */
    fromCoarse?: boolean;
    /** Store the packed depth map (default true; a coarse level only keeps its planes). */
    pack?: boolean;
}

/** A half-resolution solve whose planes seed this level (coarse-to-fine). */
export interface CoarseLevel {
    hypotheses: GPUBuffer;
    width: number;
    height: number;
}

export interface StereoOptions {
    coarse?: CoarseLevel;
    /** Keep a depth map per view (needed for the geometric pass, filter and fusion). */
    depthMaps?: boolean;
}

const MAX_SOURCES = 8;
const PARAMS_BYTES = 752;
const GEOMETRIC_WEIGHT = 0.2;
/**
 * GPU time budget per submit. Phone GPUs are shared with the system compositor and a long
 * submit freezes the screen or trips the driver watchdog (which can reset the whole phone UI),
 * so the work is sliced into row bands whose height adapts to stay near this budget.
 */
const SUBMIT_MS = 20;
const MIN_BAND = 8;
const FIRST_BAND = 32;

/**
 * GPU PatchMatch for every keyframe. All gray images live in one texture array and all packed
 * depth maps (u32 per pixel) in one storage buffer, so the geometric pass and the consistency
 * filter can read any neighbour without copies.
 */
export class DenseStereo {
    private readonly images: GPUTexture;
    private readonly depths: GPUBuffer;
    private readonly hyp: GPUBuffer;
    private readonly cost: GPUBuffer;
    private readonly prior: GPUBuffer;
    readonly filtered: GPUBuffer;
    private readonly uniforms: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;
    private readonly pipelines: Record<
        'init' | 'initCoarse' | 'propagate' | 'pack' | 'consistency',
        GPUComputePipeline
    >;
    private readonly zeros: Float32Array;
    private readonly placeholder: GPUBuffer | null;
    private readonly coarse: CoarseLevel | null;
    /** Rows per dispatch, adapted to the measured GPU time. */
    private bandRows = FIRST_BAND;

    constructor(
        private readonly device: GPUDevice,
        readonly width: number,
        readonly height: number,
        private readonly views: StereoView[],
        options: StereoOptions = {},
    ) {
        const pixels = width * height;
        this.coarse = options.coarse ?? null;
        this.placeholder = this.coarse ? null : storageBuffer(device, 16, 'no coarse level');
        this.zeros = new Float32Array(pixels);
        this.images = device.createTexture({
            label: 'stereo images',
            size: { width, height, depthOrArrayLayers: Math.max(1, views.length) },
            format: 'r8unorm',
            usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
        });
        const maps = options.depthMaps === false ? 1 : views.length;
        this.depths = storageBuffer(device, maps * pixels * 4, 'depth maps');
        this.hyp = storageBuffer(device, pixels * 16, 'hypotheses');
        this.cost = storageBuffer(device, pixels * 4, 'costs');
        this.prior = storageBuffer(device, pixels * 4, 'prior');
        this.filtered = storageBuffer(device, pixels * 4, 'filtered');
        this.uniforms = device.createBuffer({
            label: 'stereo params',
            size: PARAMS_BYTES,
            usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        });

        const storage = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({
            binding,
            visibility: ShaderStage.COMPUTE,
            buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
        });
        const layout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: ShaderStage.COMPUTE,
                    texture: { sampleType: 'float', viewDimension: '2d-array' },
                },
                { binding: 1, visibility: ShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                {
                    binding: 2,
                    visibility: ShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: PARAMS_BYTES },
                },
                storage(3),
                storage(4),
                storage(5, true),
                storage(6),
                storage(7),
                storage(8, true),
            ],
        });
        this.bindGroup = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: this.images.createView({ dimension: '2d-array' }) },
                {
                    binding: 1,
                    resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
                },
                { binding: 2, resource: { buffer: this.uniforms } },
                { binding: 3, resource: { buffer: this.hyp } },
                { binding: 4, resource: { buffer: this.cost } },
                { binding: 5, resource: { buffer: this.prior } },
                { binding: 6, resource: { buffer: this.depths } },
                { binding: 7, resource: { buffer: this.filtered } },
                {
                    binding: 8,
                    resource: {
                        buffer: this.coarse?.hypotheses ?? (this.placeholder as GPUBuffer),
                    },
                },
            ],
        });
        const module = device.createShaderModule({ label: 'patchmatch', code: PATCHMATCH_WGSL });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
        const make = (entryPoint: string) =>
            device.createComputePipeline({
                label: entryPoint,
                layout: pipelineLayout,
                compute: { module, entryPoint },
            });
        this.pipelines = {
            init: make('init'),
            initCoarse: make('init_coarse'),
            propagate: make('propagate'),
            pack: make('pack'),
            consistency: make('consistency'),
        };
    }

    /** Uploads the gray image (width × height) of one view; every view needs one before solving. */
    setImage(layer: number, gray: Uint8Array): void {
        this.device.queue.writeTexture(
            { texture: this.images, origin: { x: 0, y: 0, z: layer } },
            gray as Uint8Array<ArrayBuffer>,
            { bytesPerRow: this.width, rowsPerImage: this.height },
            { width: this.width, height: this.height, depthOrArrayLayers: 1 },
        );
    }

    /** Runs PatchMatch for one reference view and stores its packed depth map. */
    async solve(ref: number, options: SolveOptions): Promise<void> {
        const sources = options.sources.slice(0, MAX_SOURCES);
        if (sources.length === 0) return;
        this.device.queue.writeBuffer(this.prior, 0, options.prior ?? this.zeros);
        const flags = options.geometric ? 1 : 0;
        const start =
            options.fromCoarse && this.coarse ? this.pipelines.initCoarse : this.pipelines.init;
        await this.sliced(start, 8, (row, rows) =>
            this.params(
                ref,
                sources,
                options,
                { iteration: 0, phase: 0, step: 1, flags },
                row,
                rows,
            ),
        );
        for (let iteration = 0; iteration < options.iterations; iteration++) {
            const step = Math.pow(0.5, iteration);
            for (const phase of [0, 1]) {
                // Red/black: a band only reads the other colour, so bands may run in any order.
                await this.sliced(this.pipelines.propagate, 16, (row, rows) =>
                    this.params(
                        ref,
                        sources,
                        options,
                        { iteration, phase, step, flags },
                        row,
                        rows,
                    ),
                );
            }
        }
        if (options.pack === false) return;
        this.device.queue.writeBuffer(
            this.uniforms,
            0,
            this.params(
                ref,
                sources,
                options,
                { iteration: 0, phase: 0, step: 0, flags },
                0,
                this.height,
            ),
        );
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.pack);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil((this.width * this.height) / 64));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    /** The current per-pixel planes (to seed a finer level). */
    get hypotheses(): GPUBuffer {
        return this.hyp;
    }

    /**
     * Multi-view consistency check of one view into `filtered` (left on the GPU): a depth
     * survives when `needed` of the checked views reproject it consistently.
     */
    async filter(ref: number, sources: number[], needed: number): Promise<void> {
        const checked = sources.slice(0, MAX_SOURCES);
        await this.sliced(this.pipelines.consistency, 8, (row, rows) =>
            this.params(
                ref,
                checked,
                { minDepth: 0.01, maxDepth: 100 },
                { iteration: 0, phase: 0, step: 0, flags: 0 },
                row,
                rows,
                needed,
            ),
        );
    }

    /**
     * Runs a per-pixel pipeline over the image in row bands, one short submit per band, waiting
     * for each so other GPU clients (the compositor) get the GPU in between.
     */
    private async sliced(
        pipeline: GPUComputePipeline,
        columnsPerGroup: number,
        params: (row: number, rows: number) => ArrayBuffer,
    ): Promise<void> {
        for (let row = 0; row < this.height;) {
            const rows = Math.min(this.bandRows, this.height - row);
            this.device.queue.writeBuffer(this.uniforms, 0, params(row, rows));
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.width / columnsPerGroup), Math.ceil(rows / 8));
            pass.end();
            const start = performance.now();
            this.device.queue.submit([encoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            this.adapt(performance.now() - start, rows);
            row += rows;
        }
    }

    private adapt(ms: number, rows: number): void {
        if (ms > SUBMIT_MS * 1.5 && this.bandRows > MIN_BAND) {
            const scaled = Math.floor((rows * SUBMIT_MS) / ms / 8) * 8;
            this.bandRows = Math.max(MIN_BAND, Math.min(this.bandRows - 8, scaled));
        } else if (ms < SUBMIT_MS / 3 && rows === this.bandRows && this.bandRows < this.height) {
            this.bandRows = Math.min(Math.ceil(this.height / 8) * 8, this.bandRows * 2);
        }
    }

    /** Filtered packed depth map of one view, read back to the CPU. */
    async filteredMap(ref: number, sources: number[], needed: number): Promise<Uint32Array> {
        await this.filter(ref, sources, needed);
        return new Uint32Array(
            await readBuffer(this.device, this.filtered, this.width * this.height * 4),
        );
    }

    /** Raw packed depth map of one view (before the consistency filter). */
    async depthMap(ref: number): Promise<Uint32Array> {
        const bytes = this.width * this.height * 4;
        return new Uint32Array(await readBuffer(this.device, this.depths, bytes, ref * bytes));
    }

    private params(
        ref: number,
        sources: number[],
        range: { minDepth: number; maxDepth: number },
        state: { iteration: number; phase: number; step: number; flags: number },
        row: number,
        rows: number,
        needed = 0,
    ): ArrayBuffer {
        const { iteration, phase, step, flags } = state;
        const buffer = new ArrayBuffer(PARAMS_BYTES);
        const f = new Float32Array(buffer);
        const u = new Uint32Array(buffer);
        const reference = this.views[ref];
        const k = reference.intrinsics;
        f.set([k.fx, k.fy, k.cx, k.cy], 0);
        u.set([this.width, this.height, ref, sources.length], 4);
        f.set([1 / range.minDepth, 1 / range.maxDepth, step, GEOMETRIC_WEIGHT], 8);
        // Deterministic seed per dispatch so that runs are reproducible.
        const seed = Math.imul(ref + 1, 0x9e3779b1) + iteration * 97 + phase * 13 + flags * 7;
        u.set([iteration, phase, seed >>> 0, flags], 12);
        u.set([row, rows, needed, 0], 16);
        if (this.coarse) {
            const ck = scaleIntrinsics(
                k,
                this.coarse.width / this.width,
                this.coarse.height / this.height,
            );
            f.set([ck.fx, ck.fy, ck.cx, ck.cy], 20);
            u.set([this.coarse.width, this.coarse.height, 0, 0], 24);
        }
        sources.forEach((source, s) => {
            const view = this.views[source];
            const { R, t } = relativePose(reference.pose, view.pose);
            const o = 28 + s * 20;
            const ks = view.intrinsics;
            f.set([ks.fx, ks.fy, ks.cx, ks.cy], o);
            f.set([R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0], o + 4);
            f.set([t[0], t[1], t[2], source], o + 16);
        });
        return buffer;
    }

    destroy(): void {
        this.images.destroy();
        for (const buffer of [
            this.depths,
            this.hyp,
            this.cost,
            this.prior,
            this.filtered,
            this.uniforms,
        ]) {
            buffer.destroy();
        }
        this.placeholder?.destroy();
    }
}
