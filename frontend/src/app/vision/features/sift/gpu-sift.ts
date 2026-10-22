import { BufferUsage, ShaderStage, readBuffer, storageBuffer } from '../../gpu/gpu-device';
import { GrayImage } from '../../image/gray-image';
import { FeatureSet } from '../feature-set';
import { selectOnGrid } from '../grid-selection';
import { SIFT_METRIC } from '../sift-features';
import { SIFT_DIMENSIONS } from './descriptor';
import { CONTRAST, EDGE_RATIO, SiftKeypoint, distinctKeypoints, keypointAt } from './keypoints';
import { INTERVALS, blurSteps, gaussianKernel, octaveSizes } from './scale-space';
import { SIFT_WGSL } from './sift.wgsl';

/** Uniform slot per dispatch (dynamic offsets must be multiples of 256). */
const SLOT = 256;
const JOB_BYTES = 176;
const CAPACITY = 8192;
const MAX_OCTAVES = 8;
const CELL = 24;

type Entry = 'convert' | 'blur_h' | 'blur_v' | 'downsample' | 'extrema' | 'describe';

interface Job {
    entry: Entry;
    groups: [number, number];
    src?: number;
    dst?: number;
    width?: number;
    height?: number;
    radius?: number;
    kernel?: number;
    octave?: number;
    srcWidth?: number;
}

/**
 * The SIFT of sift-features.ts on the GPU, for images of one size: the scale space, DoG extrema
 * and descriptors run as compute passes; only the keypoint list comes back to the CPU (for the
 * grid selection) between the two submits. Buffers are allocated once and reused per image.
 */
export class GpuSift {
    private readonly octaves: { width: number; height: number; base: number }[];
    private readonly pyramid: GPUBuffer;
    private readonly kernels: GPUBuffer;
    private readonly uniforms: GPUBuffer;
    private readonly gray: GPUBuffer;
    private readonly found: GPUBuffer;
    private readonly selected: GPUBuffer;
    private readonly descriptors: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;
    private readonly pipelines: Record<Entry, GPUComputePipeline>;
    private readonly detection: Job[] = [];
    private readonly describeSlot: number;

    constructor(
        private readonly device: GPUDevice,
        readonly width: number,
        readonly height: number,
    ) {
        let offset = 0;
        this.octaves = octaveSizes(width, height)
            .slice(0, MAX_OCTAVES)
            .map((size) => {
                const octave = { ...size, base: offset };
                offset += size.width * size.height * (INTERVALS + 3);
                return octave;
            });
        // Scratch planes after the octaves: the unblurred input and the horizontal blur pass.
        const plane = this.octaves[0].width * this.octaves[0].height;
        const raw = offset;
        const temp = offset + plane;
        this.pyramid = storageBuffer(device, (offset + 2 * plane) * 4, 'sift pyramid');

        const steps = blurSteps().map(gaussianKernel);
        const kernelOffsets: number[] = [];
        const weights: number[] = [];
        for (const kernel of steps) {
            kernelOffsets.push(weights.length);
            weights.push(...kernel);
        }
        this.kernels = storageBuffer(device, weights.length * 4, 'sift kernels');
        device.queue.writeBuffer(this.kernels, 0, new Float32Array(weights));

        const groups = (w: number, h: number): [number, number] => [
            Math.ceil(w / 8),
            Math.ceil(h / 8),
        ];
        const blur = (src: number, dst: number, w: number, h: number, step: number) => {
            const common = { width: w, height: h, radius: steps[step].length - 1 };
            const kernel = kernelOffsets[step];
            this.detection.push(
                { entry: 'blur_h', groups: groups(w, h), src, dst: temp, kernel, ...common },
                { entry: 'blur_v', groups: groups(w, h), src: temp, dst, kernel, ...common },
            );
        };
        this.octaves.forEach((octave, o) => {
            const { width: w, height: h, base } = octave;
            const layer = (l: number) => base + l * w * h;
            if (o === 0) {
                this.detection.push({
                    entry: 'convert',
                    groups: groups(w, h),
                    dst: raw,
                    width: w,
                    height: h,
                    srcWidth: width,
                });
                blur(raw, layer(0), w, h, 0);
            } else {
                const previous = this.octaves[o - 1];
                this.detection.push({
                    entry: 'downsample',
                    groups: groups(w, h),
                    src: previous.base + INTERVALS * previous.width * previous.height,
                    dst: layer(0),
                    width: w,
                    height: h,
                    srcWidth: previous.width,
                });
            }
            for (let l = 1; l < INTERVALS + 3; l++) blur(layer(l - 1), layer(l), w, h, l);
            this.detection.push({ entry: 'extrema', groups: groups(w, h), octave: o });
        });
        this.describeSlot = this.detection.length;

        this.uniforms = device.createBuffer({
            label: 'sift jobs',
            size: (this.detection.length + 1) * SLOT,
            usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        });
        const data = new ArrayBuffer((this.detection.length + 1) * SLOT);
        this.detection.forEach((job, slot) => this.writeJob(data, slot, job, CAPACITY));
        device.queue.writeBuffer(this.uniforms, 0, data);

        this.gray = storageBuffer(device, width * height, 'sift gray');
        this.found = storageBuffer(device, (8 + CAPACITY * 8) * 4, 'sift keypoints');
        this.selected = storageBuffer(device, CAPACITY * 16, 'sift selected');
        this.descriptors = storageBuffer(device, CAPACITY * 33 * 4, 'sift descriptors');

        const storage = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({
            binding,
            visibility: ShaderStage.COMPUTE,
            buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
        });
        const layout = device.createBindGroupLayout({
            entries: [
                storage(0),
                storage(1, true),
                {
                    binding: 2,
                    visibility: ShaderStage.COMPUTE,
                    buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: JOB_BYTES },
                },
                storage(3, true),
                storage(4),
                storage(5, true),
                storage(6),
            ],
        });
        this.bindGroup = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: this.pyramid } },
                { binding: 1, resource: { buffer: this.kernels } },
                { binding: 2, resource: { buffer: this.uniforms, size: JOB_BYTES } },
                { binding: 3, resource: { buffer: this.gray } },
                { binding: 4, resource: { buffer: this.found } },
                { binding: 5, resource: { buffer: this.selected } },
                { binding: 6, resource: { buffer: this.descriptors } },
            ],
        });
        const module = device.createShaderModule({ label: 'sift', code: SIFT_WGSL });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
        const make = (entryPoint: Entry) =>
            device.createComputePipeline({
                label: `sift ${entryPoint}`,
                layout: pipelineLayout,
                compute: { module, entryPoint },
            });
        this.pipelines = {
            convert: make('convert'),
            blur_h: make('blur_h'),
            blur_v: make('blur_v'),
            downsample: make('downsample'),
            extrema: make('extrema'),
            describe: make('describe'),
        };
    }

    async extract(image: GrayImage, budget: number): Promise<FeatureSet> {
        if (image.width !== this.width || image.height !== this.height) {
            throw new Error('GpuSift: image size differs from the one it was built for');
        }
        const { device } = this;
        const bytes = new Uint8Array(Math.ceil(image.data.length / 4) * 4);
        bytes.set(image.data);
        device.queue.writeBuffer(this.gray, 0, bytes);
        device.queue.writeBuffer(this.found, 0, new Uint32Array(1));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        this.detection.forEach((job, slot) => {
            pass.setPipeline(this.pipelines[job.entry]);
            pass.setBindGroup(0, this.bindGroup, [slot * SLOT]);
            pass.dispatchWorkgroups(job.groups[0], job.groups[1]);
        });
        pass.end();
        device.queue.submit([encoder.finish()]);
        const raw = new Uint32Array(await readBuffer(device, this.found, (8 + CAPACITY * 8) * 4));
        const floats = new Float32Array(raw.buffer);
        const count = Math.min(raw[0], CAPACITY);
        const candidates: SiftKeypoint[] = [];
        for (let i = 0; i < count; i++) {
            const o = 8 + i * 8;
            candidates.push(
                keypointAt(floats[o], floats[o + 1], floats[o + 2], floats[o + 3], floats[o + 4]),
            );
        }
        const picked = selectOnGrid(
            distinctKeypoints(candidates),
            this.width,
            this.height,
            CELL,
            budget,
        );
        if (picked.length === 0) return emptySift();

        const keys = new Float32Array(picked.length * 4);
        picked.forEach((k, i) => keys.set([k.octave, k.ox, k.oy, k.scale], i * 4));
        device.queue.writeBuffer(this.selected, 0, keys);
        const jobs = new ArrayBuffer(SLOT);
        this.writeJob(jobs, 0, { entry: 'describe', groups: [picked.length, 1] }, picked.length);
        device.queue.writeBuffer(this.uniforms, this.describeSlot * SLOT, jobs);
        const describe = device.createCommandEncoder();
        const second = describe.beginComputePass();
        second.setPipeline(this.pipelines.describe);
        second.setBindGroup(0, this.bindGroup, [this.describeSlot * SLOT]);
        second.dispatchWorkgroups(picked.length);
        second.end();
        device.queue.submit([describe.finish()]);
        const out = new Uint32Array(
            await readBuffer(device, this.descriptors, picked.length * 33 * 4),
        );
        const words = SIFT_DIMENSIONS / 4;
        const descriptors = new Uint32Array(picked.length * words);
        const angles = new Float32Array(picked.length);
        const angleBits = new Float32Array(out.buffer);
        for (let i = 0; i < picked.length; i++) {
            descriptors.set(out.subarray(i * 33, i * 33 + words), i * words);
            angles[i] = angleBits[i * 33 + words];
        }
        return {
            count: picked.length,
            x: Float32Array.from(picked, (k) => k.x),
            y: Float32Array.from(picked, (k) => k.y),
            level: Uint8Array.from(picked, (k) => k.octave),
            angle: angles,
            descriptors,
            metric: SIFT_METRIC,
        };
    }

    private writeJob(data: ArrayBuffer, slot: number, job: Job, capacity: number): void {
        const u = new Uint32Array(data, slot * SLOT, JOB_BYTES / 4);
        const f = new Float32Array(data, slot * SLOT, JOB_BYTES / 4);
        u.set([job.src ?? 0, job.dst ?? 0, job.width ?? 0, job.height ?? 0], 0);
        u.set([job.radius ?? 0, job.kernel ?? 0, job.octave ?? 0, job.srcWidth ?? 0], 4);
        f.set([(0.5 * CONTRAST) / INTERVALS, CONTRAST, EDGE_RATIO], 8);
        u[11] = capacity;
        this.octaves.forEach((octave, o) =>
            u.set(
                [octave.base, octave.width, octave.height, octave.width * octave.height],
                12 + o * 4,
            ),
        );
    }

    destroy(): void {
        for (const buffer of [
            this.pyramid,
            this.kernels,
            this.uniforms,
            this.gray,
            this.found,
            this.selected,
            this.descriptors,
        ]) {
            buffer.destroy();
        }
    }
}

function emptySift(): FeatureSet {
    return {
        count: 0,
        x: new Float32Array(0),
        y: new Float32Array(0),
        level: new Uint8Array(0),
        angle: new Float32Array(0),
        descriptors: new Uint32Array(0),
        metric: SIFT_METRIC,
    };
}
