import { StageEvent, WorkerStageId } from '../../../core/models/pipeline-stages';
import { StageTiming } from '../../../core/models/scan-results';
import { PointCloud } from '../../fusion/point-accumulator';
import { Mesh } from '../../fusion/mesh';
import { GpuContext } from '../../gpu/gpu-device';

/** Receives intermediate results so the viewer can show them while the next stage runs. */
export interface ReconstructionSink {
    stage(event: StageEvent): void;
    cameras(cameras: Float32Array): void;
    sparse(positions: Float32Array, colors: Uint8Array): void;
    dense(cloud: PointCloud): void;
    mesh(mesh: Mesh): void;
}

export type Progress = (fraction: number, detail?: string | null) => void;

/**
 * Runs pipeline stages one after another: reports start / progress / end with the measured
 * time, and turns asynchronous GPU failures into exceptions naming the stage.
 */
export class StageRunner {
    readonly timings: StageTiming[] = [];

    constructor(
        private readonly sink: ReconstructionSink,
        private readonly gpu: GpuContext,
    ) {}

    progress(stage: WorkerStageId): Progress {
        return (fraction, detail = null) =>
            this.sink.stage({ stage, status: 'running', fraction, detail, ms: null });
    }

    async run<T>(stage: WorkerStageId, work: (progress: Progress) => Promise<T> | T): Promise<T> {
        const progress = this.progress(stage);
        progress(-1);
        const start = performance.now();
        const result = await work(progress);
        if (this.gpu.error) {
            const message = this.gpu.error;
            this.gpu.error = null;
            throw new Error(`GPU error during ${stage}: ${message}`);
        }
        if (this.gpu.lost) {
            throw new Error(
                `the phone reset the GPU during ${stage} (${this.gpu.lost}) — close other apps and try a shorter scan or a lower quality`,
            );
        }
        const ms = performance.now() - start;
        this.timings.push({ name: stage, ms });
        this.sink.stage({ stage, status: 'done', fraction: 1, detail: null, ms });
        return result;
    }
}
