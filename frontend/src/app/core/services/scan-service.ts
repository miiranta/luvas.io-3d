import { Injectable, computed, signal } from '@angular/core';
import { PIPELINE_STAGES, StageEvent, StageId } from '../models/pipeline-stages';
import { DEFAULT_QUALITY, QUALITY, Quality } from '../models/quality';
import { ScanMetrics } from '../models/scan-results';
import {
    ExportFormat,
    KeyframeMessage,
    WorkerRequest,
    WorkerResponse,
} from '../models/worker-protocol';

export type Phase = 'idle' | 'capturing' | 'processing' | 'done';

export interface Cloud {
    positions: Float32Array;
    colors: Uint8Array;
}

/** The final model: textured (uvs + atlas) or vertex-coloured. */
export interface FinalModel {
    positions: Float32Array;
    indices: Uint32Array;
    uvs: Float32Array | null;
    colors: Uint8Array | null;
    atlas: ImageBitmap | null;
}

export interface StageState {
    id: StageId;
    label: string;
    /** The method this stage uses at the selected quality. */
    method: string;
    status: 'pending' | 'running' | 'done' | 'failed';
    /** Final duration (ms) once done. */
    ms: number | null;
    /** Main-thread clock when the stage started, for the live timer. */
    startedAt: number | null;
    fraction: number;
    note: string | null;
}

function initialStages(quality: Quality): StageState[] {
    return PIPELINE_STAGES.map((stage) => ({
        id: stage.id,
        label: stage.label,
        method: QUALITY[quality].methods[stage.id],
        status: 'pending',
        ms: null,
        startedAt: null,
        fraction: -1,
        note: null,
    }));
}

const CLOCK_MS = 200;
const QUALITY_KEY = 'scan3d.quality';

function storedQuality(): Quality {
    try {
        const value = localStorage.getItem(QUALITY_KEY);
        if (value === 'low' || value === 'medium' || value === 'high') return value;
    } catch {
        // Storage can be unavailable (private mode); the default applies.
    }
    return DEFAULT_QUALITY;
}

type VerdictListener = (id: number, accepted: boolean, reason: string | null) => void;

/** Bridge to the reconstruction worker; every result is exposed as a signal. */
@Injectable({ providedIn: 'root' })
export class ScanService {
    private worker: Worker | null = null;
    private readonly verdictListeners = new Set<VerdictListener>();
    private pendingExport: {
        resolve: (blob: Blob) => void;
        reject: (error: Error) => void;
    } | null = null;
    private ticker: ReturnType<typeof setInterval> | null = null;

    readonly ready = signal(false);
    readonly gpu = signal<string | null>(null);
    readonly gpuError = signal<string | null>(null);
    readonly phase = signal<Phase>('idle');
    readonly keyframes = signal(0);
    readonly rejected = signal(0);
    readonly sparsePoints = signal(0);
    readonly sceneDepth = signal(1);
    readonly inFlight = signal(0);
    readonly preview = signal<Cloud | null>(null);
    /** Quality of the next build (chosen before recording, or for a rebuild). */
    readonly quality = signal<Quality>(storedQuality());
    /** Quality of the model on screen / being built. */
    readonly builtQuality = signal<Quality | null>(null);
    /** The worker reached its keyframe budget: recording should stop. */
    readonly full = signal(false);
    readonly stages = signal<StageState[]>(initialStages(this.quality()));
    /** Ticks while capturing or building, so running stages show a live timer. */
    readonly clock = signal(0);
    readonly cameras = signal<Float32Array | null>(null);
    readonly sparse = signal<Cloud | null>(null);
    readonly dense = signal<Cloud | null>(null);
    readonly mesh = signal<{ positions: Float32Array; indices: Uint32Array } | null>(null);
    readonly model = signal<FinalModel | null>(null);
    readonly metrics = signal<ScanMetrics | null>(null);
    readonly error = signal<string | null>(null);

    readonly busy = computed(() => this.phase() === 'processing');
    readonly currentStage = computed(
        () => this.stages().find((stage) => stage.status === 'running') ?? null,
    );
    readonly totalMs = computed(() =>
        this.stages().reduce((sum, stage) => sum + (stage.ms ?? this.elapsedOf(stage)), 0),
    );

    /** Elapsed time of a running stage (ms), 0 otherwise. */
    elapsedOf(stage: StageState): number {
        if (stage.status !== 'running' || stage.startedAt === null) return 0;
        return Math.max(0, this.clock() - stage.startedAt);
    }

    start(): void {
        if (this.worker) return;
        this.worker = new Worker(new URL('../workers/reconstruct.worker', import.meta.url), {
            type: 'module',
        });
        this.worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) =>
            this.handle(event.data),
        );
    }

    onVerdict(listener: VerdictListener): () => void {
        this.verdictListeners.add(listener);
        return () => this.verdictListeners.delete(listener);
    }

    private send(request: WorkerRequest, transfer: Transferable[] = []): void {
        this.start();
        this.worker?.postMessage(request, transfer);
    }

    private handle(message: WorkerResponse): void {
        switch (message.kind) {
            case 'ready':
                this.ready.set(true);
                this.gpu.set(message.gpu);
                this.gpuError.set(message.error);
                break;
            case 'keyframe':
                this.inFlight.update((n) => Math.max(0, n - 1));
                this.keyframes.set(message.keyframes);
                this.sparsePoints.set(message.sparsePoints);
                this.sceneDepth.set(message.sceneDepth);
                if (!message.accepted && message.reason === 'blurry')
                    this.rejected.update((n) => n + 1);
                if (!message.accepted && message.reason === 'full') this.full.set(true);
                for (const listener of this.verdictListeners) {
                    listener(message.id, message.accepted, message.reason);
                }
                break;
            case 'preview':
                this.preview.set({
                    positions: new Float32Array(message.positions),
                    colors: new Uint8Array(message.colors),
                });
                break;
            case 'stage':
                this.applyStage(message.event);
                break;
            case 'cameras':
                this.cameras.set(new Float32Array(message.cameras));
                break;
            case 'sparse':
                this.sparse.set({
                    positions: new Float32Array(message.positions),
                    colors: new Uint8Array(message.colors),
                });
                break;
            case 'dense':
                this.dense.set({
                    positions: new Float32Array(message.positions),
                    colors: new Uint8Array(message.colors),
                });
                break;
            case 'mesh':
                this.mesh.set({
                    positions: new Float32Array(message.positions),
                    indices: new Uint32Array(message.indices),
                });
                break;
            case 'model': {
                const model: FinalModel = {
                    positions: new Float32Array(message.positions),
                    indices: new Uint32Array(message.indices),
                    uvs: message.uvs && new Float32Array(message.uvs),
                    colors: message.colors && new Uint8Array(message.colors),
                    atlas: null,
                };
                if (!message.atlas) {
                    this.model.set(model);
                    break;
                }
                void createImageBitmap(message.atlas).then((atlas) =>
                    this.model.set({ ...model, atlas }),
                );
                break;
            }
            case 'metrics':
                this.metrics.set(message.metrics);
                break;
            case 'done':
                this.phase.set('done');
                this.stopClock();
                break;
            case 'export':
                this.pendingExport?.resolve(message.blob);
                this.pendingExport = null;
                break;
            case 'error':
                this.error.set(message.message);
                if (message.request === 'keyframe') this.inFlight.update((n) => Math.max(0, n - 1));
                if (message.request === 'finish') {
                    this.stages.update((stages) =>
                        stages.map((stage) =>
                            stage.status === 'running'
                                ? { ...stage, status: 'failed', ms: this.elapsedOf(stage) }
                                : stage,
                        ),
                    );
                    this.stopClock();
                    // Keep the capture: the build can be retried at another quality.
                    this.phase.set(this.keyframes() >= 3 ? 'done' : 'idle');
                }
                if (message.request === 'export') {
                    this.pendingExport?.reject(new Error(message.message));
                    this.pendingExport = null;
                }
                break;
        }
    }

    setQuality(quality: Quality): void {
        this.quality.set(quality);
        try {
            localStorage.setItem(QUALITY_KEY, quality);
        } catch {
            // Not persisted; still used for this session.
        }
        if (this.phase() === 'idle' || this.phase() === 'capturing') {
            this.stages.update((stages) =>
                stages.map((stage) => ({ ...stage, method: QUALITY[quality].methods[stage.id] })),
            );
        }
    }

    beginCapture(): void {
        this.phase.set('capturing');
        this.error.set(null);
        this.full.set(false);
        const now = performance.now();
        this.stages.set(
            initialStages(this.quality()).map((stage) =>
                stage.id === 'capture' ? { ...stage, status: 'running', startedAt: now } : stage,
            ),
        );
        this.startClock();
    }

    private applyStage(event: StageEvent): void {
        const now = performance.now();
        this.clock.set(now);
        this.stages.update((stages) =>
            stages.map((stage) => {
                // Capture ends when the worker, done with the queued keyframes, starts the pipeline.
                if (stage.id === 'capture' && stage.status === 'running') {
                    return {
                        ...stage,
                        status: 'done',
                        ms: now - (stage.startedAt ?? now),
                        fraction: 1,
                    };
                }
                if (stage.id !== event.stage) return stage;
                if (event.status === 'done')
                    return { ...stage, status: 'done', ms: event.ms, fraction: 1, note: null };
                return {
                    ...stage,
                    status: 'running',
                    startedAt: stage.startedAt ?? now,
                    fraction: event.fraction,
                    note: event.detail,
                };
            }),
        );
    }

    private startClock(): void {
        this.stopClock();
        this.clock.set(performance.now());
        this.ticker = setInterval(() => this.clock.set(performance.now()), CLOCK_MS);
    }

    private stopClock(): void {
        if (this.ticker !== null) clearInterval(this.ticker);
        this.ticker = null;
        this.clock.set(performance.now());
    }

    sendKeyframe(frame: KeyframeMessage): void {
        this.inFlight.update((n) => n + 1);
        const transfer: Transferable[] = [frame.pixels];
        if (frame.depth) transfer.push(frame.depth.meters);
        this.send({ kind: 'keyframe', frame }, transfer);
    }

    finish(): void {
        this.phase.set('processing');
        this.builtQuality.set(this.quality());
        this.stages.update((stages) =>
            stages.map((stage) =>
                stage.id === 'capture' ? { ...stage, note: 'finishing keyframes' } : stage,
            ),
        );
        this.send({ kind: 'finish', quality: this.quality() });
    }

    /** Builds the same capture again at another quality. */
    rebuild(quality: Quality): void {
        if (this.phase() !== 'done') return;
        this.setQuality(quality);
        this.builtQuality.set(quality);
        this.model()?.atlas?.close();
        this.cameras.set(null);
        this.sparse.set(null);
        this.dense.set(null);
        this.mesh.set(null);
        this.model.set(null);
        this.metrics.set(null);
        this.error.set(null);
        const capture = this.stages().find((stage) => stage.id === 'capture');
        this.stages.set(
            initialStages(quality).map((stage) =>
                stage.id === 'capture' && capture ? { ...capture, note: null } : stage,
            ),
        );
        this.phase.set('processing');
        this.startClock();
        this.send({ kind: 'finish', quality });
    }

    exportFile(format: ExportFormat): Promise<Blob> {
        return new Promise((resolve, reject) => {
            this.pendingExport?.reject(new Error('export superseded'));
            this.pendingExport = { resolve, reject };
            this.send({ kind: 'export', format });
        });
    }

    /** Drops everything; the worker is replaced so that a running reconstruction stops too. */
    reset(): void {
        this.worker?.terminate();
        this.worker = null;
        this.ready.set(false);
        this.pendingExport?.reject(new Error('scan reset'));
        this.pendingExport = null;
        this.model()?.atlas?.close();
        this.preview.set(null);
        this.cameras.set(null);
        this.sparse.set(null);
        this.dense.set(null);
        this.mesh.set(null);
        this.model.set(null);
        this.metrics.set(null);
        this.builtQuality.set(null);
        this.full.set(false);
        this.keyframes.set(0);
        this.rejected.set(0);
        this.sparsePoints.set(0);
        this.inFlight.set(0);
        this.stopClock();
        this.stages.set(initialStages(this.quality()));
        this.error.set(null);
        this.phase.set('idle');
        this.start();
    }
}
