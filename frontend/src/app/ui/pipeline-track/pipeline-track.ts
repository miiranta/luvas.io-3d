import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { QUALITY, Quality } from '../../core/models/quality';
import { ScanService, StageState } from '../../core/services/scan-service';

interface Row {
    id: string;
    label: string;
    method: string;
    status: StageState['status'];
    time: string;
    fraction: number;
    note: string | null;
}

/** "0.4 s", "12.3 s", "1:07". */
export function formatDuration(ms: number): string {
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)} s`;
    const whole = Math.round(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/**
 * The reconstruction pipeline as a live checklist: which stage is running (with progress and
 * a running timer), which are done and how long each one took.
 */
@Component({
    selector: 'app-pipeline-track',
    templateUrl: './pipeline-track.html',
    styleUrl: './pipeline-track.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PipelineTrack {
    private readonly scan = inject(ScanService);

    readonly open = signal(true);

    readonly building = this.scan.busy;
    readonly failed = computed(() => this.scan.stages().some((stage) => stage.status === 'failed'));
    readonly title = computed(() => {
        if (this.failed()) return 'Build failed';
        if (this.building()) return this.scan.currentStage()?.label ?? 'Building model';
        return 'Model built';
    });
    readonly total = computed(() => formatDuration(this.scan.totalMs()));
    /** Time spent building after the recording stopped (everything but the capture). */
    readonly build = computed(() => {
        const capture = this.scan.stages().find((stage) => stage.id === 'capture');
        const captureMs = capture ? (capture.ms ?? this.scan.elapsedOf(capture)) : 0;
        return formatDuration(Math.max(0, this.scan.totalMs() - captureMs));
    });
    readonly done = computed(
        () => this.scan.stages().filter((stage) => stage.status === 'done').length,
    );
    readonly count = computed(() => this.scan.stages().length);

    readonly rows = computed<Row[]>(() =>
        this.scan.stages().map((stage) => {
            let time = '—';
            if (stage.ms !== null) time = formatDuration(stage.ms);
            else if (stage.status === 'running') time = formatDuration(this.scan.elapsedOf(stage));
            let note = stage.note;
            if (stage.id === 'capture' && stage.status === 'done')
                note = `${this.scan.keyframes()} keyframes`;
            return {
                id: stage.id,
                label: stage.label,
                method: stage.method,
                status: stage.status,
                time,
                fraction: stage.fraction,
                note,
            };
        }),
    );

    constructor() {
        // Fold away once the model is finished; the user can reopen it.
        effect(() => {
            if (this.scan.phase() === 'done') this.open.set(false);
        });
    }

    toggle(): void {
        this.open.update((open) => !open);
    }

    readonly qualities = (['low', 'medium', 'high'] as Quality[]).map((id) => QUALITY[id]);
    readonly quality = computed(() => QUALITY[this.scan.builtQuality() ?? this.scan.quality()]);
    /** A finished (or failed) build can be redone at another quality from the same capture. */
    readonly canRebuild = computed(() => this.scan.phase() === 'done');

    rebuild(quality: Quality): void {
        this.open.set(true);
        this.scan.rebuild(quality);
    }
}
