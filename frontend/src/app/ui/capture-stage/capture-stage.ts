import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    OnInit,
    computed,
    inject,
    signal,
    viewChild,
} from '@angular/core';
import { QUALITY, Quality } from '../../core/models/quality';
import { ScanService } from '../../core/services/scan-service';
import { XrCapture } from '../../core/services/xr-capture';

const MIN_KEYFRAMES = 3;

@Component({
    selector: 'app-capture-stage',
    templateUrl: './capture-stage.html',
    styleUrl: './capture-stage.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CaptureStage implements OnInit {
    private readonly xr = inject(XrCapture);
    private readonly scan = inject(ScanService);
    private readonly overlay = viewChild.required<ElementRef<HTMLElement>>('overlay');
    /** Set when the user closes AR without wanting a model. */
    private abandon = false;

    readonly starting = signal(false);
    readonly notice = signal<string | null>(null);

    readonly active = this.xr.active;
    readonly recording = this.xr.recording;
    readonly tracking = this.xr.tracking;
    readonly tooFast = this.xr.tooFast;
    readonly depthSensing = this.xr.depthSensing;
    readonly keyframes = this.scan.keyframes;
    readonly points = this.scan.sparsePoints;
    readonly rejected = this.scan.rejected;
    readonly ready = this.scan.ready;

    readonly elapsed = computed(() => {
        const seconds = Math.floor(this.xr.elapsed());
        return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    });
    readonly blocker = computed(() => {
        if (this.xr.supported() === false) {
            return (
                this.xr.error() ??
                'AR is not available here. Open this page in Chrome on an ARCore Android phone.'
            );
        }
        if (this.scan.ready() && this.scan.gpuError()) {
            return `WebGPU is needed for the reconstruction: ${this.scan.gpuError()}`;
        }
        return null;
    });
    readonly canStart = computed(
        () => this.ready() && this.xr.supported() === true && !this.blocker() && !this.starting(),
    );
    readonly qualities = (['low', 'medium', 'high'] as Quality[]).map((id) => QUALITY[id]);
    readonly quality = this.scan.quality;
    readonly qualityNote = computed(
        () =>
            ({
                low: 'Fastest: light bundle adjustment, half-resolution stereo, vertex colours.',
                medium: 'Balanced: bundle adjustment, full-resolution stereo, texture atlas.',
                high: 'Best: SIFT poses + self-calibration, 1.5× stereo, weighted fusion (slowest).',
            })[this.quality()],
    );

    setQuality(quality: Quality): void {
        this.scan.setQuality(quality);
    }

    readonly hint = computed(() => {
        if (this.scan.full()) return 'Keyframe budget full — tap the button to build the model.';
        if (this.tracking() === 'lost') return 'Looking for surfaces… move the phone slowly.';
        if (this.tracking() === 'limited')
            return 'Tracking is limited — point at textured surfaces and move slower.';
        if (!this.recording()) return 'Frame the object or room, then tap the button to record.';
        if (this.tooFast()) return 'Slow down — fast motion blurs the frames.';
        if (this.keyframes() < MIN_KEYFRAMES)
            return 'Move around slowly; keyframes are taken every few degrees.';
        return 'Keep circling. Tap the button again to build the model.';
    });

    ngOnInit(): void {
        void this.xr.checkSupport();
    }

    async start(): Promise<void> {
        if (!this.canStart()) return;
        this.notice.set(null);
        this.starting.set(true);
        try {
            await this.xr.start(this.overlay().nativeElement, () => this.sessionEnded());
            if (this.xr.error()) this.notice.set(this.xr.error());
        } finally {
            this.starting.set(false);
        }
    }

    toggleRecording(): void {
        if (this.recording()) this.xr.stopRecording();
        else if (this.tracking() !== 'lost') this.xr.startRecording();
    }

    async cancel(): Promise<void> {
        this.xr.stopRecording();
        this.abandon = true;
        await this.xr.end();
    }

    private sessionEnded(): void {
        const enough = this.scan.keyframes() + this.scan.inFlight() >= MIN_KEYFRAMES;
        if (!this.abandon && this.scan.phase() === 'capturing' && enough) {
            this.scan.finish();
        } else {
            if (!this.abandon && this.scan.phase() === 'capturing') {
                this.notice.set('That was too short — record a few seconds while moving around.');
            }
            this.scan.reset();
        }
        this.abandon = false;
    }
}
