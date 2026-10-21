import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { ScanService } from './core/services/scan-service';
import { BusyVeil } from './ui/busy-veil/busy-veil';
import { CaptureStage } from './ui/capture-stage/capture-stage';
import { ConfirmDialog } from './ui/confirm-dialog/confirm-dialog';
import { ExportSheet } from './ui/export-sheet/export-sheet';
import { InsightsSheet } from './ui/insights-sheet/insights-sheet';
import { ModelViewer } from './ui/model-viewer/model-viewer';
import { PipelineTrack } from './ui/pipeline-track/pipeline-track';

type Sheet = 'none' | 'insights' | 'export';

@Component({
    selector: 'app-root',
    imports: [
        BusyVeil,
        CaptureStage,
        ConfirmDialog,
        ExportSheet,
        InsightsSheet,
        ModelViewer,
        PipelineTrack,
    ],
    templateUrl: './app.html',
    styleUrl: './app.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class App implements OnInit {
    private readonly scan = inject(ScanService);

    readonly sheet = signal<Sheet>('none');
    readonly askingReset = signal(false);
    readonly showModel = computed(() => {
        const phase = this.scan.phase();
        return phase === 'processing' || phase === 'done';
    });
    readonly resetDetail = computed(() =>
        this.scan.phase() === 'processing'
            ? 'The reconstruction in progress will be cancelled.'
            : `The model from ${this.scan.keyframes()} keyframes will be discarded — export it first if you need it.`,
    );

    ngOnInit(): void {
        this.scan.start();
    }

    openSheet(sheet: Sheet): void {
        this.sheet.set(sheet);
    }

    confirmReset(): void {
        this.askingReset.set(false);
        this.sheet.set('none');
        this.scan.reset();
    }
}
