import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { ScanService } from '../../core/services/scan-service';

@Component({
    selector: 'app-busy-veil',
    templateUrl: './busy-veil.html',
    styleUrl: './busy-veil.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { '[class.visible]': 'visible()' },
})
export class BusyVeil {
    private readonly scan = inject(ScanService);

    /** Only while the worker boots; the pipeline track reports the reconstruction itself. */
    readonly visible = computed(() => !this.scan.ready());
    readonly label = computed(() => 'starting');
}
