import { ChangeDetectionStrategy, Component, computed, inject, output } from '@angular/core';
import { stageLabel } from '../../core/models/pipeline-stages';
import { ScanService } from '../../core/services/scan-service';

interface Row {
    label: string;
    value: string;
}

interface Section {
    title: string;
    rows: Row[];
}

const mm = (meters: number) => `${(meters * 1000).toFixed(1)} mm`;
const cm = (meters: number) => `${(meters * 100).toFixed(1)} cm`;
const count = (value: number) => value.toLocaleString('en-US');

@Component({
    selector: 'app-insights-sheet',
    templateUrl: './insights-sheet.html',
    styleUrl: './insights-sheet.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class InsightsSheet {
    private readonly scan = inject(ScanService);

    readonly close = output<void>();
    readonly metrics = this.scan.metrics;
    readonly stage = computed(() => this.scan.currentStage()?.label.toLowerCase() ?? '');
    readonly keyframes = this.scan.keyframes;

    readonly sections = computed<Section[]>(() => {
        const m = this.metrics();
        if (!m) return [];
        const closure =
            m.trajectory.length > 0 ? (m.trajectory.closure / m.trajectory.length) * 100 : 0;
        return [
            {
                title: 'capture',
                rows: [
                    {
                        label: 'keyframes',
                        value: `${m.keyframes}${m.blurred ? ` (+${m.blurred} blurred, dropped)` : ''}`,
                    },
                    { label: 'phone depth', value: m.depthSensor ? 'yes' : 'no' },
                    { label: 'stereo size', value: m.workSize.join(' × ') },
                    { label: 'texture size', value: m.textureSize.join(' × ') },
                    { label: 'path length', value: `${m.trajectory.length.toFixed(2)} m` },
                    {
                        label: 'start ↔ end',
                        value: `${cm(m.trajectory.closure)} (${closure.toFixed(1)} %)`,
                    },
                ],
            },
            {
                title: 'sparse · bundle adjustment',
                rows: [
                    { label: 'points', value: count(m.sparsePoints) },
                    { label: 'observations', value: count(m.observations) },
                    {
                        label: 'reprojection, phone poses',
                        value: `${m.reprojection.before.mean.toFixed(2)} / ${m.reprojection.before.median.toFixed(2)} px`,
                    },
                    {
                        label: 'reprojection, adjusted',
                        value: `${m.reprojection.after.mean.toFixed(2)} / ${m.reprojection.after.median.toFixed(2)} px`,
                    },
                    {
                        label: 'pose correction',
                        value: `${mm(m.poseShift.mean)} mean · ${mm(m.poseShift.max)} max`,
                    },
                    { label: 'iterations', value: String(m.bundleIterations) },
                ],
            },
            {
                title: 'dense · mesh · texture',
                rows: [
                    { label: 'dense points', value: count(m.densePoints) },
                    { label: 'density', value: `${count(Math.round(m.density))} pts/m²` },
                    { label: 'voxel', value: mm(m.voxel) },
                    { label: 'triangles', value: count(m.meshTriangles) },
                    { label: 'surface', value: `${m.surfaceArea.toFixed(3)} m²` },
                    {
                        label: 'size (x × y × z)',
                        value: m.extent.map((v) => (v * 100).toFixed(1)).join(' × ') + ' cm',
                    },
                    { label: 'atlas', value: `${m.atlas.join(' × ')} · ${count(m.charts)} charts` },
                    { label: 'untextured faces', value: count(m.untextured) },
                    { label: 'gpu', value: m.gpu },
                ],
            },
        ];
    });

    readonly timings = computed(() => {
        const m = this.metrics();
        if (!m) return null;
        const longest = Math.max(1, ...m.timings.map((t) => t.ms));
        return m.timings.map((t) => ({
            name: stageLabel(t.name).toLowerCase(),
            seconds: (t.ms / 1000).toFixed(1),
            share: (t.ms / longest) * 100,
        }));
    });
}
