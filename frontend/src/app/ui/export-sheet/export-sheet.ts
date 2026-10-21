import { ChangeDetectionStrategy, Component, inject, output, signal } from '@angular/core';
import { ExportFormat } from '../../core/models/worker-protocol';
import { ScanService } from '../../core/services/scan-service';

interface Choice {
    format: ExportFormat;
    title: string;
    detail: string;
    extension: string;
}

const REVOKE_DELAY_MS = 60_000;

@Component({
    selector: 'app-export-sheet',
    templateUrl: './export-sheet.html',
    styleUrl: './export-sheet.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExportSheet {
    private readonly scan = inject(ScanService);

    readonly close = output<void>();
    readonly saving = signal<ExportFormat | null>(null);

    readonly choices: Choice[] = [
        {
            format: 'glb',
            title: 'Model · GLB',
            detail: 'Textured mesh, camera frustums and sparse points — Blender, MeshLab, three.js.',
            extension: 'glb',
        },
        {
            format: 'dense',
            title: 'Dense cloud · PLY',
            detail: 'Fused points with normals and colour, for MeshLab / Open3D / CloudCompare.',
            extension: 'ply',
        },
        {
            format: 'sparse',
            title: 'Sparse cloud · PLY',
            detail: 'Triangulated feature points after bundle adjustment.',
            extension: 'ply',
        },
        {
            format: 'metrics',
            title: 'Metrics · JSON',
            detail: 'Reprojection error, counts, density, sizes and timings for the report.',
            extension: 'json',
        },
    ];

    async save(choice: Choice): Promise<void> {
        if (this.saving()) return;
        this.saving.set(choice.format);
        try {
            const blob = await this.scan.exportFile(choice.format);
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `scan-${choice.format}-${stamp()}.${choice.extension}`;
            anchor.click();
            setTimeout(() => URL.revokeObjectURL(url), REVOKE_DELAY_MS);
        } catch (error) {
            this.scan.error.set(error instanceof Error ? error.message : String(error));
        } finally {
            this.saving.set(null);
        }
    }
}

function stamp(): string {
    const now = new Date();
    const pad = (value: number) => String(value).padStart(2, '0');
    return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
}
