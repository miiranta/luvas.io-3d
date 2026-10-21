import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

@Component({
    selector: 'app-confirm-dialog',
    templateUrl: './confirm-dialog.html',
    styleUrl: './confirm-dialog.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConfirmDialog {
    readonly message = input.required<string>();
    readonly detail = input('');
    readonly confirmLabel = input('Confirm');
    readonly confirmed = output<void>();
    readonly dismissed = output<void>();
}
