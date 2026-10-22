export interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

export const log: string[] = [];
export const checks: Check[] = [];

export function check(name: string, pass: boolean, detail: string): void {
    checks.push({ name, pass, detail });
}

export function percentile(values: number[], q: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? NaN;
}
