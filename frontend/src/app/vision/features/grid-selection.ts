/**
 * Strongest candidates per grid cell first, then the best of the rest, up to the budget, so that
 * features cover the whole image instead of clustering on the most textured part.
 */
export function selectOnGrid<T extends { x: number; y: number; score: number }>(
    candidates: T[],
    width: number,
    height: number,
    cellSize: number,
    budget: number,
): T[] {
    candidates.sort((a, b) => b.score - a.score);
    const columns = Math.ceil(width / cellSize);
    const rows = Math.ceil(height / cellSize);
    const perCell = Math.max(1, Math.ceil((budget / (columns * rows)) * 1.2));
    const filled = new Uint16Array(columns * rows);
    const chosen: T[] = [];
    const rest: T[] = [];
    for (const candidate of candidates) {
        const cx = Math.min(columns - 1, Math.max(0, Math.floor(candidate.x / cellSize)));
        const cy = Math.min(rows - 1, Math.max(0, Math.floor(candidate.y / cellSize)));
        const cell = cy * columns + cx;
        if (filled[cell] < perCell && chosen.length < budget) {
            filled[cell]++;
            chosen.push(candidate);
        } else {
            rest.push(candidate);
        }
    }
    for (const candidate of rest) {
        if (chosen.length >= budget) break;
        chosen.push(candidate);
    }
    return chosen;
}
