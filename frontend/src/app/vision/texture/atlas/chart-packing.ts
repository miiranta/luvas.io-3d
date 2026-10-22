import { Chart } from './charts';

export interface AtlasLayout {
    width: number;
    height: number;
    scale: number;
    greyX: number;
    greyY: number;
}

/** Shelf packing (tallest first); shrinks every chart uniformly when the atlas would overflow. */
export function packCharts(charts: Chart[], maxSize: number): AtlasLayout {
    const textured = charts.filter((chart) => chart.view >= 0);
    const order = [...textured].sort((a, b) => b.h - a.h);
    let area = 16;
    for (const chart of textured) area += chart.w * chart.h;
    let widest = 0;
    for (const chart of textured) widest = Math.max(widest, chart.w);
    let scale = Math.min(1, maxSize / Math.sqrt(area * 1.12), (maxSize - 8) / widest);
    for (let attempt = 0; attempt < 12; attempt++) {
        const width = Math.min(
            maxSize,
            Math.max(64, Math.ceil(Math.sqrt(area * 1.12) * scale), Math.ceil(widest * scale) + 5),
        );
        let x = 4;
        let y = 0;
        let shelf = 4;
        let fits = true;
        for (const chart of order) {
            const w = Math.ceil(chart.w * scale) + 1;
            const h = Math.ceil(chart.h * scale) + 1;
            if (x + w > width) {
                x = 0;
                y += shelf;
                shelf = 0;
            }
            chart.ax = x;
            chart.ay = y;
            x += w;
            shelf = Math.max(shelf, h);
            if (y + shelf > maxSize) {
                fits = false;
                break;
            }
        }
        if (fits) {
            return { width, height: Math.max(4, y + shelf), scale, greyX: 1, greyY: 1 };
        }
        scale *= 0.9;
    }
    throw new Error('texture atlas does not fit');
}
