import { DenseStereo, StereoView } from '../dense-stereo';

/** What a stereo method needs for every view (layer) it reconstructs. */
export interface StereoInput {
    /** Full stereo resolution. */
    width: number;
    height: number;
    /** Intrinsics (at width × height) and pose per layer. */
    views: StereoView[];
    /** Source layers per layer. */
    sources: number[][];
    /** Depth search interval per layer (m). */
    ranges: [number, number][];
    /** Gray image of a layer at any resolution (decoded from the full image above work size). */
    gray(layer: number, width: number, height: number): Promise<Uint8Array>;
    /** Phone depth of a layer at any resolution (null when the phone had none). */
    prior(layer: number, width: number, height: number): Float32Array | null;
}

export interface StereoSettings {
    iterations: number;
    /** Full-resolution iterations after a coarse solve (coarse-to-fine only). */
    fineIterations: number;
}

/** Uploads every layer's gray image, one at a time (they are only needed on the GPU). */
export async function uploadImages(stereo: DenseStereo, input: StereoInput): Promise<void> {
    for (let layer = 0; layer < input.views.length; layer++) {
        stereo.setImage(layer, await input.gray(layer, stereo.width, stereo.height));
    }
}

/** A running reconstruction: photometric depth for one view at a time. */
export interface StereoRun {
    /** The full-resolution level (the geometric pass, the filter and fusion read it). */
    readonly stereo: DenseStereo;
    solve(layer: number): Promise<void>;
    destroy(): void;
}

/** A multi-view stereo strategy; registered in `methods/index.ts`. */
export interface StereoMethod {
    create(device: GPUDevice, input: StereoInput, settings: StereoSettings): Promise<StereoRun>;
}
