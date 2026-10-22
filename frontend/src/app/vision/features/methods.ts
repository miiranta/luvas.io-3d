import { FeatureMethodId } from '../../core/models/quality';
import { GrayImage } from '../image/gray-image';
import { FeatureSet } from './feature-set';
import { extractFeatures } from './orb-features';
import { GpuSift } from './sift/gpu-sift';
import { extractSift } from './sift-features';

/** Extracts features from many images of one size (reusing buffers between them). */
export interface FeatureExtractor {
    extract(image: GrayImage): Promise<FeatureSet>;
    destroy(): void;
}

/** A keypoint detector + descriptor; matchers work with any of them through `FeatureSet.metric`. */
export interface FeatureMethod {
    name: string;
    /** Keypoints per work-resolution image. */
    budget: number;
    /** CPU implementation (the reference). */
    extract(image: GrayImage, budget: number): FeatureSet;
    /** Batch extractor, on the GPU when the method has a GPU version and a device is given. */
    extractor(device: GPUDevice | null, width: number, height: number): FeatureExtractor;
}

function cpuExtractor(method: Pick<FeatureMethod, 'extract' | 'budget'>): FeatureExtractor {
    return {
        extract: async (image) => method.extract(image, method.budget),
        destroy: () => undefined,
    };
}

const orb: FeatureMethod = {
    name: 'ORB',
    budget: 1600,
    extract: extractFeatures,
    extractor: () => cpuExtractor(orb),
};

const sift: FeatureMethod = {
    name: 'SIFT',
    budget: 2000,
    extract: extractSift,
    extractor(device, width, height) {
        if (!device) return cpuExtractor(sift);
        const gpu = new GpuSift(device, width, height);
        return {
            extract: (image) => gpu.extract(image, sift.budget),
            destroy: () => gpu.destroy(),
        };
    },
};

export const FEATURE_METHODS: Record<FeatureMethodId, FeatureMethod> = { orb, sift };
