/** How the descriptors of one feature type are compared, so matchers work with any detector. */
export interface DescriptorMetric {
    readonly name: string;
    /** u32 words per descriptor. */
    readonly words: number;
    /**
     * Distance between the descriptors starting at word offsets ia and ib. With a `bound` it may
     * stop early once the distance is known to exceed it, returning some value above it.
     */
    distance(a: Uint32Array, ia: number, b: Uint32Array, ib: number, bound?: number): number;
    /** Largest distance a true match may have. */
    readonly maxDistance: number;
}

/** Keypoints and descriptors of one image, stored as parallel arrays. */
export interface FeatureSet {
    count: number;
    /** Positions in pixels of the full-resolution (level 0) image. */
    x: Float32Array;
    y: Float32Array;
    /** Pyramid level (scale index) of each keypoint. */
    level: Uint8Array;
    angle: Float32Array;
    /** `metric.words` u32 words per keypoint. */
    descriptors: Uint32Array;
    metric: DescriptorMetric;
}
