import { StageId } from './pipeline-stages';

export type Quality = 'low' | 'medium' | 'high';

export const DEFAULT_QUALITY: Quality = 'high';

/**
 * Interchangeable implementations, registered in vision/features/methods.ts, vision/mvs/methods
 * and vision/texture/methods.
 */
export type FeatureMethodId = 'orb' | 'sift';
export type StereoMethodId = 'single-scale' | 'coarse-to-fine';
export type TextureMethodId = 'vertex' | 'atlas' | 'levelled' | 'mrf';

/** How each stage runs at one quality level. Shared by the worker (what to run) and the UI. */
export interface QualityProfile {
    id: Quality;
    label: string;
    /** Keyframes used for dense reconstruction (redundant / blurry ones are dropped first). */
    views: number;
    bundle: {
        /** Features the poses are refined with: the live ORB map, or re-extracted ones. */
        features: FeatureMethodId;
        rounds: number;
        points: number;
        calibrateFocal: boolean;
    };
    stereo: {
        method: StereoMethodId;
        /**
         * Resolution relative to the work images (640 px wide); above 1 it is decoded from the
         * full keyframes and lowered back towards 1 if the GPU cannot hold every view at it.
         */
        scale: number;
        iterations: number;
        /** Iterations at full resolution after the coarse solve. */
        fineIterations: number;
        sources: number;
    };
    /** Geometric-consistency PatchMatch iterations (0 = skipped). */
    geometric: number;
    /**
     * Source views checked by the consistency filter (at most 8) and the fraction that must
     * agree for the dense cloud; the TSDF may accept a looser agreement (`meshAgreement`),
     * since it averages many depths and outliers carry little weight there.
     */
    filter: { views: number; agreement: number; meshAgreement?: number };
    mesh: {
        /** Voxel size in multiples of the pixel footprint at the median depth. */
        voxelFactor: number;
        /** Cap on surface cells (≈ half the triangle count). */
        cellBudget: number;
        /** Weight depth samples by matching cost and depth (instead of viewing angle alone). */
        confidence: boolean;
    };
    /** Visibility z-buffer resolution relative to the texture images. */
    visibility: number;
    texture: TextureMethodId;
    /** One short line per stage for the pipeline card. */
    methods: Record<StageId, string>;
}

export const QUALITY: Record<Quality, QualityProfile> = {
    low: {
        id: 'low',
        label: 'Low',
        views: 40,
        bundle: { features: 'orb', rounds: 1, points: 3000, calibrateFocal: false },
        stereo: {
            method: 'single-scale',
            scale: 0.5,
            iterations: 3,
            fineIterations: 0,
            sources: 4,
        },
        geometric: 0,
        filter: { views: 4, agreement: 0.5 },
        mesh: { voxelFactor: 2.6, cellBudget: 150_000, confidence: false },
        visibility: 0.25,
        texture: 'vertex',
        methods: {
            capture: 'ORB · epipolar-guided matching · live triangulation',
            bundle: 'bundle adjustment · 3k points · 1 round',
            stereo: 'PatchMatch · ½ resolution · 3 iterations · 4 views',
            geometric: 'skipped',
            filter: 'agreement of 2 of 4 views',
            mesh: 'TSDF · coarse voxels · surface nets',
            views: 'best view per vertex (¼-res z-buffer)',
            texture: 'vertex colours',
        },
    },
    medium: {
        id: 'medium',
        label: 'Medium',
        views: 80,
        bundle: { features: 'orb', rounds: 2, points: 8000, calibrateFocal: false },
        stereo: {
            method: 'single-scale',
            scale: 1,
            iterations: 5,
            fineIterations: 0,
            sources: 5,
        },
        geometric: 1,
        filter: { views: 8, agreement: 0.4 },
        mesh: { voxelFactor: 1.6, cellBudget: 350_000, confidence: false },
        visibility: 0.5,
        texture: 'atlas',
        methods: {
            capture: 'ORB · epipolar-guided matching · live triangulation',
            bundle: 'bundle adjustment · 8k points · robust odometry priors',
            stereo: 'PatchMatch · full resolution · 5 iterations · 5 views',
            geometric: 'ACMM consistency · 1 iteration',
            filter: 'agreement of 40 % of 8 views',
            mesh: 'TSDF · surface nets',
            views: 'chart growing (½-res z-buffer)',
            texture: 'atlas · exposure gains',
        },
    },
    high: {
        id: 'high',
        label: 'High',
        views: 150,
        bundle: { features: 'sift', rounds: 3, points: 20000, calibrateFocal: true },
        stereo: {
            method: 'coarse-to-fine',
            scale: 1.5,
            iterations: 6,
            fineIterations: 4,
            sources: 8,
        },
        geometric: 3,
        filter: { views: 8, agreement: 0.4, meshAgreement: 0.25 },
        mesh: { voxelFactor: 1.2, cellBudget: 600_000, confidence: true },
        visibility: 1,
        texture: 'mrf',
        methods: {
            capture: 'ORB · epipolar-guided matching · live triangulation',
            bundle: 'SIFT on the GPU · 20k points · focal self-calibration',
            stereo: 'coarse-to-fine PatchMatch · 1.5× resolution (960 px) · 8 views',
            geometric: 'ACMM consistency · 3 iterations',
            filter: '40 % of 8 views agree (mesh: 2 of 8)',
            mesh: 'confidence-weighted TSDF · fine voxels · surface nets',
            views: 'MRF view selection (full-res z-buffer)',
            texture: 'atlas · exposure gains + seam levelling',
        },
    },
};
