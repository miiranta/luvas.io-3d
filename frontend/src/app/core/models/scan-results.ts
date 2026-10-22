import { Quality } from './quality';

export interface ErrorSummary {
    mean: number;
    median: number;
}

export interface StageTiming {
    name: string;
    ms: number;
}

/** Numbers reported in the diagnostics sheet and exported as JSON for the report. */
export interface ScanMetrics {
    quality: Quality;
    gpu: string;
    keyframes: number;
    /** Keyframes used for dense reconstruction at this quality. */
    keyframesUsed: number;
    /** Odometry breaks (tracking loss / pose jumps), unguided rescues and re-anchored segments. */
    tracking: { breaks: number; fallbacks: number; realigned: number };
    /** Relative focal-length correction found by self-calibration (0 when not calibrated). */
    focalCorrection: number;
    stereoSize: [number, number];
    blurred: number;
    depthSensor: boolean;
    workSize: [number, number];
    textureSize: [number, number];
    tracks: number;
    sparsePoints: number;
    observations: number;
    /** Reprojection error with the phone poses and after bundle adjustment (px, work res). */
    reprojection: { before: ErrorSummary; after: ErrorSummary };
    /** How far bundle adjustment moved the camera centres (m). */
    poseShift: { mean: number; max: number };
    bundleIterations: number;
    /** Length of the camera path (m) and distance between first and last centre (m). */
    trajectory: { length: number; closure: number };
    densePoints: number;
    voxel: number;
    meshVertices: number;
    meshTriangles: number;
    surfaceArea: number;
    /** Dense points per square metre of mesh surface. */
    density: number;
    /** Axis-aligned size of the mesh in the gravity-aligned world frame (m). */
    extent: [number, number, number];
    /** Atlas size, null for vertex colours. */
    atlas: [number, number] | null;
    charts: number;
    untextured: number;
    timings: StageTiming[];
}

/**
 * Cameras for drawing: per camera 14 floats — world → camera rotation (9, row-major, CV axes),
 * centre (3), tan of the half field of view in x and y.
 */
export const CAMERA_STRIDE = 14;
