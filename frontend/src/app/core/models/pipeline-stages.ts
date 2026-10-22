/** The reconstruction pipeline, in execution order (shown in the app and in the metrics). */
export const PIPELINE_STAGES = [
    {
        id: 'capture',
        label: 'Capture & live tracking',
        detail: 'keyframes · ORB features · guided matching · triangulation',
    },
    {
        id: 'bundle',
        label: 'Bundle adjustment',
        detail: 'poses + points (+ focal) with the phone odometry as priors',
    },
    { id: 'stereo', label: 'Dense stereo', detail: 'PatchMatch multi-view stereo on the GPU' },
    {
        id: 'geometric',
        label: 'Geometric refinement',
        detail: 'PatchMatch with multi-view depth consistency',
    },
    { id: 'filter', label: 'Depth fusion', detail: 'consistency filter → dense point cloud' },
    { id: 'mesh', label: 'Meshing', detail: 'TSDF volume → surface nets → clean-up' },
    { id: 'views', label: 'View selection', detail: 'visibility and best camera per face' },
    { id: 'texture', label: 'Texture', detail: 'atlas with colour levelling, or vertex colours' },
] as const;

export type StageId = (typeof PIPELINE_STAGES)[number]['id'];

/** Worker-side stages (capture is timed on the main thread while recording). */
export type WorkerStageId = Exclude<StageId, 'capture'>;

export interface StageEvent {
    stage: WorkerStageId;
    status: 'running' | 'done';
    /** 0–1, or −1 when the stage has no measurable progress. */
    fraction: number;
    /** Short live detail, e.g. "37/119". */
    detail: string | null;
    /** Duration in milliseconds, set when the stage is done. */
    ms: number | null;
}

export function stageLabel(id: string): string {
    return PIPELINE_STAGES.find((stage) => stage.id === id)?.label ?? id;
}
