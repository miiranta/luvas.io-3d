import { Intrinsics } from '../../vision/geometry/camera';
import { StageEvent } from './pipeline-stages';
import { ScanMetrics } from './scan-results';

export type ExportFormat = 'glb' | 'dense' | 'sparse' | 'metrics';

export interface KeyframeMessage {
    id: number;
    /** Seconds since recording started. */
    time: number;
    width: number;
    height: number;
    /** RGBA, first row = top of the view. */
    pixels: ArrayBuffer;
    intrinsics: Intrinsics;
    /** WebXR view transform (camera → world, column-major, OpenGL axes). */
    transform: number[];
    depth: { width: number; height: number; meters: ArrayBuffer } | null;
}

export type WorkerRequest =
    | { kind: 'keyframe'; frame: KeyframeMessage }
    | { kind: 'finish' }
    | { kind: 'reset' }
    | { kind: 'export'; format: ExportFormat };

export type WorkerResponse =
    | { kind: 'ready'; gpu: string | null; error: string | null }
    | {
          kind: 'keyframe';
          id: number;
          accepted: boolean;
          reason: string | null;
          keyframes: number;
          sparsePoints: number;
          sceneDepth: number;
      }
    | { kind: 'preview'; positions: ArrayBuffer; colors: ArrayBuffer }
    | { kind: 'stage'; event: StageEvent }
    | { kind: 'cameras'; cameras: ArrayBuffer }
    | { kind: 'sparse'; positions: ArrayBuffer; colors: ArrayBuffer }
    | { kind: 'dense'; positions: ArrayBuffer; colors: ArrayBuffer }
    | { kind: 'mesh'; positions: ArrayBuffer; normals: ArrayBuffer; indices: ArrayBuffer }
    | {
          kind: 'textured';
          positions: ArrayBuffer;
          uvs: ArrayBuffer;
          indices: ArrayBuffer;
          atlas: Blob;
      }
    | { kind: 'metrics'; metrics: ScanMetrics }
    | { kind: 'done' }
    | { kind: 'export'; format: ExportFormat; blob: Blob }
    | { kind: 'error'; message: string; request: WorkerRequest['kind'] | null };
