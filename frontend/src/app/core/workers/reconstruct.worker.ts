/// <reference lib="webworker" />

import { ExportFormat, WorkerRequest, WorkerResponse } from '../models/worker-protocol';
import { cameraLines } from '../../vision/export/camera-lines';
import { writeGlb } from '../../vision/export/glb-writer';
import { writePly } from '../../vision/export/ply-writer';
import { GpuContext, requestGpu } from '../../vision/gpu/gpu-device';
import { Reconstruction, reconstruct } from '../../vision/pipeline/reconstructor';
import { JpegImageStore } from '../../vision/pipeline/jpeg-image-store';
import { ScanSession } from '../../vision/pipeline/scan-session';

let session = new ScanSession(new JpegImageStore());
let result: Reconstruction | null = null;
let gpu: Promise<GpuContext> | null = null;
let queue: Promise<void> = Promise.resolve();

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
    self.postMessage(message, transfer);
}

function copy(array: ArrayBufferView): ArrayBuffer {
    return new Uint8Array(array.buffer, array.byteOffset, array.byteLength).slice().buffer;
}

function messageOf(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function handle(request: WorkerRequest): Promise<void> {
    switch (request.kind) {
        case 'keyframe': {
            const frame = request.frame;
            const verdict = await session.addKeyframe({
                time: frame.time,
                image: {
                    width: frame.width,
                    height: frame.height,
                    data: new Uint8Array(frame.pixels),
                },
                intrinsics: frame.intrinsics,
                xrTransform: frame.transform,
                depth: frame.depth && {
                    width: frame.depth.width,
                    height: frame.depth.height,
                    meters: new Float32Array(frame.depth.meters),
                },
            });
            post({
                kind: 'keyframe',
                id: frame.id,
                accepted: verdict.accepted,
                reason: verdict.reason ?? null,
                keyframes: session.keyframes.length,
                sparsePoints: session.map.points(2).ids.length,
                sceneDepth: session.sceneDepth(),
            });
            if (verdict.accepted) {
                const cloud = session.previewCloud();
                const positions = copy(cloud.positions);
                const colors = copy(cloud.colors);
                post({ kind: 'preview', positions, colors }, [positions, colors]);
            }
            break;
        }
        case 'finish': {
            const context = await (gpu ??= requestGpu());
            result = await reconstruct(session, context, {
                stage: (event) => post({ kind: 'stage', event }),
                cameras: (cameras) => {
                    const buffer = copy(cameras);
                    post({ kind: 'cameras', cameras: buffer }, [buffer]);
                },
                sparse: (positions, colors) => {
                    const p = copy(positions);
                    const c = copy(colors);
                    post({ kind: 'sparse', positions: p, colors: c }, [p, c]);
                },
                dense: (cloud) => {
                    const p = copy(cloud.positions);
                    const c = copy(cloud.colors);
                    post({ kind: 'dense', positions: p, colors: c }, [p, c]);
                },
                mesh: (mesh) => {
                    const p = copy(mesh.positions);
                    const n = copy(mesh.normals);
                    const i = copy(mesh.indices);
                    post({ kind: 'mesh', positions: p, normals: n, indices: i }, [p, n, i]);
                },
            });
            const mesh = result.mesh;
            const positions = copy(mesh.positions);
            const uvs = copy(mesh.uvs);
            const indices = copy(mesh.indices);
            post(
                {
                    kind: 'textured',
                    positions,
                    uvs,
                    indices,
                    atlas: new Blob([result.atlasJpeg as Uint8Array<ArrayBuffer>], {
                        type: 'image/jpeg',
                    }),
                },
                [positions, uvs, indices],
            );
            post({ kind: 'metrics', metrics: result.metrics });
            post({ kind: 'done' });
            break;
        }
        case 'reset': {
            session = new ScanSession(new JpegImageStore());
            result = null;
            break;
        }
        case 'export': {
            if (!result) throw new Error('nothing reconstructed yet');
            post({
                kind: 'export',
                format: request.format,
                blob: exportBlob(result, request.format),
            });
            break;
        }
    }
}

function exportBlob(scan: Reconstruction, format: ExportFormat): Blob {
    switch (format) {
        case 'glb': {
            const extent = Math.max(...scan.metrics.extent, 0.1);
            const lines = cameraLines(scan.cameras, extent * 0.03);
            const sparseColors = new Uint8Array((scan.sparse.positions.length / 3) * 4);
            for (let i = 0; i < scan.sparse.positions.length / 3; i++) {
                sparseColors.set(scan.sparse.colors.subarray(i * 3, i * 3 + 3), i * 4);
                sparseColors[i * 4 + 3] = 255;
            }
            const glb = writeGlb([
                {
                    name: 'mesh',
                    mode: 4,
                    positions: scan.mesh.positions,
                    uvs: scan.mesh.uvs,
                    indices: scan.mesh.indices,
                    texture: { bytes: scan.atlasJpeg, mimeType: 'image/jpeg' },
                },
                { name: 'cameras', mode: 1, positions: lines.positions, colors: lines.colors },
                { name: 'sparse', mode: 0, positions: scan.sparse.positions, colors: sparseColors },
            ]);
            return new Blob([glb], { type: 'model/gltf-binary' });
        }
        case 'dense':
            return new Blob(
                [writePly(scan.dense.positions, scan.dense.colors, scan.dense.normals)],
                {
                    type: 'application/octet-stream',
                },
            );
        case 'sparse':
            return new Blob([writePly(scan.sparse.positions, scan.sparse.colors)], {
                type: 'application/octet-stream',
            });
        case 'metrics':
            return new Blob([JSON.stringify(scan.metrics, null, 2)], { type: 'application/json' });
    }
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
    const request = event.data;
    queue = queue.then(() =>
        handle(request).catch((error) =>
            post({ kind: 'error', message: messageOf(error), request: request.kind }),
        ),
    );
});

gpu = requestGpu();
gpu.then(
    (context) => post({ kind: 'ready', gpu: context.label, error: null }),
    (error) => {
        gpu = null;
        post({ kind: 'ready', gpu: null, error: messageOf(error) });
    },
);
