import { Injectable, effect, inject, signal } from '@angular/core';
import { CAMERA_STRIDE } from '../models/scan-results';
import { KeyframeMessage } from '../models/worker-protocol';
import { cameraLines } from '../../vision/export/camera-lines';
import { intrinsicsFromProjection } from '../../vision/geometry/camera';
import { GlScene } from '../../render/gl-scene';
import { multiply } from '../../render/mat4';
import { ScanService } from './scan-service';

export type Tracking = 'ok' | 'limited' | 'lost';

/** Long side of the keyframe images (texture resolution). */
const CAPTURE_LONG_SIDE = 1280;
const DEPTH_GRID = 64;
/** A keyframe every ~6° of view change or a baseline of 10 % of the scene depth. */
const KEYFRAME_ANGLE = (6 * Math.PI) / 180;
const KEYFRAME_BASELINE = 0.1;
const MAX_ANGULAR_SPEED = 0.5;
const MAX_LINEAR_SPEED = 0.35;
const MIN_INTERVAL_MS = 150;
const MAX_IN_FLIGHT = 2;
const PREVIEW_POINT = 0.012;

const BLIT_VS = `#version 300 es
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    // Row 0 of the target receives the top of the camera image (GL textures start at the bottom).
    vUv = vec2(p.x, 1.0 - p.y);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const BLIT_FS = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uCamera;
out vec4 color;
void main() {
    color = vec4(texture(uCamera, vUv).rgb, 1.0);
}`;

interface KeyPose {
    id: number;
    matrix: Float32Array;
}

interface Readback {
    id: number;
    fence: WebGLSync;
    width: number;
    height: number;
    message: Omit<KeyframeMessage, 'pixels'>;
}

/**
 * WebXR capture (Chrome on ARCore phones): the phone tracks its own pose (visual–inertial
 * odometry), the raw camera image is copied off the GPU asynchronously at keyframes, and the
 * live sparse/depth preview from the worker is drawn in AR over the scene.
 */
@Injectable({ providedIn: 'root' })
export class XrCapture {
    private readonly scan = inject(ScanService);
    private session: XRSession | null = null;
    private gl: WebGL2RenderingContext | null = null;
    private layer: XRWebGLLayer | null = null;
    private binding: XRWebGLBinding | null = null;
    private space: XRReferenceSpace | null = null;
    private scene: GlScene | null = null;
    private blit: WebGLProgram | null = null;
    private target: {
        framebuffer: WebGLFramebuffer;
        texture: WebGLTexture;
        pbo: WebGLBuffer;
        width: number;
        height: number;
    } | null = null;
    private readback: Readback | null = null;
    private keyPoses: KeyPose[] = [];
    private nextId = 0;
    private lastCapture = 0;
    private recordStart = 0;
    private previous: { matrix: Float32Array; time: number } | null = null;
    private stopping = false;
    private ended: (() => void) | null = null;

    readonly supported = signal<boolean | null>(null);
    readonly active = signal(false);
    readonly recording = signal(false);
    readonly tracking = signal<Tracking>('lost');
    readonly tooFast = signal(false);
    readonly depthSensing = signal(false);
    readonly elapsed = signal(0);
    readonly error = signal<string | null>(null);

    private angularSpeed = 0;
    private linearSpeed = 0;

    constructor() {
        this.scan.onVerdict((id, accepted) => {
            if (!accepted) this.keyPoses = this.keyPoses.filter((pose) => pose.id !== id);
        });
        effect(() => {
            const preview = this.scan.preview();
            this.scene?.setColored(
                'preview',
                preview && { positions: preview.positions, colors: preview.colors, channels: 3 },
            );
        });
    }

    async checkSupport(): Promise<void> {
        if (typeof window !== 'undefined' && !window.isSecureContext) {
            this.supported.set(false);
            this.error.set('the camera needs HTTPS (or localhost).');
            return;
        }
        try {
            this.supported.set((await navigator.xr?.isSessionSupported('immersive-ar')) ?? false);
        } catch {
            this.supported.set(false);
        }
    }

    /** Starts the AR session; `overlay` is shown on top of the camera while it runs. */
    async start(overlay: HTMLElement, onEnded: () => void): Promise<boolean> {
        this.error.set(null);
        const xr = navigator.xr;
        if (!xr) {
            this.error.set('WebXR is not available in this browser.');
            return false;
        }
        let session: XRSession;
        try {
            session = await xr.requestSession('immersive-ar', {
                requiredFeatures: ['local', 'camera-access'],
                optionalFeatures: ['dom-overlay', 'depth-sensing'],
                domOverlay: { root: overlay },
                depthSensing: {
                    usagePreference: ['cpu-optimized'],
                    dataFormatPreference: ['luminance-alpha', 'float32'],
                },
            });
        } catch (cause) {
            const name = cause instanceof DOMException ? cause.name : '';
            this.error.set(
                name === 'NotSupportedError'
                    ? 'this browser has AR but no camera access for it (needs Chrome on an ARCore phone).'
                    : `could not start AR${name ? ` (${name})` : ''}.`,
            );
            return false;
        }
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2', {
            xrCompatible: true,
            alpha: true,
            antialias: false,
        });
        if (!gl) {
            await session.end();
            this.error.set('WebGL 2 is not available.');
            return false;
        }
        this.session = session;
        this.gl = gl;
        this.layer = new XRWebGLLayer(session, gl, { alpha: true, antialias: false, depth: true });
        session.updateRenderState({ baseLayer: this.layer, depthNear: 0.02, depthFar: 30 });
        this.binding = new XRWebGLBinding(session, gl);
        this.space = await session.requestReferenceSpace('local');
        this.scene = new GlScene(gl);
        this.blit = this.compileBlit(gl);
        const preview = this.scan.preview();
        if (preview) this.scene.setColored('preview', { ...preview, channels: 3 });
        this.depthSensing.set(session.enabledFeatures?.includes('depth-sensing') ?? false);
        this.ended = onEnded;
        session.addEventListener('end', () => this.cleanup());
        this.active.set(true);
        this.tracking.set('lost');
        session.requestAnimationFrame(this.onFrame);
        return true;
    }

    startRecording(): void {
        if (!this.active() || this.recording()) return;
        this.recordStart = performance.now();
        this.keyPoses = [];
        this.stopping = false;
        this.recording.set(true);
        this.scan.beginCapture();
    }

    /** Stops recording once the last readback is out, then ends the AR session. */
    stopRecording(): void {
        if (!this.recording()) return;
        this.recording.set(false);
        this.stopping = true;
    }

    async end(): Promise<void> {
        await this.session?.end().catch(() => undefined);
    }

    private readonly onFrame = (time: number, frame: XRFrame): void => {
        const session = frame.session;
        session.requestAnimationFrame(this.onFrame);
        const gl = this.gl;
        const layer = this.layer;
        if (!gl || !layer || !this.space) return;
        this.collectReadback();
        if (this.stopping && !this.readback) {
            this.stopping = false;
            void session.end();
            return;
        }
        if (this.recording()) this.elapsed.set((performance.now() - this.recordStart) / 1000);

        gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        const pose = frame.getViewerPose(this.space);
        if (!pose) {
            this.tracking.set('lost');
            return;
        }
        this.tracking.set(pose.emulatedPosition ? 'limited' : 'ok');
        const view = pose.views[0];
        if (!view) return;
        this.updateMotion(view.transform.matrix, time);
        if (this.recording() && !pose.emulatedPosition && view.camera)
            this.maybeCapture(frame, view, view.camera);

        const viewport = layer.getViewport(view);
        if (!viewport || !this.scene) return;
        gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
        gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.disable(gl.DEPTH_TEST);
        const viewProjection = multiply(view.projectionMatrix, view.transform.inverse.matrix);
        const scale = (viewport.height / 2) * view.projectionMatrix[5];
        this.scene.drawPoints('preview', viewProjection, PREVIEW_POINT, scale, 0.85);
        this.scene.drawLines('keyframes', viewProjection, 0.9);
    };

    private updateMotion(matrix: Float32Array, time: number): void {
        const previous = this.previous;
        this.previous = { matrix: new Float32Array(matrix), time };
        if (!previous) return;
        const dt = Math.max(1e-3, (time - previous.time) / 1000);
        const linear =
            Math.hypot(
                matrix[12] - previous.matrix[12],
                matrix[13] - previous.matrix[13],
                matrix[14] - previous.matrix[14],
            ) / dt;
        const angular = rotationBetween(matrix, previous.matrix) / dt;
        this.linearSpeed += 0.3 * (linear - this.linearSpeed);
        this.angularSpeed += 0.3 * (angular - this.angularSpeed);
    }

    /** Novelty of a pose w.r.t. the keyframes so far (≥ 1 means "take a keyframe"). */
    private novelty(matrix: Float32Array): number {
        const depth = Math.max(0.15, this.scan.sceneDepth());
        let nearest = Infinity;
        for (const key of this.keyPoses) {
            const m = key.matrix;
            const baseline = Math.hypot(matrix[12] - m[12], matrix[13] - m[13], matrix[14] - m[14]);
            const change = Math.max(
                baseline / (KEYFRAME_BASELINE * depth),
                rotationBetween(matrix, m) / KEYFRAME_ANGLE,
            );
            nearest = Math.min(nearest, change);
        }
        return nearest;
    }

    private maybeCapture(frame: XRFrame, view: XRView, camera: XRCamera): void {
        const now = performance.now();
        if (
            this.readback ||
            this.scan.inFlight() >= MAX_IN_FLIGHT ||
            now - this.lastCapture < MIN_INTERVAL_MS
        )
            return;
        if (this.novelty(view.transform.matrix) < 1) {
            this.tooFast.set(false);
            return;
        }
        const fast = this.angularSpeed > MAX_ANGULAR_SPEED || this.linearSpeed > MAX_LINEAR_SPEED;
        this.tooFast.set(fast);
        if (fast) return;
        const texture = this.binding?.getCameraImage(camera);
        if (!texture) return;
        this.grab(frame, view, camera, texture);
        this.lastCapture = now;
    }

    private grab(frame: XRFrame, view: XRView, camera: XRCamera, texture: WebGLTexture): void {
        const gl = this.gl as WebGL2RenderingContext;
        const scale = Math.min(1, CAPTURE_LONG_SIDE / Math.max(camera.width, camera.height));
        const width = Math.round(camera.width * scale);
        const height = Math.round(camera.height * scale);
        const target = this.ensureTarget(width, height);

        gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
        gl.viewport(0, 0, width, height);
        gl.disable(gl.BLEND);
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(this.blit);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(gl.getUniformLocation(this.blit as WebGLProgram, 'uCamera'), 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, target.pbo);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) as WebGLSync;
        gl.flush();

        const id = this.nextId++;
        const matrix = new Float32Array(view.transform.matrix);
        this.keyPoses.push({ id, matrix });
        this.readback = {
            id,
            fence,
            width,
            height,
            message: {
                id,
                time: (performance.now() - this.recordStart) / 1000,
                width,
                height,
                intrinsics: intrinsicsFromProjection(view.projectionMatrix, width, height),
                transform: Array.from(matrix),
                depth: this.sampleDepth(frame, view, width, height),
            },
        };
        this.updateKeyframeLines();
    }

    /** Phone depth on a regular grid of normalised view coordinates (0 = no data). */
    private sampleDepth(
        frame: XRFrame,
        view: XRView,
        width: number,
        height: number,
    ): KeyframeMessage['depth'] {
        if (!this.depthSensing()) return null;
        let info: XRCPUDepthInformation | null = null;
        try {
            info = frame.getDepthInformation(view);
        } catch {
            return null;
        }
        if (!info) return null;
        const long = Math.max(width, height);
        const gw = Math.round((DEPTH_GRID * width) / long);
        const gh = Math.round((DEPTH_GRID * height) / long);
        const meters = new Float32Array(gw * gh);
        for (let y = 0; y < gh; y++) {
            for (let x = 0; x < gw; x++) {
                try {
                    const d = info.getDepthInMeters((x + 0.5) / gw, (y + 0.5) / gh);
                    meters[y * gw + x] = d > 0.05 && d < 8 ? d : 0;
                } catch {
                    meters[y * gw + x] = 0;
                }
            }
        }
        return { width: gw, height: gh, meters: meters.buffer };
    }

    private collectReadback(): void {
        const gl = this.gl;
        const readback = this.readback;
        if (!gl || !readback || !this.target) return;
        const status = gl.clientWaitSync(readback.fence, 0, 0);
        if (status === gl.TIMEOUT_EXPIRED) return;
        gl.deleteSync(readback.fence);
        this.readback = null;
        const pixels = new Uint8Array(readback.width * readback.height * 4);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.target.pbo);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, pixels);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        this.scan.sendKeyframe({ ...readback.message, pixels: pixels.buffer });
    }

    private ensureTarget(width: number, height: number) {
        const gl = this.gl as WebGL2RenderingContext;
        if (this.target && this.target.width === width && this.target.height === height)
            return this.target;
        if (this.target) {
            gl.deleteFramebuffer(this.target.framebuffer);
            gl.deleteTexture(this.target.texture);
            gl.deleteBuffer(this.target.pbo);
        }
        const texture = gl.createTexture() as WebGLTexture;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
        const framebuffer = gl.createFramebuffer() as WebGLFramebuffer;
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        const pbo = gl.createBuffer() as WebGLBuffer;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, width * height * 4, gl.STREAM_READ);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        this.target = { framebuffer, texture, pbo, width, height };
        return this.target;
    }

    /** Small frustums of the keyframes taken so far, drawn in AR. */
    private updateKeyframeLines(): void {
        const cameras = new Float32Array(this.keyPoses.length * CAMERA_STRIDE);
        this.keyPoses.forEach((key, i) => {
            const m = key.matrix;
            // World → CV camera rotation rows: x = col0, y = −col1, z = −col2.
            cameras.set(
                [m[0], m[1], m[2], -m[4], -m[5], -m[6], -m[8], -m[9], -m[10]],
                i * CAMERA_STRIDE,
            );
            cameras.set([m[12], m[13], m[14], 0.5, 0.5], i * CAMERA_STRIDE + 9);
        });
        const lines = cameraLines(cameras, 0.03);
        this.scene?.setColored('keyframes', { ...lines, channels: 4 });
    }

    private compileBlit(gl: WebGL2RenderingContext): WebGLProgram {
        const program = gl.createProgram();
        for (const [type, source] of [
            [gl.VERTEX_SHADER, BLIT_VS],
            [gl.FRAGMENT_SHADER, BLIT_FS],
        ] as const) {
            const shader = gl.createShader(type) as WebGLShader;
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            gl.attachShader(program, shader);
        }
        gl.linkProgram(program);
        return program;
    }

    private cleanup(): void {
        this.scene?.dispose();
        this.scene = null;
        this.session = null;
        this.gl = null;
        this.layer = null;
        this.binding = null;
        this.space = null;
        this.target = null;
        this.readback = null;
        this.previous = null;
        this.active.set(false);
        this.recording.set(false);
        this.tooFast.set(false);
        this.tracking.set('lost');
        const ended = this.ended;
        this.ended = null;
        ended?.();
    }
}

/** Angle (rad) between the rotation parts of two column-major rigid transforms. */
function rotationBetween(a: Float32Array, b: Float32Array): number {
    let trace = 0;
    for (let c = 0; c < 3; c++) {
        trace += a[c * 4] * b[c * 4] + a[c * 4 + 1] * b[c * 4 + 1] + a[c * 4 + 2] * b[c * 4 + 2];
    }
    return Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2)));
}
