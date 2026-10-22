import { Injectable, effect, inject, signal } from '@angular/core';
import { CAMERA_STRIDE } from '../models/scan-results';
import { KeyframeMessage } from '../models/worker-protocol';
import { CameraReadback } from '../xr/camera-readback';
import { sampleDepth } from '../xr/depth-sampler';
import { KeyframePolicy } from '../xr/keyframe-policy';
import { cameraLines } from '../../vision/export/camera-lines';
import { intrinsicsFromProjection } from '../../vision/geometry/camera';
import { GlScene } from '../../render/gl-scene';
import { multiply } from '../../render/mat4';
import { ScanService } from './scan-service';

export type Tracking = 'ok' | 'limited' | 'lost';

/** Long side of the keyframe images (texture resolution). */
const CAPTURE_LONG_SIDE = 1280;
const MIN_INTERVAL_MS = 150;
const MAX_IN_FLIGHT = 2;
const PREVIEW_POINT = 0.012;

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
    private readback: CameraReadback | null = null;
    /** The keyframe whose pixels are being read back. */
    private pending: Omit<KeyframeMessage, 'pixels'> | null = null;
    private readonly policy = new KeyframePolicy();
    private nextId = 0;
    private lastCapture = 0;
    private recordStart = 0;
    private stopping = false;
    /** Tracking segment sent with each keyframe; increments after tracking loss or a jump. */
    private segment = 0;
    private trackingDropped = false;
    private ended: (() => void) | null = null;

    readonly supported = signal<boolean | null>(null);
    readonly active = signal(false);
    readonly recording = signal(false);
    readonly tracking = signal<Tracking>('lost');
    readonly tooFast = signal(false);
    readonly depthSensing = signal(false);
    readonly elapsed = signal(0);
    readonly error = signal<string | null>(null);

    constructor() {
        this.scan.onVerdict((id, accepted) => {
            if (!accepted) this.policy.forget(id);
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
        this.readback = new CameraReadback(gl);
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
        this.policy.reset();
        this.segment = 0;
        this.trackingDropped = false;
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
        if (this.stopping && !this.pending) {
            this.stopping = false;
            void session.end();
            return;
        }
        if (this.recording()) this.elapsed.set((performance.now() - this.recordStart) / 1000);

        gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        const pose = frame.getViewerPose(this.space);
        if (!pose || pose.emulatedPosition) {
            this.tracking.set(pose ? 'limited' : 'lost');
            this.trackingDropped = true;
            if (!pose) return;
        } else {
            this.tracking.set('ok');
            // Poses after a tracking loss may have jumped: start a new odometry segment.
            if (this.trackingDropped) this.segment++;
            this.trackingDropped = false;
        }
        const view = pose.views[0];
        if (!view) return;
        if (this.policy.move(view.transform.matrix, time)) this.segment++;
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

    private maybeCapture(frame: XRFrame, view: XRView, camera: XRCamera): void {
        const now = performance.now();
        if (
            this.scan.full() ||
            this.pending ||
            this.scan.inFlight() >= MAX_IN_FLIGHT ||
            now - this.lastCapture < MIN_INTERVAL_MS
        )
            return;
        const { take, tooFast } = this.policy.decide(view.transform.matrix, this.scan.sceneDepth());
        this.tooFast.set(tooFast);
        if (!take) return;
        const texture = this.binding?.getCameraImage(camera);
        if (!texture) return;
        this.grab(frame, view, camera, texture);
        this.lastCapture = now;
    }

    private grab(frame: XRFrame, view: XRView, camera: XRCamera, texture: WebGLTexture): void {
        const scale = Math.min(1, CAPTURE_LONG_SIDE / Math.max(camera.width, camera.height));
        const width = Math.round(camera.width * scale);
        const height = Math.round(camera.height * scale);
        this.readback?.start(texture, width, height);

        const id = this.nextId++;
        const matrix = new Float32Array(view.transform.matrix);
        this.policy.poses.push({ id, matrix });
        this.pending = {
            id,
            time: (performance.now() - this.recordStart) / 1000,
            width,
            height,
            intrinsics: intrinsicsFromProjection(view.projectionMatrix, width, height),
            transform: Array.from(matrix),
            depth: this.depthSensing() ? sampleDepth(frame, view, width, height) : null,
            segment: this.segment,
        };
        this.updateKeyframeLines();
    }

    private collectReadback(): void {
        const pixels = this.pending && this.readback?.collect();
        if (!pixels || !this.pending) return;
        this.scan.sendKeyframe({ ...this.pending, pixels: pixels.buffer as ArrayBuffer });
        this.pending = null;
    }

    /** Small frustums of the keyframes taken so far, drawn in AR. */
    private updateKeyframeLines(): void {
        const poses = this.policy.poses;
        const cameras = new Float32Array(poses.length * CAMERA_STRIDE);
        poses.forEach((key, i) => {
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

    private cleanup(): void {
        this.scene?.dispose();
        this.scene = null;
        this.session = null;
        this.gl = null;
        this.layer = null;
        this.binding = null;
        this.space = null;
        this.readback = null;
        this.pending = null;
        this.policy.stop();
        this.active.set(false);
        this.recording.set(false);
        this.tooFast.set(false);
        this.tracking.set('lost');
        const ended = this.ended;
        this.ended = null;
        ended?.();
    }
}
