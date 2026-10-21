/**
 * A stand-in WebXR runtime for `npm run e2e`, injected before the app loads. It implements the
 * parts the app uses — immersive-ar sessions, XRWebGLLayer, raw camera access through
 * XRWebGLBinding.getCameraImage and CPU depth — and plays a person orbiting the synthetic scene
 * twice (two heights) while the "phone odometry" drifts slowly.
 */
import { CameraState, SceneRenderer, lookAt, xrMatrix } from './scene-renderer';
import { rayDepth } from './synthetic-scene';

const WIDTH = 720;
const HEIGHT = 1280;
const K = { fx: 1000, fy: 1000, cx: 359.5, cy: 639.5 };
const RING_SECONDS = 16;
const RINGS = [
    { height: 0.2, radius: 0.42 },
    { height: 0.36, radius: 0.36 },
];

interface FakeState {
    session: FakeSession | null;
    started: number;
    frames: number;
}

declare global {
    interface Window {
        __fakeXr: FakeState;
    }
}

const state: FakeState = { session: null, started: 0, frames: 0 };
window.__fakeXr = state;

/** True pose at time t (s): ring 1 then ring 2, looking at the objects. */
function truePose(t: number): { R: number[]; center: number[] } {
    const ring = Math.min(RINGS.length - 1, Math.floor(t / RING_SECONDS));
    const phase = (Math.min(t, RING_SECONDS * RINGS.length) - ring * RING_SECONDS) / RING_SECONDS;
    const angle = phase * Math.PI * 2 + ring * 0.3;
    const { height, radius } = RINGS[ring];
    // A little hand shake on top of the orbit.
    const shake = 0.004 * Math.sin(t * 7.3);
    return lookAt([Math.sin(angle) * radius, height + shake, Math.cos(angle) * radius]);
}

/** Odometry drift: a slow yaw about the world origin plus a translation, growing with time. */
function phonePose(t: number): { R: number[]; center: number[] } {
    const pose = truePose(t);
    const yaw = t * 0.0002;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const D = [c, 0, s, 0, 1, 0, -s, 0, c];
    const R = [0, 1, 2].flatMap((r) =>
        [0, 1, 2].map(
            (col) =>
                pose.R[r * 3] * D[col * 3] +
                pose.R[r * 3 + 1] * D[col * 3 + 1] +
                pose.R[r * 3 + 2] * D[col * 3 + 2],
        ),
    );
    const offset = [t * 0.00025, -t * 0.0001, t * 0.00015];
    const center = [0, 1, 2].map(
        (k) =>
            D[k * 3] * pose.center[0] +
            D[k * 3 + 1] * pose.center[1] +
            D[k * 3 + 2] * pose.center[2] +
            offset[k],
    );
    return { R, center };
}

function projection(near: number, far: number): Float32Array {
    const p = new Float32Array(16);
    p[0] = (2 * K.fx) / WIDTH;
    p[5] = (2 * K.fy) / HEIGHT;
    p[8] = 1 - (2 * (K.cx + 0.5)) / WIDTH;
    p[9] = (2 * (K.cy + 0.5)) / HEIGHT - 1;
    p[10] = -(far + near) / (far - near);
    p[11] = -1;
    p[14] = (-2 * far * near) / (far - near);
    return p;
}

function rigid(matrix: number[]) {
    const m = new Float32Array(matrix);
    // Inverse of a rigid transform: [Rᵀ | −Rᵀ t].
    const inverse = new Float32Array(16);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) inverse[c * 4 + r] = m[r * 4 + c];
        inverse[12 + r] = -(m[r * 4] * m[12] + m[r * 4 + 1] * m[13] + m[r * 4 + 2] * m[14]);
    }
    inverse[15] = 1;
    return { matrix: m, inverse: { matrix: inverse, inverse: null as unknown } };
}

class FakeSession extends EventTarget {
    readonly enabledFeatures = ['local', 'camera-access', 'dom-overlay', 'depth-sensing'];
    readonly depthUsage = 'cpu-optimized';
    readonly depthDataFormat = 'luminance-alpha';
    private callbacks = new Map<number, XRFrameRequestCallback>();
    private nextHandle = 1;
    private timer: number | null = null;
    private ended = false;
    renderer: SceneRenderer | null = null;
    cameraTexture: WebGLTexture | null = null;
    private framebuffer: WebGLFramebuffer | null = null;
    gl: WebGL2RenderingContext | null = null;
    current = 0;

    updateRenderState(): void {}

    async requestReferenceSpace(): Promise<EventTarget> {
        return new EventTarget();
    }

    requestAnimationFrame(callback: XRFrameRequestCallback): number {
        const handle = this.nextHandle++;
        this.callbacks.set(handle, callback);
        this.timer ??= window.setTimeout(() => this.tick(), 33);
        return handle;
    }

    cancelAnimationFrame(handle: number): void {
        this.callbacks.delete(handle);
    }

    async end(): Promise<void> {
        if (this.ended) return;
        this.ended = true;
        if (this.timer !== null) clearTimeout(this.timer);
        state.session = null;
        this.dispatchEvent(new Event('end'));
    }

    private tick(): void {
        this.timer = null;
        if (this.ended) return;
        this.current = (performance.now() - state.started) / 1000;
        state.frames++;
        const callbacks = [...this.callbacks.values()];
        this.callbacks.clear();
        const frame = new FakeFrame(this, this.current);
        for (const callback of callbacks) callback(performance.now(), frame as unknown as XRFrame);
    }

    /** Ray-casts the scene for the current true pose into a texture of the app's context. */
    renderCamera(): WebGLTexture | null {
        const gl = this.gl;
        if (!gl) return null;
        if (!this.renderer) {
            this.renderer = new SceneRenderer(gl);
            this.cameraTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.cameraTexture);
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, WIDTH, HEIGHT);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            this.framebuffer = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                this.cameraTexture,
                0,
            );
        }
        const previous = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        const camera: CameraState = { ...truePose(this.current), k: K };
        this.renderer.draw(camera, WIDTH, HEIGHT);
        gl.bindFramebuffer(gl.FRAMEBUFFER, previous);
        return this.cameraTexture;
    }
}

class FakeFrame {
    constructor(
        readonly session: FakeSession,
        private readonly time: number,
    ) {}

    getViewerPose() {
        const phone = phonePose(this.time);
        const transform = rigid(xrMatrix(phone.R, phone.center));
        const view = {
            projectionMatrix: projection(0.02, 30),
            transform,
            camera: { width: WIDTH, height: HEIGHT },
        };
        return { views: [view], transform, emulatedPosition: false };
    }

    getDepthInformation() {
        const truth = truePose(this.time);
        const { R, center } = truth;
        const forward = [R[6], R[7], R[8]];
        return {
            width: 160,
            height: 90,
            getDepthInMeters: (u: number, v: number) => {
                const px = u * WIDTH - 0.5;
                const py = v * HEIGHT - 0.5;
                const x = (px - K.cx) / K.fx;
                const y = (py - K.cy) / K.fy;
                const direction = [
                    R[0] * x + R[3] * y + R[6],
                    R[1] * x + R[4] * y + R[7],
                    R[2] * x + R[5] * y + R[8],
                ];
                const d = rayDepth(center, direction, forward);
                // Phone depth is smooth but biased and noisy at the centimetre level.
                return d > 0 ? d * (1 + 0.01 * Math.sin(px * 0.05 + py * 0.031)) : 0;
            },
        };
    }
}

class FakeLayer {
    readonly framebuffer = null;
    constructor(
        readonly session: FakeSession,
        readonly gl: WebGL2RenderingContext,
    ) {
        gl.canvas.width = WIDTH / 2;
        gl.canvas.height = HEIGHT / 2;
        session.gl = gl;
    }
    get framebufferWidth(): number {
        return this.gl.canvas.width;
    }
    get framebufferHeight(): number {
        return this.gl.canvas.height;
    }
    getViewport() {
        return { x: 0, y: 0, width: this.gl.canvas.width, height: this.gl.canvas.height };
    }
}

class FakeBinding {
    constructor(
        readonly session: FakeSession,
        gl: WebGL2RenderingContext,
    ) {
        session.gl = gl;
    }
    getCameraImage(): WebGLTexture | null {
        return this.session.renderCamera();
    }
}

Object.defineProperty(navigator, 'xr', {
    configurable: true,
    value: {
        isSessionSupported: async () => true,
        requestSession: async (_mode: string, init: XRSessionInit) => {
            if (!init.requiredFeatures?.includes('camera-access'))
                throw new DOMException('no camera', 'NotSupportedError');
            const session = new FakeSession();
            state.session = session;
            state.started = performance.now();
            return session;
        },
    },
});
Object.assign(window, { XRWebGLLayer: FakeLayer, XRWebGLBinding: FakeBinding });
