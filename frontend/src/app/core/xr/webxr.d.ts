// Minimal WebXR typings for what the app uses: immersive AR, DOM overlay, raw camera access
// (XRWebGLBinding.getCameraImage) and CPU depth sensing. TypeScript's DOM lib has none of these.

interface XRSystem {
    isSessionSupported(mode: 'immersive-ar'): Promise<boolean>;
    requestSession(mode: 'immersive-ar', init?: XRSessionInit): Promise<XRSession>;
}

interface Navigator {
    readonly xr?: XRSystem;
}

interface XRSessionInit {
    requiredFeatures?: string[];
    optionalFeatures?: string[];
    domOverlay?: { root: Element };
    depthSensing?: { usagePreference: string[]; dataFormatPreference: string[] };
}

type XRFrameRequestCallback = (time: DOMHighResTimeStamp, frame: XRFrame) => void;

interface XRSession extends EventTarget {
    readonly enabledFeatures?: readonly string[];
    readonly depthUsage?: string;
    readonly depthDataFormat?: string;
    updateRenderState(state: {
        baseLayer?: XRWebGLLayer;
        depthNear?: number;
        depthFar?: number;
    }): void;
    requestReferenceSpace(type: 'local' | 'viewer'): Promise<XRReferenceSpace>;
    requestAnimationFrame(callback: XRFrameRequestCallback): number;
    cancelAnimationFrame(handle: number): void;
    end(): Promise<void>;
}

type XRReferenceSpace = EventTarget;

interface XRFrame {
    readonly session: XRSession;
    getViewerPose(space: XRReferenceSpace): XRViewerPose | null;
    getDepthInformation(view: XRView): XRCPUDepthInformation | null;
}

interface XRViewerPose {
    readonly views: readonly XRView[];
    readonly transform: XRRigidTransform;
    readonly emulatedPosition: boolean;
}

interface XRView {
    readonly projectionMatrix: Float32Array;
    readonly transform: XRRigidTransform;
    readonly camera?: XRCamera | null;
}

interface XRCamera {
    readonly width: number;
    readonly height: number;
}

interface XRRigidTransform {
    readonly matrix: Float32Array;
    readonly inverse: XRRigidTransform;
}

interface XRCPUDepthInformation {
    readonly width: number;
    readonly height: number;
    getDepthInMeters(x: number, y: number): number;
}

interface XRViewport {
    x: number;
    y: number;
    width: number;
    height: number;
}

declare class XRWebGLLayer {
    constructor(
        session: XRSession,
        context: WebGL2RenderingContext,
        init?: { alpha?: boolean; antialias?: boolean; depth?: boolean },
    );
    readonly framebuffer: WebGLFramebuffer | null;
    readonly framebufferWidth: number;
    readonly framebufferHeight: number;
    getViewport(view: XRView): XRViewport | null;
}

declare class XRWebGLBinding {
    constructor(session: XRSession, context: WebGL2RenderingContext);
    getCameraImage(camera: XRCamera): WebGLTexture | null;
}
