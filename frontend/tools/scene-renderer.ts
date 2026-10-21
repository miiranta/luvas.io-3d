import { SCENE_GLSL, TARGET, rayDepth } from './synthetic-scene';

export interface CameraState {
    /** World → camera rotation, row-major, CV axes. */
    R: number[];
    center: number[];
    k: { fx: number; fy: number; cx: number; cy: number };
}

/** Camera at `center` looking at `target` with world +y up. */
export function lookAt(
    center: number[],
    target: number[] = TARGET,
): { R: number[]; center: number[] } {
    const z = normalize([target[0] - center[0], target[1] - center[1], target[2] - center[2]]);
    const x = normalize(cross(z, [0, 1, 0]));
    const y = cross(z, x);
    return { R: [...x, ...y, ...z], center };
}

/** Ray-casts the synthetic scene with WebGL2 into the bound framebuffer. */
export class SceneRenderer {
    private readonly program: WebGLProgram;
    private readonly vao: WebGLVertexArrayObject;

    constructor(private readonly gl: WebGL2RenderingContext) {
        const vertex = `#version 300 es
        void main() {
            vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        }`;
        const program = gl.createProgram();
        for (const [type, source] of [
            [gl.VERTEX_SHADER, vertex],
            [gl.FRAGMENT_SHADER, SCENE_GLSL],
        ] as const) {
            const shader = gl.createShader(type) as WebGLShader;
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                throw new Error(gl.getShaderInfoLog(shader) ?? 'scene shader failed');
            }
            gl.attachShader(program, shader);
        }
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program) ?? 'scene program failed');
        }
        this.program = program;
        this.vao = gl.createVertexArray() as WebGLVertexArrayObject;
    }

    draw(camera: CameraState, width: number, height: number): void {
        const gl = this.gl;
        gl.viewport(0, 0, width, height);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);
        const location = (name: string) => gl.getUniformLocation(this.program, name);
        const { k, R } = camera;
        gl.uniform4f(location('uK'), k.fx, k.fy, k.cx, k.cy);
        // Camera → world = Rᵀ; GLSL mat3 is column-major, so pass R's rows as columns.
        gl.uniformMatrix3fv(location('uCameraToWorld'), false, R);
        gl.uniform3fv(location('uCenter'), camera.center);
        gl.uniform2f(location('uSize'), width, height);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindVertexArray(null);
    }

    /** Renders and reads back an RGBA image with the first row at the top. */
    capture(camera: CameraState, width: number, height: number): Uint8Array {
        const gl = this.gl;
        this.draw(camera, width, height);
        const pixels = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        const flipped = new Uint8Array(pixels.length);
        const row = width * 4;
        for (let y = 0; y < height; y++) {
            flipped.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
        }
        return flipped;
    }
}

/** Depth grid in normalised view coordinates, like WebXR's getDepthInMeters. */
export function depthGrid(
    camera: CameraState,
    width: number,
    height: number,
    gridWidth: number,
    gridHeight: number,
    noise: () => number,
): Float32Array {
    const { R, center, k } = camera;
    const forward = [R[6], R[7], R[8]];
    const out = new Float32Array(gridWidth * gridHeight);
    for (let gy = 0; gy < gridHeight; gy++) {
        for (let gx = 0; gx < gridWidth; gx++) {
            const u = ((gx + 0.5) / gridWidth) * width - 0.5;
            const v = ((gy + 0.5) / gridHeight) * height - 0.5;
            const x = (u - k.cx) / k.fx;
            const y = (v - k.cy) / k.fy;
            const direction = [
                R[0] * x + R[3] * y + R[6],
                R[1] * x + R[4] * y + R[7],
                R[2] * x + R[5] * y + R[8],
            ];
            const d = rayDepth(center, direction, forward);
            out[gy * gridWidth + gx] = d > 0 ? d * (1 + noise()) : 0;
        }
    }
    return out;
}

/** WebXR-style camera → world matrix (column-major, OpenGL axes) of a CV pose. */
export function xrMatrix(R: number[], center: number[]): number[] {
    return [
        R[0],
        R[1],
        R[2],
        0,
        -R[3],
        -R[4],
        -R[5],
        0,
        -R[6],
        -R[7],
        -R[8],
        0,
        center[0],
        center[1],
        center[2],
        1,
    ];
}

function cross(a: number[], b: number[]): number[] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(a: number[]): number[] {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
}
