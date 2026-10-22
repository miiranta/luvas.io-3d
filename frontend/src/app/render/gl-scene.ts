import { Mat4 } from './mat4';

const POINT_VS = `#version 300 es
in vec3 aPosition;
in vec4 aColor;
uniform mat4 uViewProjection;
uniform float uSize;
uniform float uScale;
out vec4 vColor;
void main() {
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
    gl_PointSize = clamp(uSize * uScale / max(gl_Position.w, 1e-3), 1.5, 14.0);
    vColor = aColor;
}`;

const POINT_FS = `#version 300 es
precision mediump float;
in vec4 vColor;
uniform float uAlpha;
out vec4 color;
void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;
    color = vec4(vColor.rgb, vColor.a * uAlpha);
}`;

const LINE_VS = `#version 300 es
in vec3 aPosition;
in vec4 aColor;
uniform mat4 uViewProjection;
out vec4 vColor;
void main() {
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
    vColor = aColor;
}`;

const LINE_FS = `#version 300 es
precision mediump float;
in vec4 vColor;
uniform float uAlpha;
out vec4 color;
void main() {
    color = vec4(vColor.rgb, vColor.a * uAlpha);
}`;

const MESH_VS = `#version 300 es
in vec3 aPosition;
in vec4 aColor;
in vec2 aUv;
uniform mat4 uViewProjection;
out vec2 vUv;
out vec3 vWorld;
out vec4 vColor;
void main() {
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
    vUv = aUv;
    vWorld = aPosition;
    vColor = aColor;
}`;

const MESH_FS = `#version 300 es
precision highp float;
in vec2 vUv;
in vec3 vWorld;
in vec4 vColor;
uniform sampler2D uAtlas;
/** 0 = shaded preview, 1 = baked atlas, 2 = vertex colours. */
uniform int uMode;
uniform vec3 uEye;
out vec4 color;
void main() {
    if (uMode == 1) {
        color = vec4(texture(uAtlas, vUv).rgb, 1.0);
        return;
    }
    if (uMode == 2) {
        color = vec4(vColor.rgb, 1.0);
        return;
    }
    // Untextured preview: flat normal from screen derivatives, headlight shading.
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    float light = abs(dot(n, normalize(uEye - vWorld)));
    color = vec4(mix(vec3(0.17, 0.11, 0.33), vec3(1.0, 0.85, 0.55), 0.15 + 0.85 * light), 1.0);
}`;

interface Program {
    program: WebGLProgram;
    uniforms: Map<string, WebGLUniformLocation | null>;
}

/** GPU buffers of one drawable. */
interface Drawable {
    vao: WebGLVertexArrayObject;
    buffers: WebGLBuffer[];
    count: number;
    uvs?: boolean;
    colors?: boolean;
}

export interface ColoredGeometry {
    positions: Float32Array;
    /** RGB8 or RGBA8 per vertex. */
    colors: Uint8Array;
    channels: 3 | 4;
}

/**
 * Small WebGL2 renderer shared by the AR overlay and the model viewer: coloured round points,
 * coloured lines and a mesh that is either textured (baked atlas, unlit) or shaded.
 */
export class GlScene {
    private readonly points: Program;
    private readonly lines: Program;
    private readonly mesh: Program;
    private readonly drawables = new Map<string, Drawable>();
    private atlas: WebGLTexture | null = null;

    constructor(readonly gl: WebGL2RenderingContext) {
        this.points = this.compile(POINT_VS, POINT_FS);
        this.lines = this.compile(LINE_VS, LINE_FS);
        this.mesh = this.compile(MESH_VS, MESH_FS);
    }

    private compile(vertex: string, fragment: string): Program {
        const gl = this.gl;
        const program = gl.createProgram();
        for (const [type, source] of [
            [gl.VERTEX_SHADER, vertex],
            [gl.FRAGMENT_SHADER, fragment],
        ] as const) {
            const shader = gl.createShader(type) as WebGLShader;
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                throw new Error(gl.getShaderInfoLog(shader) ?? 'shader compilation failed');
            }
            gl.attachShader(program, shader);
        }
        gl.bindAttribLocation(program, 0, 'aPosition');
        gl.bindAttribLocation(program, 1, 'aColor');
        gl.bindAttribLocation(program, 2, 'aUv');
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(program) ?? 'program link failed');
        }
        return { program, uniforms: new Map() };
    }

    private uniform(program: Program, name: string): WebGLUniformLocation | null {
        if (!program.uniforms.has(name)) {
            program.uniforms.set(name, this.gl.getUniformLocation(program.program, name));
        }
        return program.uniforms.get(name) ?? null;
    }

    /** Uploads (or replaces) a point or line set under `name`. */
    setColored(name: string, geometry: ColoredGeometry | null): void {
        this.remove(name);
        if (!geometry || geometry.positions.length === 0) return;
        const gl = this.gl;
        const vao = gl.createVertexArray() as WebGLVertexArrayObject;
        gl.bindVertexArray(vao);
        const position = this.attribute(0, geometry.positions, 3, gl.FLOAT, false);
        const color = this.attribute(1, geometry.colors, geometry.channels, gl.UNSIGNED_BYTE, true);
        if (geometry.channels === 3) gl.vertexAttrib4f(1, 1, 1, 1, 1);
        gl.bindVertexArray(null);
        this.drawables.set(name, {
            vao,
            buffers: [position, color],
            count: geometry.positions.length / 3,
        });
    }

    /** Uploads a mesh (positions + optional atlas UVs or RGBA8 vertex colours). */
    setMesh(
        name: string,
        positions: Float32Array,
        indices: Uint32Array,
        uvs: Float32Array | null,
        colors: Uint8Array | null = null,
    ): void {
        this.remove(name);
        const gl = this.gl;
        const vao = gl.createVertexArray() as WebGLVertexArrayObject;
        gl.bindVertexArray(vao);
        const buffers = [this.attribute(0, positions, 3, gl.FLOAT, false)];
        if (uvs) buffers.push(this.attribute(2, uvs, 2, gl.FLOAT, false));
        if (colors) buffers.push(this.attribute(1, colors, 4, gl.UNSIGNED_BYTE, true));
        const index = gl.createBuffer() as WebGLBuffer;
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices as Uint32Array<ArrayBuffer>, gl.STATIC_DRAW);
        buffers.push(index);
        gl.bindVertexArray(null);
        this.drawables.set(name, {
            vao,
            buffers,
            count: indices.length,
            uvs: uvs !== null,
            colors: colors !== null,
        });
    }

    setAtlas(image: ImageBitmap | null): void {
        const gl = this.gl;
        if (this.atlas) gl.deleteTexture(this.atlas);
        this.atlas = null;
        if (!image) return;
        this.atlas = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.atlas);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    has(name: string): boolean {
        return this.drawables.has(name);
    }

    remove(name: string): void {
        const drawable = this.drawables.get(name);
        if (!drawable) return;
        for (const buffer of drawable.buffers) this.gl.deleteBuffer(buffer);
        this.gl.deleteVertexArray(drawable.vao);
        this.drawables.delete(name);
    }

    private attribute(
        location: number,
        data: ArrayBufferView,
        size: number,
        type: number,
        normalized: boolean,
    ): WebGLBuffer {
        const gl = this.gl;
        const buffer = gl.createBuffer() as WebGLBuffer;
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data as ArrayBufferView<ArrayBuffer>, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, size, type, normalized, 0, 0);
        return buffer;
    }

    /**
     * Draws points; `size` is the world-space diameter and `scale` the pixels per metre at unit
     * distance (viewport height / (2·tan(fov/2))).
     */
    drawPoints(name: string, viewProjection: Mat4, size: number, scale: number, alpha = 1): void {
        const drawable = this.drawables.get(name);
        if (!drawable) return;
        const gl = this.gl;
        gl.useProgram(this.points.program);
        gl.uniformMatrix4fv(this.uniform(this.points, 'uViewProjection'), false, viewProjection);
        gl.uniform1f(this.uniform(this.points, 'uSize'), size);
        gl.uniform1f(this.uniform(this.points, 'uScale'), scale);
        gl.uniform1f(this.uniform(this.points, 'uAlpha'), alpha);
        gl.bindVertexArray(drawable.vao);
        gl.drawArrays(gl.POINTS, 0, drawable.count);
        gl.bindVertexArray(null);
    }

    drawLines(name: string, viewProjection: Mat4, alpha = 1): void {
        const drawable = this.drawables.get(name);
        if (!drawable) return;
        const gl = this.gl;
        gl.useProgram(this.lines.program);
        gl.uniformMatrix4fv(this.uniform(this.lines, 'uViewProjection'), false, viewProjection);
        gl.uniform1f(this.uniform(this.lines, 'uAlpha'), alpha);
        gl.bindVertexArray(drawable.vao);
        gl.drawArrays(gl.LINES, 0, drawable.count);
        gl.bindVertexArray(null);
    }

    /** Draws a mesh with its atlas, its vertex colours, or shaded when it has neither. */
    drawMesh(name: string, viewProjection: Mat4, eye: number[]): void {
        const drawable = this.drawables.get(name);
        if (!drawable) return;
        const gl = this.gl;
        gl.useProgram(this.mesh.program);
        gl.uniformMatrix4fv(this.uniform(this.mesh, 'uViewProjection'), false, viewProjection);
        gl.uniform3fv(this.uniform(this.mesh, 'uEye'), eye);
        const useAtlas = drawable.uvs === true && this.atlas !== null;
        gl.uniform1i(this.uniform(this.mesh, 'uMode'), useAtlas ? 1 : drawable.colors ? 2 : 0);
        if (useAtlas) {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.atlas);
            gl.uniform1i(this.uniform(this.mesh, 'uAtlas'), 0);
        }
        gl.bindVertexArray(drawable.vao);
        gl.drawElements(gl.TRIANGLES, drawable.count, gl.UNSIGNED_INT, 0);
        gl.bindVertexArray(null);
    }

    dispose(): void {
        for (const name of [...this.drawables.keys()]) this.remove(name);
        this.setAtlas(null);
    }
}
