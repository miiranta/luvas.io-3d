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

interface Target {
    framebuffer: WebGLFramebuffer;
    texture: WebGLTexture;
    pbo: WebGLBuffer;
    width: number;
    height: number;
}

/**
 * Copies the WebXR camera texture to RGBA pixels without stalling the frame: the texture is
 * blitted (flipped upright, resized) into a framebuffer, read into a pixel-pack buffer and
 * fenced; the pixels are fetched on a later frame once the GPU has finished.
 */
export class CameraReadback {
    private readonly blit: WebGLProgram;
    private target: Target | null = null;
    private pending: { fence: WebGLSync; width: number; height: number } | null = null;

    constructor(private readonly gl: WebGL2RenderingContext) {
        this.blit = compile(gl);
    }

    /** A copy is on its way (one at a time). */
    get busy(): boolean {
        return this.pending !== null;
    }

    start(texture: WebGLTexture, width: number, height: number): void {
        const gl = this.gl;
        const target = this.ensureTarget(width, height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
        gl.viewport(0, 0, width, height);
        gl.disable(gl.BLEND);
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(this.blit);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(gl.getUniformLocation(this.blit, 'uCamera'), 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, target.pbo);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, 0);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) as WebGLSync;
        gl.flush();
        this.pending = { fence, width, height };
    }

    /** The pixels of the pending copy once the GPU is done, otherwise null. */
    collect(): Uint8Array | null {
        const gl = this.gl;
        const pending = this.pending;
        if (!pending || !this.target) return null;
        if (gl.clientWaitSync(pending.fence, 0, 0) === gl.TIMEOUT_EXPIRED) return null;
        gl.deleteSync(pending.fence);
        this.pending = null;
        const pixels = new Uint8Array(pending.width * pending.height * 4);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.target.pbo);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, pixels);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        return pixels;
    }

    private ensureTarget(width: number, height: number): Target {
        const gl = this.gl;
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
}

function compile(gl: WebGL2RenderingContext): WebGLProgram {
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
