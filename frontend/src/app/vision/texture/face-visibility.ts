import { Intrinsics, Pose, cameraCenter } from '../geometry/camera';
import { BufferUsage, MapMode, TextureUsage, storageBuffer } from '../gpu/gpu-device';

export interface VisibilityView {
    intrinsics: Intrinsics;
    pose: Pose;
    /** Relative image quality (sharpness), multiplies the projected area. */
    quality: number;
}

/** Best views per face: `slots` (view, score) pairs per face, score 0 = empty. */
export interface FaceCandidates {
    slots: number;
    view: Int16Array;
    score: Float32Array;
}

export const CANDIDATE_SLOTS = 4;

/** A face counts as visible when its centroid is at most this much behind the z-buffer. */
const DEPTH_TOLERANCE = 0.01;
const MIN_COSINE = 0.12;
const BORDER = 2;

const DEPTH_WGSL = /* wgsl */ `
struct Camera {
    k: vec4<f32>,
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    t: vec4<f32>,
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) depth: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Varyings {
    let index = indices[vi];
    let p = vec3<f32>(positions[index * 3u], positions[index * 3u + 1u], positions[index * 3u + 2u]);
    let r = transpose(mat3x3<f32>(camera.r0.xyz, camera.r1.xyz, camera.r2.xyz));
    let c = r * p + camera.t.xyz;
    let near = camera.size.z;
    let far = camera.size.w;
    let u = camera.k.x * c.x / max(c.z, 1e-4) + camera.k.z + 0.5;
    let v = camera.k.y * c.y / max(c.z, 1e-4) + camera.k.w + 0.5;
    let ndc = vec2<f32>(u / camera.size.x * 2.0 - 1.0, 1.0 - v / camera.size.y * 2.0);
    var out: Varyings;
    // Perspective depth for the z-test (interpolates linearly in screen space).
    let z = select(far * (c.z - near) / (far - near), -1.0, c.z <= near);
    out.position = vec4<f32>(ndc * c.z, z, c.z);
    out.depth = c.z;
    return out;
}

@fragment
fn fs(in: Varyings) -> @location(0) f32 {
    return in.depth;
}
`;

/**
 * Chooses texturing candidates per face. Each view's z-buffer is rendered on the GPU (linear
 * depth), then every face that faces the camera, projects inside the image and is not behind
 * the z-buffer at its centroid gets a score = projected area in pixels × view quality. This
 * works for faces smaller than a pixel, where counting covered pixels would be too noisy.
 */
export async function faceCandidates(
    device: GPUDevice,
    positions: Float32Array,
    indices: Uint32Array,
    views: VisibilityView[],
    width: number,
    height: number,
    near: number,
    far: number,
    progress: (fraction: number) => void,
): Promise<FaceCandidates> {
    const faces = indices.length / 3;
    const candidates: FaceCandidates = {
        slots: CANDIDATE_SLOTS,
        view: new Int16Array(faces * CANDIDATE_SLOTS).fill(-1),
        score: new Float32Array(faces * CANDIDATE_SLOTS),
    };
    const geometry = faceGeometry(positions, indices);

    const positionBuffer = storageBuffer(device, positions.byteLength, 'mesh positions');
    const indexBuffer = storageBuffer(device, indices.byteLength, 'mesh indices');
    device.queue.writeBuffer(positionBuffer, 0, positions as Float32Array<ArrayBuffer>);
    device.queue.writeBuffer(indexBuffer, 0, indices as Uint32Array<ArrayBuffer>);
    const uniforms = device.createBuffer({
        size: 96,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    const module = device.createShaderModule({ label: 'view depth', code: DEPTH_WGSL });
    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: 'r32float' }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniforms } },
            { binding: 1, resource: { buffer: positionBuffer } },
            { binding: 2, resource: { buffer: indexBuffer } },
        ],
    });
    const target = device.createTexture({
        size: { width, height },
        format: 'r32float',
        usage: TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
    });
    const depth = device.createTexture({
        size: { width, height },
        format: 'depth32float',
        usage: TextureUsage.RENDER_ATTACHMENT,
    });
    const rowBytes = Math.ceil((width * 4) / 256) * 256;
    const staging = device.createBuffer({
        size: rowBytes * height,
        usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
    });

    for (let v = 0; v < views.length; v++) {
        const view = views[v];
        const k = view.intrinsics;
        const { R, t } = view.pose;
        device.queue.writeBuffer(
            uniforms,
            0,
            new Float32Array([
                k.fx,
                k.fy,
                k.cx,
                k.cy,
                R[0],
                R[1],
                R[2],
                0,
                R[3],
                R[4],
                R[5],
                0,
                R[6],
                R[7],
                R[8],
                0,
                t[0],
                t[1],
                t[2],
                0,
                width,
                height,
                near,
                far,
            ]),
        );
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: target.createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: depth.createView(),
                depthClearValue: 1,
                depthLoadOp: 'clear',
                depthStoreOp: 'discard',
            },
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(indices.length);
        pass.end();
        encoder.copyTextureToBuffer(
            { texture: target },
            { buffer: staging, bytesPerRow: rowBytes },
            { width, height },
        );
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(MapMode.READ);
        scoreView(
            candidates,
            geometry,
            view,
            v,
            new Float32Array(staging.getMappedRange()),
            rowBytes / 4,
            width,
            height,
        );
        staging.unmap();
        progress((v + 1) / views.length);
    }
    for (const resource of [positionBuffer, indexBuffer, uniforms, staging]) resource.destroy();
    target.destroy();
    depth.destroy();
    return candidates;
}

/** Centroid (3), unit normal (3) and area (1) per face. */
function faceGeometry(p: Float32Array, indices: Uint32Array): Float32Array {
    const faces = indices.length / 3;
    const data = new Float32Array(faces * 7);
    for (let f = 0; f < faces; f++) {
        const a = indices[f * 3] * 3;
        const b = indices[f * 3 + 1] * 3;
        const c = indices[f * 3 + 2] * 3;
        const e1x = p[b] - p[a];
        const e1y = p[b + 1] - p[a + 1];
        const e1z = p[b + 2] - p[a + 2];
        const e2x = p[c] - p[a];
        const e2y = p[c + 1] - p[a + 1];
        const e2z = p[c + 2] - p[a + 2];
        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;
        const length = Math.hypot(nx, ny, nz) || 1e-12;
        const o = f * 7;
        data[o] = (p[a] + p[b] + p[c]) / 3;
        data[o + 1] = (p[a + 1] + p[b + 1] + p[c + 1]) / 3;
        data[o + 2] = (p[a + 2] + p[b + 2] + p[c + 2]) / 3;
        data[o + 3] = nx / length;
        data[o + 4] = ny / length;
        data[o + 5] = nz / length;
        data[o + 6] = length / 2;
    }
    return data;
}

function scoreView(
    candidates: FaceCandidates,
    data: Float32Array,
    view: VisibilityView,
    index: number,
    zbuffer: Float32Array,
    stride: number,
    width: number,
    height: number,
): void {
    const { R, t } = view.pose;
    const k = view.intrinsics;
    const [cx, cy, cz] = cameraCenter(view.pose);
    const faces = data.length / 7;
    const pixelArea = k.fx * k.fy * view.quality;
    for (let f = 0; f < faces; f++) {
        const o = f * 7;
        const x = data[o];
        const y = data[o + 1];
        const z = data[o + 2];
        const vx = cx - x;
        const vy = cy - y;
        const vz = cz - z;
        const distance2 = vx * vx + vy * vy + vz * vz;
        const facing = data[o + 3] * vx + data[o + 4] * vy + data[o + 5] * vz;
        if (facing <= MIN_COSINE * Math.sqrt(distance2)) continue;
        const camZ = R[6] * x + R[7] * y + R[8] * z + t[2];
        if (camZ <= 0.01) continue;
        const u = (k.fx * (R[0] * x + R[1] * y + R[2] * z + t[0])) / camZ + k.cx;
        const v = (k.fy * (R[3] * x + R[4] * y + R[5] * z + t[1])) / camZ + k.cy;
        if (u < BORDER || v < BORDER || u > width - 1 - BORDER || v > height - 1 - BORDER) continue;
        const surface = zbuffer[Math.round(v) * stride + Math.round(u)];
        if (!(surface > 0) || camZ > surface * (1 + DEPTH_TOLERANCE)) continue;
        // Projected area ≈ A · cosθ / d² · fx · fy (pixels²), with facing = cosθ · d.
        const score = (data[o + 6] * facing * pixelArea) / (distance2 * Math.sqrt(distance2));
        insertCandidate(candidates, f, index, score);
    }
}

function insertCandidate(c: FaceCandidates, face: number, view: number, score: number): void {
    const base = face * c.slots;
    let position = c.slots;
    while (position > 0 && c.score[base + position - 1] < score) position--;
    if (position >= c.slots) return;
    for (let s = c.slots - 1; s > position; s--) {
        c.score[base + s] = c.score[base + s - 1];
        c.view[base + s] = c.view[base + s - 1];
    }
    c.score[base + position] = score;
    c.view[base + position] = view;
}
