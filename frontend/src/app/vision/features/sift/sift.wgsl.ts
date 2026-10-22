/**
 * SIFT on the GPU. All Gaussian images of all octaves live in one f32 buffer (octave o, image l
 * at `octaves[o].x + l * octaves[o].w`); every dispatch reads its parameters from one slot of a
 * uniform buffer with a dynamic offset, so a whole image is one command buffer.
 *   convert (2× upsampling) → blur_h / blur_v (base, then 5 increments per octave) → downsample → extrema
 *   (DoG computed on the fly, sub-pixel refinement, appended with an atomic counter)
 *   … CPU picks the keypoints … → describe (one workgroup per keypoint).
 * Must agree with the CPU reference in scale-space.ts, keypoints.ts and descriptor.ts.
 */
export const SIFT_WGSL = /* wgsl */ `
struct Job {
    src: u32,
    dst: u32,
    width: u32,
    height: u32,
    radius: u32,
    kernel: u32,
    octave: u32,
    src_width: u32,
    prefilter: f32,
    contrast: f32,
    edge: f32,
    capacity: u32,
    // Per octave: base offset, width, height, plane size.
    octaves: array<vec4<u32>, 8>,
};

@group(0) @binding(0) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(1) var<storage, read> kernels: array<f32>;
@group(0) @binding(2) var<uniform> job: Job;
@group(0) @binding(3) var<storage, read> gray: array<u32>;
// [0] = count, then 8 words per keypoint: octave, ox, oy, scale, score (as f32 bits).
@group(0) @binding(4) var<storage, read_write> found: array<atomic<u32>>;
// Per selected keypoint: octave, ox, oy, scale.
@group(0) @binding(5) var<storage, read> selected: array<vec4<f32>>;
// Per selected keypoint: 32 words of packed descriptor bytes, then the angle (f32 bits).
@group(0) @binding(6) var<storage, read_write> descriptors: array<u32>;

const INTERVALS: f32 = 3.0;
const SIGMA0: f32 = 1.6;
const BORDER: i32 = 5;
const TAU: f32 = 6.283185307;
const FIXED: f32 = 262144.0;

fn input_pixel(x: u32, y: u32) -> f32 {
    let i = y * job.src_width + x;
    return f32((gray[i / 4u] >> ((i % 4u) * 8u)) & 255u);
}

/** 2× bilinear enlargement of the 8-bit input (see upsample in scale-space.ts). */
@compute @workgroup_size(8, 8)
fn convert(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= job.width || gid.y >= job.height) { return; }
    let last = vec2<u32>(job.src_width - 1u, job.height / 2u - 1u);
    let p = vec2<f32>(gid.xy) * 0.5;
    let p0 = min(vec2<u32>(floor(p)), last);
    let p1 = min(p0 + 1u, last);
    let f = p - vec2<f32>(p0);
    let top = mix(input_pixel(p0.x, p0.y), input_pixel(p1.x, p0.y), f.x);
    let bottom = mix(input_pixel(p0.x, p1.y), input_pixel(p1.x, p1.y), f.x);
    pyramid[job.dst + gid.y * job.width + gid.x] = mix(top, bottom, f.y) / 255.0;
}

@compute @workgroup_size(8, 8)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= job.width || gid.y >= job.height) { return; }
    let row = job.src + gid.y * job.width;
    let x = i32(gid.x);
    let last = i32(job.width) - 1;
    var value = kernels[job.kernel] * pyramid[row + gid.x];
    for (var i = 1; i <= i32(job.radius); i++) {
        let left = u32(max(x - i, 0));
        let right = u32(min(x + i, last));
        value += kernels[job.kernel + u32(i)] * (pyramid[row + left] + pyramid[row + right]);
    }
    pyramid[job.dst + gid.y * job.width + gid.x] = value;
}

@compute @workgroup_size(8, 8)
fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= job.width || gid.y >= job.height) { return; }
    let y = i32(gid.y);
    let last = i32(job.height) - 1;
    let w = job.width;
    var value = kernels[job.kernel] * pyramid[job.src + gid.y * w + gid.x];
    for (var i = 1; i <= i32(job.radius); i++) {
        let up = u32(max(y - i, 0));
        let down = u32(min(y + i, last));
        value += kernels[job.kernel + u32(i)] *
            (pyramid[job.src + up * w + gid.x] + pyramid[job.src + down * w + gid.x]);
    }
    pyramid[job.dst + gid.y * w + gid.x] = value;
}

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= job.width || gid.y >= job.height) { return; }
    pyramid[job.dst + gid.y * job.width + gid.x] =
        pyramid[job.src + gid.y * 2u * job.src_width + gid.x * 2u];
}

fn gaussian(l: i32, x: i32, y: i32) -> f32 {
    let o = job.octaves[job.octave];
    return pyramid[o.x + u32(l) * o.w + u32(y) * o.y + u32(x)];
}

fn dog(s: i32, x: i32, y: i32) -> f32 {
    return gaussian(s + 1, x, y) - gaussian(s, x, y);
}

/** Solves the symmetric system [[a, d, e], [d, b, f], [e, f, c]] · x = g (Cramer). */
fn solve3(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: vec3<f32>) -> vec4<f32> {
    let det = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e);
    if (abs(det) < 1e-12) { return vec4<f32>(0.0); }
    let x = (g.x * (b * c - f * f) - d * (g.y * c - f * g.z) + e * (g.y * f - b * g.z)) / det;
    let y = (a * (g.y * c - f * g.z) - g.x * (d * c - f * e) + e * (d * g.z - g.y * e)) / det;
    let z = (a * (b * g.z - g.y * f) - d * (d * g.z - g.y * e) + g.x * (d * f - b * e)) / det;
    return vec4<f32>(x, y, z, 1.0);
}

fn refine(s0: i32, x0: i32, y0: i32) {
    let o = job.octaves[job.octave];
    let w = i32(o.y);
    let h = i32(o.z);
    var x = x0;
    var y = y0;
    var s = s0;
    var offset = vec3<f32>(0.0);
    var gradient = vec3<f32>(0.0);
    var dxx = 0.0;
    var dyy = 0.0;
    var dxy = 0.0;
    var converged = false;
    for (var step = 0; step < 5; step++) {
        let c = dog(s, x, y);
        gradient = vec3<f32>(
            0.5 * (dog(s, x + 1, y) - dog(s, x - 1, y)),
            0.5 * (dog(s, x, y + 1) - dog(s, x, y - 1)),
            0.5 * (dog(s + 1, x, y) - dog(s - 1, x, y)),
        );
        dxx = dog(s, x + 1, y) + dog(s, x - 1, y) - 2.0 * c;
        dyy = dog(s, x, y + 1) + dog(s, x, y - 1) - 2.0 * c;
        let dss = dog(s + 1, x, y) + dog(s - 1, x, y) - 2.0 * c;
        dxy = 0.25 * (dog(s, x + 1, y + 1) - dog(s, x - 1, y + 1) - dog(s, x + 1, y - 1) +
            dog(s, x - 1, y - 1));
        let dxs = 0.25 * (dog(s + 1, x + 1, y) - dog(s + 1, x - 1, y) - dog(s - 1, x + 1, y) +
            dog(s - 1, x - 1, y));
        let dys = 0.25 * (dog(s + 1, x, y + 1) - dog(s + 1, x, y - 1) - dog(s - 1, x, y + 1) +
            dog(s - 1, x, y - 1));
        let solved = solve3(dxx, dyy, dss, dxy, dxs, dys, gradient);
        if (solved.w == 0.0) { return; }
        offset = -solved.xyz;
        if (all(abs(offset) < vec3<f32>(0.5))) {
            converged = true;
            break;
        }
        if (any(abs(offset) > vec3<f32>(4.0))) { return; }
        x += i32(round(offset.x));
        y += i32(round(offset.y));
        s += i32(round(offset.z));
        if (s < 1 || s > 3 || x < BORDER || x >= w - BORDER || y < BORDER || y >= h - BORDER) {
            return;
        }
    }
    if (!converged) { return; }
    let contrast = dog(s, x, y) + 0.5 * dot(gradient, offset);
    if (abs(contrast) * INTERVALS < job.contrast) { return; }
    let trace = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if (det <= 0.0 || trace * trace * job.edge >= (job.edge + 1.0) * (job.edge + 1.0) * det) {
        return;
    }
    let index = atomicAdd(&found[0], 1u);
    if (index >= job.capacity) { return; }
    let base = 8u + index * 8u;
    atomicStore(&found[base], bitcast<u32>(f32(job.octave)));
    atomicStore(&found[base + 1u], bitcast<u32>(f32(x) + offset.x));
    atomicStore(&found[base + 2u], bitcast<u32>(f32(y) + offset.y));
    atomicStore(&found[base + 3u], bitcast<u32>(f32(s) + offset.z));
    atomicStore(&found[base + 4u], bitcast<u32>(abs(contrast)));
}

@compute @workgroup_size(8, 8)
fn extrema(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = job.octaves[job.octave];
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x < BORDER || y < BORDER || x >= i32(o.y) - BORDER || y >= i32(o.z) - BORDER) { return; }
    for (var s = 1; s <= 3; s++) {
        let v = dog(s, x, y);
        if (abs(v) <= job.prefilter) { continue; }
        var extremum = true;
        for (var ds = -1; ds <= 1 && extremum; ds++) {
            for (var dy = -1; dy <= 1 && extremum; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    if (ds == 0 && dy == 0 && dx == 0) { continue; }
                    let n = dog(s + ds, x + dx, y + dy);
                    if ((v > 0.0 && n >= v) || (v < 0.0 && n <= v)) {
                        extremum = false;
                        break;
                    }
                }
            }
        }
        if (extremum) { refine(s, x, y); }
    }
}

var<workgroup> orientations: array<atomic<u32>, 36>;
var<workgroup> histogram: array<atomic<u32>, 360>;
var<workgroup> keypoint_angle: f32;

fn image_gradient(base: u32, w: u32, x: i32, y: i32) -> vec2<f32> {
    let i = base + u32(y) * w + u32(x);
    return vec2<f32>(pyramid[i + 1u] - pyramid[i - 1u], pyramid[i + w] - pyramid[i - w]);
}

/** Orientation and descriptor of one selected keypoint per workgroup (see descriptor.ts). */
@compute @workgroup_size(64)
fn describe(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let k = wid.x;
    if (k >= job.capacity) { return; }
    let key = selected[k];
    let o = job.octaves[u32(key.x)];
    let w = i32(o.y);
    let h = i32(o.z);
    let scale = key.w;
    let image = o.x + u32(clamp(round(scale), 0.0, 5.0)) * o.w;
    let sigma = SIGMA0 * exp2(scale / INTERVALS);
    let cx = i32(round(key.y));
    let cy = i32(round(key.z));
    for (var i = lid; i < 360u; i += 64u) { atomicStore(&histogram[i], 0u); }
    if (lid < 36u) { atomicStore(&orientations[lid], 0u); }
    workgroupBarrier();

    let spread = 1.5 * sigma;
    let reach = i32(round(3.0 * spread));
    let side = 2 * reach + 1;
    for (var i = i32(lid); i < side * side; i += 64) {
        let dx = i % side - reach;
        let dy = i / side - reach;
        let x = cx + dx;
        let y = cy + dy;
        if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) { continue; }
        let g = image_gradient(image, o.y, x, y);
        let weight = exp(-f32(dx * dx + dy * dy) / (2.0 * spread * spread));
        let bin = (i32(round(36.0 * atan2(g.y, g.x) / TAU)) + 36) % 36;
        atomicAdd(&orientations[bin], u32(length(g) * weight * FIXED));
    }
    workgroupBarrier();
    if (lid == 0u) {
        var smoothed: array<f32, 36>;
        var peak = 0;
        for (var i = 0; i < 36; i++) {
            let h2 = f32(atomicLoad(&orientations[(i + 34) % 36])) +
                f32(atomicLoad(&orientations[(i + 2) % 36]));
            let h1 = f32(atomicLoad(&orientations[(i + 35) % 36])) +
                f32(atomicLoad(&orientations[(i + 1) % 36]));
            smoothed[i] = h2 / 16.0 + h1 * 4.0 / 16.0 + f32(atomicLoad(&orientations[i])) * 6.0 / 16.0;
            if (smoothed[i] > smoothed[peak]) { peak = i; }
        }
        let left = smoothed[(peak + 35) % 36];
        let right = smoothed[(peak + 1) % 36];
        let curvature = left - 2.0 * smoothed[peak] + right;
        var offset = 0.0;
        if (curvature < 0.0) { offset = 0.5 * (left - right) / curvature; }
        keypoint_angle = (f32(peak) + offset) / 36.0 * TAU;
    }
    workgroupBarrier();

    let angle = keypoint_angle;
    let cell = 3.0 * sigma;
    let radius = i32(round(cell * 1.41421356 * 5.0 * 0.5));
    let window = 2 * radius + 1;
    let c = cos(angle) / cell;
    let s = sin(angle) / cell;
    for (var i = i32(lid); i < window * window; i += 64) {
        let x = cx + i % window - radius;
        let y = cy + i / window - radius;
        if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) { continue; }
        let px = f32(x) - key.y;
        let py = f32(y) - key.z;
        let u = c * px + s * py;
        let v = -s * px + c * py;
        let column = u + 1.5;
        let row = v + 1.5;
        if (row <= -1.0 || row >= 4.0 || column <= -1.0 || column >= 4.0) { continue; }
        let g = image_gradient(image, o.y, x, y);
        var bin = (atan2(g.y, g.x) - angle) * 8.0 / TAU;
        bin -= floor(bin / 8.0) * 8.0;
        let value = length(g) * exp(-(u * u + v * v) / 8.0) * FIXED;
        let r0 = floor(row);
        let c0 = floor(column);
        let b0 = min(floor(bin), 7.0);
        let fr = row - r0;
        let fc = column - c0;
        let fo = bin - b0;
        let base = u32((i32(r0) + 1) * 60 + (i32(c0) + 1) * 10 + i32(b0));
        let top = value * (1.0 - fr);
        let bottom = value * fr;
        atomicAdd(&histogram[base], u32(top * (1.0 - fc) * (1.0 - fo)));
        atomicAdd(&histogram[base + 1u], u32(top * (1.0 - fc) * fo));
        atomicAdd(&histogram[base + 10u], u32(top * fc * (1.0 - fo)));
        atomicAdd(&histogram[base + 11u], u32(top * fc * fo));
        atomicAdd(&histogram[base + 60u], u32(bottom * (1.0 - fc) * (1.0 - fo)));
        atomicAdd(&histogram[base + 61u], u32(bottom * (1.0 - fc) * fo));
        atomicAdd(&histogram[base + 70u], u32(bottom * fc * (1.0 - fo)));
        atomicAdd(&histogram[base + 71u], u32(bottom * fc * fo));
    }
    workgroupBarrier();
    if (lid == 0u) {
        var d: array<f32, 128>;
        var norm = 0.0;
        for (var r = 0u; r < 4u; r++) {
            for (var q = 0u; q < 4u; q++) {
                let base = (r + 1u) * 60u + (q + 1u) * 10u;
                for (var b = 0u; b < 8u; b++) {
                    var value = f32(atomicLoad(&histogram[base + b]));
                    if (b < 2u) { value += f32(atomicLoad(&histogram[base + 8u + b])); }
                    d[(r * 4u + q) * 8u + b] = value;
                    norm += value * value;
                }
            }
        }
        let limit = 0.2 * sqrt(norm);
        var l1 = 0.0;
        for (var i = 0u; i < 128u; i++) {
            d[i] = min(d[i], limit);
            l1 += d[i];
        }
        let out = k * 33u;
        for (var word = 0u; word < 32u; word++) {
            var packed = 0u;
            for (var b = 0u; b < 4u; b++) {
                var root = 0.0;
                if (l1 > 0.0) { root = sqrt(d[word * 4u + b] / l1); }
                packed |= u32(min(255.0, round(root * 512.0))) << (b * 8u);
            }
            descriptors[out + word] = packed;
        }
        descriptors[out + 32u] = bitcast<u32>(angle);
    }
}
`;
