/** Shared helpers for the packed depth format: f16 depth | 6+6-bit octahedral normal | 4-bit cost. */
export const PACKING_WGSL = /* wgsl */ `
fn unpack_depth(packed: u32) -> f32 {
    return unpack2x16float(packed & 0xffffu).x;
}

fn unpack_normal(packed: u32) -> vec3<f32> {
    let e = vec2<f32>(f32((packed >> 16u) & 63u), f32((packed >> 22u) & 63u)) / 63.0 * 2.0 - 1.0;
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
        n = vec3<f32>((1.0 - abs(n.yx)) * s, n.z);
    }
    return normalize(n);
}

fn unpack_cost(packed: u32) -> u32 {
    return packed >> 28u;
}

fn pack_sample(depth: f32, normal: vec3<f32>, cost: f32) -> u32 {
    if (!(depth > 0.0) || cost >= 1.0) {
        return 0u;
    }
    var e = normal.xy / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    if (normal.z < 0.0) {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), e >= vec2<f32>(0.0));
        e = (1.0 - abs(e.yx)) * s;
    }
    let q = vec2<u32>(clamp(e * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0)) * 63.0 + 0.5);
    let c = min(15u, u32(cost * 15.0));
    return (pack2x16float(vec2<f32>(depth, 0.0)) & 0xffffu) | (q.x << 16u) | (q.y << 22u) | (c << 28u);
}
`;

/**
 * PatchMatch multi-view stereo (Gipuma / ACMH family) for one reference view:
 *  - per-pixel plane hypotheses (normal, depth) in the reference camera,
 *  - bilateral-weighted NCC over an 11×11 window sampled every 2 px, warped by the plane
 *    homography into each source view; the view costs are fused as the mean of the best k,
 *  - red/black checkerboard propagation that picks, in each of 8 regions around the pixel,
 *    the neighbour with the lowest current cost (adaptive checkerboard sampling),
 *  - random + perturbed refinement whose step shrinks every iteration,
 *  - optional geometric-consistency term (ACMM): forward–backward reprojection error
 *    against the source views' depth maps from the previous pass.
 */
export const PATCHMATCH_WGSL = /* wgsl */ `
struct View {
    k: vec4<f32>,
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    t: vec4<f32>,
};

struct Params {
    k: vec4<f32>,
    dims: vec4<u32>,
    range: vec4<f32>,
    state: vec4<u32>,
    views: array<View, 8>,
};

@group(0) @binding(0) var images: texture_2d_array<f32>;
@group(0) @binding(1) var bilinear: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> hyp: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> cost: array<f32>;
@group(0) @binding(5) var<storage, read> prior: array<f32>;
@group(0) @binding(6) var<storage, read_write> depths: array<u32>;
@group(0) @binding(7) var<storage, read_write> filtered: array<u32>;

const SAMPLES: u32 = 36u;
const SIGMA_COLOR: f32 = 0.06;
const SIGMA_SPACE: f32 = 5.0;
const INVALID: f32 = 2.0;
const GEOMETRIC_CAP: f32 = 3.0;
const FILTER_COST: u32 = 7u;
const FILTER_PIXELS: f32 = 1.0;
const FILTER_DEPTH: f32 = 0.01;

${PACKING_WGSL}

var<private> ref_value: array<f32, 36>;
var<private> ref_weight: array<f32, 36>;
var<private> ref_sum: f32;
var<private> ref_mean: f32;
var<private> ref_var: f32;
var<private> rng: u32;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random() -> f32 {
    rng = pcg(rng);
    return f32(rng >> 8u) / 16777216.0;
}

fn width() -> u32 { return params.dims.x; }
fn height() -> u32 { return params.dims.y; }

fn offset_of(i: u32) -> vec2<i32> {
    return vec2<i32>(i32(i % 6u) * 2 - 5, i32(i / 6u) * 2 - 5);
}

fn ray_of(p: vec2<f32>) -> vec3<f32> {
    let k = params.k;
    return vec3<f32>((p.x - k.z) / k.x, (p.y - k.w) / k.y, 1.0);
}

fn ref_pixel(p: vec2<i32>) -> f32 {
    let q = clamp(p, vec2<i32>(0), vec2<i32>(i32(width()) - 1, i32(height()) - 1));
    return textureLoad(images, q, i32(params.dims.z), 0).r;
}

fn prepare_window(p: vec2<i32>) {
    let center = ref_pixel(p);
    var sw = 0.0;
    var sr = 0.0;
    var srr = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let o = offset_of(i);
        let v = ref_pixel(p + o);
        let dc = v - center;
        let ds = f32(o.x * o.x + o.y * o.y);
        let w = exp(-dc * dc / (2.0 * SIGMA_COLOR * SIGMA_COLOR) - ds / (2.0 * SIGMA_SPACE * SIGMA_SPACE));
        ref_value[i] = v;
        ref_weight[i] = w;
        sw += w;
        sr += w * v;
        srr += w * v * v;
    }
    ref_sum = sw;
    ref_mean = sr / sw;
    ref_var = max(srr / sw - ref_mean * ref_mean, 0.0);
}

fn rotation_of(v: View) -> mat3x3<f32> {
    return transpose(mat3x3<f32>(v.r0.xyz, v.r1.xyz, v.r2.xyz));
}

fn homography(s: u32, n: vec3<f32>, rho: f32) -> mat3x3<f32> {
    let v = params.views[s];
    let t = v.t.xyz / rho;
    let m = rotation_of(v) + mat3x3<f32>(t * n.x, t * n.y, t * n.z);
    let ks = mat3x3<f32>(vec3<f32>(v.k.x, 0.0, 0.0), vec3<f32>(0.0, v.k.y, 0.0), vec3<f32>(v.k.z, v.k.w, 1.0));
    let k = params.k;
    let kinv = mat3x3<f32>(
        vec3<f32>(1.0 / k.x, 0.0, 0.0),
        vec3<f32>(0.0, 1.0 / k.y, 0.0),
        vec3<f32>(-k.z / k.x, -k.w / k.y, 1.0),
    );
    return ks * m * kinv;
}

fn view_cost(s: u32, h: mat3x3<f32>, p: vec2<f32>) -> f32 {
    let size = vec2<f32>(f32(width()), f32(height()));
    let layer = i32(params.views[s].t.w);
    var ss = 0.0;
    var sss = 0.0;
    var srs = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let q = h * vec3<f32>(p + vec2<f32>(offset_of(i)), 1.0);
        if (q.z <= 1e-6) {
            return INVALID;
        }
        let uv = q.xy / q.z;
        if (any(uv < vec2<f32>(0.0)) || any(uv > size - 1.0)) {
            return INVALID;
        }
        let v = textureSampleLevel(images, bilinear, (uv + 0.5) / size, layer, 0.0).r;
        let w = ref_weight[i];
        ss += w * v;
        sss += w * v * v;
        srs += w * v * ref_value[i];
    }
    let mean = ss / ref_sum;
    let var_s = max(sss / ref_sum - mean * mean, 0.0);
    let denominator = sqrt(ref_var * var_s);
    if (denominator < 1e-5) {
        return 1.0;
    }
    let cov = srs / ref_sum - ref_mean * mean;
    return clamp(1.0 - cov / denominator, 0.0, 2.0);
}

fn source_depth(view: u32, pixel: vec2<i32>) -> u32 {
    return depths[view * width() * height() + u32(pixel.y) * width() + u32(pixel.x)];
}

/** Forward–backward reprojection: ref pixel → source depth map → back into the reference. */
fn reproject(s: u32, p: vec2<f32>, x: vec3<f32>) -> vec4<f32> {
    let v = params.views[s];
    let r = rotation_of(v);
    let xs = r * x + v.t.xyz;
    if (xs.z <= 1e-6) {
        return vec4<f32>(-1.0);
    }
    let pixel = vec2<i32>(round(vec2<f32>(v.k.x * xs.x / xs.z + v.k.z, v.k.y * xs.y / xs.z + v.k.w)));
    if (any(pixel < vec2<i32>(0)) || pixel.x >= i32(width()) || pixel.y >= i32(height())) {
        return vec4<f32>(-1.0);
    }
    let packed = source_depth(u32(v.t.w), pixel);
    let ds = unpack_depth(packed);
    if (!(ds > 0.0) || unpack_cost(packed) > FILTER_COST) {
        return vec4<f32>(-1.0);
    }
    let back_s = vec3<f32>((f32(pixel.x) - v.k.z) / v.k.x, (f32(pixel.y) - v.k.w) / v.k.y, 1.0) * ds;
    let back = transpose(r) * (back_s - v.t.xyz);
    if (back.z <= 1e-6) {
        return vec4<f32>(-1.0);
    }
    let k = params.k;
    let q = vec2<f32>(k.x * back.x / back.z + k.z, k.y * back.y / back.z + k.w);
    return vec4<f32>(distance(q, p), back.z, 0.0, 1.0);
}

fn hypothesis_cost(p: vec2<f32>, n: vec3<f32>, d: f32) -> f32 {
    let inv = 1.0 / d;
    if (!(d > 0.0) || inv > params.range.x || inv < params.range.y) {
        return INVALID;
    }
    let ray = ray_of(p);
    if (dot(n, ray) >= -1e-3) {
        return INVALID;
    }
    let x = ray * d;
    let rho = dot(n, x);
    let count = params.dims.w;
    if (count == 0u) {
        return INVALID;
    }
    let geometric = (params.state.w & 1u) != 0u;
    var costs: array<f32, 8>;
    for (var s = 0u; s < count; s++) {
        var c = view_cost(s, homography(s, n, rho), p);
        if (geometric) {
            let r = reproject(s, p, x);
            let e = select(GEOMETRIC_CAP, min(r.x, GEOMETRIC_CAP), r.w > 0.0);
            c += params.range.w * e;
        }
        costs[s] = c;
    }
    let k = max(1u, min(3u, count - 1u));
    var total = 0.0;
    for (var pick = 0u; pick < k; pick++) {
        var best = pick;
        for (var j = pick + 1u; j < count; j++) {
            if (costs[j] < costs[best]) {
                best = j;
            }
        }
        let tmp = costs[pick];
        costs[pick] = costs[best];
        costs[best] = tmp;
        total += costs[pick];
    }
    return total / f32(k);
}

fn random_normal(ray: vec3<f32>) -> vec3<f32> {
    let z = random() * 2.0 - 1.0;
    let a = random() * 6.2831853;
    let r = sqrt(max(0.0, 1.0 - z * z));
    var n = vec3<f32>(r * cos(a), r * sin(a), z);
    let view = normalize(ray);
    if (dot(n, view) > 0.0) {
        n = -n;
    }
    return normalize(n - view * 0.5);
}

fn random_depth() -> f32 {
    return 1.0 / mix(params.range.y, params.range.x, random());
}

fn seed(index: u32) {
    rng = pcg(index ^ pcg(params.state.z + params.state.x * 7919u + params.state.y * 104729u));
}

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width() || gid.y >= height()) {
        return;
    }
    let index = gid.y * width() + gid.x;
    let p = vec2<f32>(f32(gid.x), f32(gid.y));
    seed(index);
    prepare_window(vec2<i32>(gid.xy));
    let ray = ray_of(p);
    var n = random_normal(ray);
    var d = random_depth();
    if ((params.state.w & 1u) != 0u) {
        let packed = depths[params.dims.z * width() * height() + index];
        if (unpack_depth(packed) > 0.0) {
            d = unpack_depth(packed);
            n = unpack_normal(packed);
        }
    }
    var c = hypothesis_cost(p, n, d);
    // The phone depth competes with the random guess instead of replacing it: where the image
    // has texture, propagation finds better planes; where it has none, the prior survives.
    if ((params.state.w & 1u) == 0u && prior[index] > 0.0) {
        let pn = -normalize(ray);
        let pc = hypothesis_cost(p, pn, prior[index]);
        if (pc < c) {
            c = pc;
            n = pn;
            d = prior[index];
        }
    }
    hyp[index] = vec4<f32>(n, d);
    cost[index] = c;
}

fn near_offset(i: u32) -> vec2<i32> {
    let m = i32((i + 1u) / 2u);
    return vec2<i32>(select(m, -m, (i & 1u) == 1u), -(m + 1));
}

fn orient(o: vec2<i32>, direction: u32) -> vec2<i32> {
    switch direction {
        case 1u: { return vec2<i32>(o.x, -o.y); }
        case 2u: { return vec2<i32>(o.y, o.x); }
        case 3u: { return vec2<i32>(-o.y, o.x); }
        default: { return o; }
    }
}

@compute @workgroup_size(8, 8)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x = gid.x * 2u + ((y + params.state.y) & 1u);
    if (x >= width() || y >= height()) {
        return;
    }
    let index = y * width() + x;
    let pi = vec2<i32>(i32(x), i32(y));
    let p = vec2<f32>(f32(x), f32(y));
    seed(index);
    prepare_window(pi);
    let ray = ray_of(p);
    var best = hyp[index];
    var best_cost = cost[index];

    for (var region = 0u; region < 8u; region++) {
        let count = select(11u, 7u, region < 4u);
        var chosen = -1;
        var chosen_cost = 1e9;
        for (var i = 0u; i < count; i++) {
            var o = vec2<i32>(0, -3 - 2 * i32(i));
            if (region < 4u) {
                o = near_offset(i);
            }
            let q = pi + orient(o, region % 4u);
            if (any(q < vec2<i32>(0)) || q.x >= i32(width()) || q.y >= i32(height())) {
                continue;
            }
            let qi = u32(q.y) * width() + u32(q.x);
            if (cost[qi] < chosen_cost) {
                chosen_cost = cost[qi];
                chosen = i32(qi);
            }
        }
        if (chosen < 0) {
            continue;
        }
        let h = hyp[u32(chosen)];
        let q = vec2<f32>(f32(u32(chosen) % width()), f32(u32(chosen) / width()));
        let denominator = dot(h.xyz, ray);
        if (abs(denominator) < 1e-6) {
            continue;
        }
        let d = dot(h.xyz, ray_of(q) * h.w) / denominator;
        let c = hypothesis_cost(p, h.xyz, d);
        if (c < best_cost) {
            best_cost = c;
            best = vec4<f32>(h.xyz, d);
        }
    }

    let step = params.range.z;
    let view = normalize(ray);
    let n_random = random_normal(ray);
    let d_random = random_depth();
    let d_perturbed = 1.0 / ((1.0 / best.w) * (1.0 + (random() * 2.0 - 1.0) * 0.15 * step));
    var n_perturbed = normalize(best.xyz + (vec3<f32>(random(), random(), random()) * 2.0 - 1.0) * 0.6 * step);
    if (dot(n_perturbed, view) >= 0.0) {
        n_perturbed = best.xyz;
    }
    let current_n = best.xyz;
    let current_d = best.w;
    var normals = array<vec3<f32>, 6>(n_random, n_perturbed, current_n, n_random, current_n, n_perturbed);
    var depths_try = array<f32, 6>(d_random, d_perturbed, d_random, current_d, d_perturbed, current_d);
    for (var c = 0u; c < 6u; c++) {
        let candidate = hypothesis_cost(p, normals[c], depths_try[c]);
        if (candidate < best_cost) {
            best_cost = candidate;
            best = vec4<f32>(normals[c], depths_try[c]);
        }
    }
    hyp[index] = best;
    cost[index] = best_cost;
}

@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = width() * height();
    if (gid.x >= total) {
        return;
    }
    let h = hyp[gid.x];
    depths[params.dims.z * total + gid.x] = pack_sample(h.w, h.xyz, cost[gid.x]);
}

/** Keeps a depth only if enough source views reproject it consistently (forward–backward). */
@compute @workgroup_size(8, 8)
fn consistency(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width() || gid.y >= height()) {
        return;
    }
    let index = gid.y * width() + gid.x;
    let packed = depths[params.dims.z * width() * height() + index];
    let d = unpack_depth(packed);
    if (!(d > 0.0) || unpack_cost(packed) > FILTER_COST) {
        filtered[index] = 0u;
        return;
    }
    let p = vec2<f32>(f32(gid.x), f32(gid.y));
    let x = ray_of(p) * d;
    var agree = 0u;
    var sum = d;
    for (var s = 0u; s < params.dims.w; s++) {
        let r = reproject(s, p, x);
        if (r.w > 0.0 && r.x < FILTER_PIXELS && abs(r.y - d) < FILTER_DEPTH * d) {
            agree++;
            sum += r.y;
        }
    }
    // About 40 % of the checked views must agree (at least two): rejects the "fattened"
    // depths that leak across silhouettes and only agree with the closest neighbours.
    let needed = min(params.dims.w, clamp((params.dims.w * 2u + 2u) / 5u, 2u, 4u));
    if (agree < needed) {
        filtered[index] = 0u;
        return;
    }
    let mean = sum / f32(agree + 1u);
    filtered[index] = (packed & 0xffff0000u) | (pack2x16float(vec2<f32>(mean, 0.0)) & 0xffffu);
}
`;
