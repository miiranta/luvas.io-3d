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
