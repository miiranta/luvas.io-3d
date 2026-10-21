/**
 * Analytic test scene shared by `npm run verify` and the fake WebXR runtime of `npm run e2e`:
 * a textured box and sphere on a textured table top, lit by a fixed light (Lambertian, so the
 * appearance does not depend on the viewpoint). The GLSL ray caster renders camera images; the
 * JavaScript twins give exact depth, distance-to-surface and colour for scoring.
 */

export const BOX_CENTER = [0, 0.06, 0];
export const BOX_HALF = [0.07, 0.06, 0.05];
export const SPHERE_CENTER = [0.13, 0.05, 0.07];
export const SPHERE_RADIUS = 0.05;
export const TABLE_HALF = 0.5;
export const TARGET = [0.03, 0.05, 0.02];

/** Known dimensions of the box (m) for the metric check. */
export const BOX_SIZE = [BOX_HALF[0] * 2, BOX_HALF[1] * 2, BOX_HALF[2] * 2];

export const SCENE_GLSL = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
uniform vec4 uK;
uniform mat3 uCameraToWorld;
uniform vec3 uCenter;
uniform vec2 uSize;
out vec4 color;

const vec3 BOX_C = vec3(${BOX_CENTER.join(', ')});
const vec3 BOX_H = vec3(${BOX_HALF.join(', ')});
const vec3 SPHERE_C = vec3(${SPHERE_CENTER.join(', ')});
const float SPHERE_R = ${SPHERE_RADIUS};
const float TABLE = ${TABLE_HALF};

uint hash3(ivec3 c) {
    uvec3 q = uvec3(c) * uvec3(1597334673u, 3812015801u, 2798796415u);
    return (q.x ^ q.y ^ q.z) * 1597334673u;
}

float lattice(ivec3 c) {
    return float(hash3(c) >> 8u) / 16777216.0;
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 w = f * f * (3.0 - 2.0 * f);
    ivec3 c = ivec3(i);
    float x00 = mix(lattice(c), lattice(c + ivec3(1, 0, 0)), w.x);
    float x10 = mix(lattice(c + ivec3(0, 1, 0)), lattice(c + ivec3(1, 1, 0)), w.x);
    float x01 = mix(lattice(c + ivec3(0, 0, 1)), lattice(c + ivec3(1, 0, 1)), w.x);
    float x11 = mix(lattice(c + ivec3(0, 1, 1)), lattice(c + ivec3(1, 1, 1)), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

vec3 albedo(vec3 p, int object) {
    float a = noise(p * 70.0);
    float b = noise(p * 210.0 + 11.0);
    float c = noise(p * 18.0 + 5.0);
    vec3 low = object == 0 ? vec3(0.32, 0.2, 0.12) : object == 1 ? vec3(0.1, 0.16, 0.42) : vec3(0.42, 0.08, 0.2);
    vec3 high = object == 0 ? vec3(0.92, 0.74, 0.5) : object == 1 ? vec3(0.45, 0.85, 0.95) : vec3(1.0, 0.75, 0.3);
    return mix(low, high, clamp(0.55 * a + 0.3 * b + 0.25 * c - 0.05, 0.0, 1.0));
}

float shade(vec3 n) {
    return 0.6 + 0.4 * max(0.0, dot(n, normalize(vec3(0.4, 1.0, 0.3))));
}

vec3 trace(vec3 o, vec3 d) {
    float best = 1e9;
    vec3 n = vec3(0.0);
    int object = -1;
    if (d.y < 0.0) {
        float t = -o.y / d.y;
        vec3 p = o + d * t;
        if (abs(p.x) < TABLE && abs(p.z) < TABLE) { best = t; n = vec3(0, 1, 0); object = 0; }
    }
    vec3 inv = 1.0 / d;
    vec3 t0 = (BOX_C - BOX_H - o) * inv;
    vec3 t1 = (BOX_C + BOX_H - o) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float leave = min(min(tmax.x, tmax.y), tmax.z);
    if (enter > 0.0 && enter < leave && enter < best) {
        best = enter;
        object = 1;
        n = enter == tmin.x ? vec3(-sign(d.x), 0, 0) : enter == tmin.y ? vec3(0, -sign(d.y), 0) : vec3(0, 0, -sign(d.z));
    }
    vec3 oc = o - SPHERE_C;
    float b = dot(oc, d);
    float h = b * b - dot(oc, oc) + SPHERE_R * SPHERE_R;
    if (h > 0.0) {
        float t = -b - sqrt(h);
        if (t > 0.0 && t < best) { best = t; object = 2; n = normalize(o + d * t - SPHERE_C); }
    }
    if (object < 0) return mix(vec3(0.85, 0.88, 0.95), vec3(0.55, 0.6, 0.7), clamp(d.y * 0.5 + 0.5, 0.0, 1.0));
    return albedo(o + d * best, object) * shade(n);
}

void main() {
    vec3 sum = vec3(0.0);
    for (int s = 0; s < 4; s++) {
        vec2 jitter = vec2(float(s & 1), float(s >> 1)) * 0.5 - 0.25;
        float u = gl_FragCoord.x - 0.5 + jitter.x;
        float v = uSize.y - gl_FragCoord.y - 0.5 + jitter.y;
        vec3 ray = uCameraToWorld * vec3((u - uK.z) / uK.x, (v - uK.w) / uK.y, 1.0);
        sum += trace(uCenter, normalize(ray));
    }
    color = vec4(pow(sum / 4.0, vec3(1.0 / 1.6)), 1.0);
}
`;

function hash3(x: number, y: number, z: number): number {
    const qx = Math.imul(x, 1597334673) >>> 0;
    const qy = Math.imul(y, 3812015801 | 0) >>> 0;
    const qz = Math.imul(z, 2798796415 | 0) >>> 0;
    return Math.imul((qx ^ qy ^ qz) >>> 0, 1597334673) >>> 0;
}

function lattice(x: number, y: number, z: number): number {
    return (hash3(x, y, z) >>> 8) / 16777216;
}

function noise(x: number, y: number, z: number): number {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const iz = Math.floor(z);
    const fx = x - ix;
    const fy = y - iy;
    const fz = z - iz;
    const wx = fx * fx * (3 - 2 * fx);
    const wy = fy * fy * (3 - 2 * fy);
    const wz = fz * fz * (3 - 2 * fz);
    const mix = (a: number, b: number, t: number) => a + (b - a) * t;
    const x00 = mix(lattice(ix, iy, iz), lattice(ix + 1, iy, iz), wx);
    const x10 = mix(lattice(ix, iy + 1, iz), lattice(ix + 1, iy + 1, iz), wx);
    const x01 = mix(lattice(ix, iy, iz + 1), lattice(ix + 1, iy, iz + 1), wx);
    const x11 = mix(lattice(ix, iy + 1, iz + 1), lattice(ix + 1, iy + 1, iz + 1), wx);
    return mix(mix(x00, x10, wy), mix(x01, x11, wy), wz);
}

/** Rendered colour (0–255) of a surface point, matching the shader. */
export function surfaceColor(p: number[]): [number, number, number] {
    const object = closestObject(p);
    const a = noise(p[0] * 70, p[1] * 70, p[2] * 70);
    const b = noise(p[0] * 210 + 11, p[1] * 210 + 11, p[2] * 210 + 11);
    const c = noise(p[0] * 18 + 5, p[1] * 18 + 5, p[2] * 18 + 5);
    const lows = [
        [0.32, 0.2, 0.12],
        [0.1, 0.16, 0.42],
        [0.42, 0.08, 0.2],
    ];
    const highs = [
        [0.92, 0.74, 0.5],
        [0.45, 0.85, 0.95],
        [1.0, 0.75, 0.3],
    ];
    const t = Math.min(1, Math.max(0, 0.55 * a + 0.3 * b + 0.25 * c - 0.05));
    const n = surfaceNormal(p, object);
    const light = [0.4, 1, 0.3];
    const length = Math.hypot(...light);
    const shading =
        0.6 + 0.4 * Math.max(0, (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / length);
    return [0, 1, 2].map((k) => {
        const value = (lows[object][k] + (highs[object][k] - lows[object][k]) * t) * shading;
        return Math.round(255 * Math.pow(value, 1 / 1.6));
    }) as [number, number, number];
}

export function boxDistance(p: number[]): number {
    const q = [0, 1, 2].map((k) => Math.abs(p[k] - BOX_CENTER[k]) - BOX_HALF[k]);
    const outside = Math.hypot(Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0));
    return outside + Math.min(Math.max(q[0], q[1], q[2]), 0);
}

function sphereDistance(p: number[]): number {
    return (
        Math.hypot(p[0] - SPHERE_CENTER[0], p[1] - SPHERE_CENTER[1], p[2] - SPHERE_CENTER[2]) -
        SPHERE_RADIUS
    );
}

function tableDistance(p: number[]): number {
    const dx = Math.max(0, Math.abs(p[0]) - TABLE_HALF);
    const dz = Math.max(0, Math.abs(p[2]) - TABLE_HALF);
    return Math.hypot(p[1], dx, dz);
}

function closestObject(p: number[]): number {
    const d = [Math.abs(tableDistance(p)), Math.abs(boxDistance(p)), Math.abs(sphereDistance(p))];
    return d.indexOf(Math.min(...d));
}

function surfaceNormal(p: number[], object: number): number[] {
    if (object === 0) return [0, 1, 0];
    if (object === 2) {
        const v = [0, 1, 2].map((k) => p[k] - SPHERE_CENTER[k]);
        const l = Math.hypot(...v);
        return v.map((x) => x / l);
    }
    const q = [0, 1, 2].map((k) => (p[k] - BOX_CENTER[k]) / BOX_HALF[k]);
    const axis = [0, 1, 2].reduce((best, k) => (Math.abs(q[k]) > Math.abs(q[best]) ? k : best), 0);
    const n = [0, 0, 0];
    n[axis] = Math.sign(q[axis]);
    return n;
}

/** Unsigned distance (m) from a point to the nearest scene surface. */
export function surfaceDistance(p: number[]): number {
    return Math.min(
        Math.abs(tableDistance(p)),
        Math.abs(boxDistance(p)),
        Math.abs(sphereDistance(p)),
    );
}

/** Ray depth (along the camera z axis) of the first hit, 0 when the ray escapes. */
export function rayDepth(origin: number[], ray: number[], forward: number[]): number {
    const norm = Math.hypot(ray[0], ray[1], ray[2]);
    const direction = [ray[0] / norm, ray[1] / norm, ray[2] / norm];
    let best = Infinity;
    if (direction[1] < 0) {
        const t = -origin[1] / direction[1];
        const x = origin[0] + direction[0] * t;
        const z = origin[2] + direction[2] * t;
        if (Math.abs(x) < TABLE_HALF && Math.abs(z) < TABLE_HALF) best = t;
    }
    let enter = -Infinity;
    let leave = Infinity;
    for (let k = 0; k < 3; k++) {
        const t0 = (BOX_CENTER[k] - BOX_HALF[k] - origin[k]) / direction[k];
        const t1 = (BOX_CENTER[k] + BOX_HALF[k] - origin[k]) / direction[k];
        enter = Math.max(enter, Math.min(t0, t1));
        leave = Math.min(leave, Math.max(t0, t1));
    }
    if (enter > 0 && enter < leave) best = Math.min(best, enter);
    const oc = [0, 1, 2].map((k) => origin[k] - SPHERE_CENTER[k]);
    const b = oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2];
    const h = b * b - (oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2) + SPHERE_RADIUS ** 2;
    if (h > 0) {
        const t = -b - Math.sqrt(h);
        if (t > 0) best = Math.min(best, t);
    }
    if (!Number.isFinite(best)) return 0;
    return (
        best * (direction[0] * forward[0] + direction[1] * forward[1] + direction[2] * forward[2])
    );
}

/** Visible-surface samples used to measure completeness (box sides + top, sphere cap, table). */
export function surfaceSamples(count: number, random: () => number): number[][] {
    const samples: number[][] = [];
    while (samples.length < count) {
        const kind = random();
        if (kind < 0.4) {
            const face = Math.floor(random() * 5);
            const u = random() * 2 - 1;
            const v = random() * 2 - 1;
            const c = BOX_CENTER;
            const h = BOX_HALF;
            const faces = [
                [c[0] + u * h[0], c[1] + h[1], c[2] + v * h[2]],
                [c[0] + h[0], c[1] + u * h[1], c[2] + v * h[2]],
                [c[0] - h[0], c[1] + u * h[1], c[2] + v * h[2]],
                [c[0] + u * h[0], c[1] + v * h[1], c[2] + h[2]],
                [c[0] + u * h[0], c[1] + v * h[1], c[2] - h[2]],
            ];
            samples.push(faces[face]);
        } else if (kind < 0.7) {
            const z = random() * 0.8 + 0.2;
            const a = random() * Math.PI * 2;
            const r = Math.sqrt(1 - z * z);
            samples.push([
                SPHERE_CENTER[0] + SPHERE_RADIUS * r * Math.cos(a),
                SPHERE_CENTER[1] + SPHERE_RADIUS * z,
                SPHERE_CENTER[2] + SPHERE_RADIUS * r * Math.sin(a),
            ]);
        } else {
            const x = (random() * 2 - 1) * 0.22;
            const z = (random() * 2 - 1) * 0.22;
            const p = [x, 0, z];
            if (boxDistance(p) > 0.004 && sphereDistance(p) > 0.01) samples.push(p);
        }
    }
    return samples;
}
