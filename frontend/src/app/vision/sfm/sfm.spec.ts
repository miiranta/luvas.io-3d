import { DESCRIPTOR_WORDS, FeatureSet, ORB_METRIC } from '../features/orb-features';
import {
    Intrinsics,
    Pose,
    cameraCenter,
    clonePose,
    glViewMatrix,
    intrinsicsFromProjection,
    poseFromXrTransform,
    project,
} from '../geometry/camera';
import { triangulate } from '../geometry/triangulate';
import { bruteForceMatch } from '../matching/brute-force-matcher';
import { fundamentalInliers } from '../matching/fundamental-ransac';
import { guidedMatch } from '../matching/guided-matcher';
import { seededRandom, symmetricEigen } from '../math/dense';
import { cross3, expSO3, multiply3, normalize3, rotationAngle, sub3 } from '../math/mat3';
import { bundleAdjust } from './bundle-adjuster';

const K: Intrinsics = { fx: 500, fy: 500, cx: 319.5, cy: 239.5 };

/** Camera at `center` looking at `target` with world +y up (CV axes: y down). */
function lookAt(center: number[], target: number[]): Pose {
    const z = normalize3(sub3(target, center));
    const x = normalize3(cross3(z, [0, 1, 0]));
    const y = cross3(z, x);
    const R = new Float64Array([...x, ...y, ...z]);
    const t = new Float64Array([
        -(R[0] * center[0] + R[1] * center[1] + R[2] * center[2]),
        -(R[3] * center[0] + R[4] * center[1] + R[5] * center[2]),
        -(R[6] * center[0] + R[7] * center[1] + R[8] * center[2]),
    ]);
    return { R, t };
}

function orbit(count: number, radius: number): Pose[] {
    return Array.from({ length: count }, (_, i) => {
        const a = (i / count) * Math.PI * 1.2;
        return lookAt([Math.sin(a) * radius, 0.25, Math.cos(a) * radius], [0, 0, 0]);
    });
}

function cloud(count: number, random: () => number): Float64Array {
    const points = new Float64Array(count * 3);
    for (let i = 0; i < count * 3; i++) points[i] = (random() - 0.5) * 0.4;
    return points;
}

describe('camera conventions', () => {
    it('round-trips a pose through the WebXR camera-to-world matrix', () => {
        const pose = lookAt([0.3, 0.2, 0.8], [0, 0, 0]);
        const view = glViewMatrix(pose);
        // Invert the rigid GL view matrix to get the XR-style camera → world transform.
        const r = [
            view[0],
            view[4],
            view[8],
            view[1],
            view[5],
            view[9],
            view[2],
            view[6],
            view[10],
        ];
        const t = [view[12], view[13], view[14]];
        const world = new Float64Array(16);
        // Camera → world = [Rᵀ | −Rᵀ t]; element (row, col) of Rᵀ is r[col * 3 + row].
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) world[col * 4 + row] = r[col * 3 + row];
            world[12 + row] = -(r[row] * t[0] + r[3 + row] * t[1] + r[6 + row] * t[2]);
        }
        world[15] = 1;
        const back = poseFromXrTransform(world);
        expect(rotationAngle(back.R, pose.R)).toBeLessThan(1e-6);
        expect(Math.hypot(...sub3(cameraCenter(back), cameraCenter(pose)))).toBeLessThan(1e-6);
    });

    it('recovers intrinsics from an OpenGL projection matrix', () => {
        const width = 360;
        const height = 640;
        const k = { fx: 410, fy: 412, cx: 177.5, cy: 322.25 };
        // Pixel-centre convention: continuous coordinate = index + 0.5.
        const projection = new Float64Array(16);
        projection[0] = (2 * k.fx) / width;
        projection[5] = (2 * k.fy) / height;
        projection[8] = 1 - (2 * (k.cx + 0.5)) / width;
        projection[9] = (2 * (k.cy + 0.5)) / height - 1;
        projection[11] = -1;
        const back = intrinsicsFromProjection(projection, width, height);
        expect(back.fx).toBeCloseTo(k.fx, 9);
        expect(back.fy).toBeCloseTo(k.fy, 9);
        expect(back.cx).toBeCloseTo(k.cx, 9);
        expect(back.cy).toBeCloseTo(k.cy, 9);
    });
});

describe('triangulation', () => {
    it('is exact without noise and reports the ray angle', () => {
        const poses = orbit(4, 1);
        const truth = [0.05, -0.03, 0.02];
        const result = triangulate(
            poses.map((pose) => {
                const [u, v] = project(K, pose, truth);
                return { k: K, pose, u, v };
            }),
        );
        expect(result).not.toBeNull();
        expect(Math.hypot(...sub3(result!.point, truth))).toBeLessThan(1e-9);
        expect(result!.angle).toBeGreaterThan(0.5);
    });
});

describe('symmetric eigen-decomposition', () => {
    it('diagonalises a random symmetric matrix', () => {
        const random = seededRandom(3);
        const n = 9;
        const a = new Float64Array(n * n);
        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) a[i * n + j] = a[j * n + i] = random() - 0.5;
        }
        const { values, vectors } = symmetricEigen(a, n);
        for (let e = 0; e < n; e++) {
            for (let i = 0; i < n; i++) {
                let av = 0;
                for (let j = 0; j < n; j++) av += a[i * n + j] * vectors[e * n + j];
                expect(av).toBeCloseTo(values[e] * vectors[e * n + i], 8);
            }
        }
        for (let e = 1; e < n; e++) expect(values[e]).toBeGreaterThanOrEqual(values[e - 1]);
    });
});

/** Observations of the points by the true cameras, with ±0.3 px noise. */
function observe(truth: Pose[], points: Float64Array, k: Intrinsics, random: () => number) {
    const camera: number[] = [];
    const point: number[] = [];
    const u: number[] = [];
    const v: number[] = [];
    for (let p = 0; p < points.length / 3; p++) {
        const x = points.subarray(p * 3, p * 3 + 3);
        truth.forEach((pose, c) => {
            const [pu, pv, z] = project(k, pose, x);
            if (z <= 0 || pu < 0 || pu > 640 || pv < 0 || pv > 480) return;
            camera.push(c);
            point.push(p);
            u.push(pu + (random() - 0.5) * 0.6);
            v.push(pv + (random() - 0.5) * 0.6);
        });
    }
    return {
        camera: Int32Array.from(camera),
        point: Int32Array.from(point),
        u: Float64Array.from(u),
        v: Float64Array.from(v),
    };
}

describe('bundle adjustment', () => {
    it('self-calibrates a wrong focal length', () => {
        const random = seededRandom(3);
        const truth = orbit(16, 0.9);
        const points = cloud(400, random);
        const reported = { ...K, fx: K.fx * 1.02, fy: K.fy * 1.02 };
        const noisyPoints = new Float64Array(points);
        for (let i = 0; i < noisyPoints.length; i++) noisyPoints[i] += (random() - 0.5) * 0.01;
        const problem = {
            intrinsics: truth.map(() => reported),
            poses: truth.map(clonePose),
            priors: truth.map(clonePose),
            points: noisyPoints,
            ...observe(truth, points, K, random),
        };
        const fixed = bundleAdjust(
            { ...problem, poses: truth.map(clonePose), points: new Float64Array(noisyPoints) },
            { iterations: 30 },
        );
        const report = bundleAdjust(problem, { iterations: 30, calibrateFocal: true });
        expect(Math.abs(report.focalScale - 1 / 1.02)).toBeLessThan(0.002);
        expect(report.finalRms).toBeLessThan(fixed.finalRms);
    });

    it('does not let a tracking jump across a break drag the poses', () => {
        const random = seededRandom(4);
        const truth = orbit(16, 0.9);
        const points = cloud(400, random);
        // The phone poses jump 3 cm after keyframe 8 (relocalisation) and stay shifted.
        const priors = truth.map((pose, i) =>
            i < 8
                ? clonePose(pose)
                : { R: pose.R, t: new Float64Array([pose.t[0] + 0.03, pose.t[1], pose.t[2]]) },
        );
        const noisyPoints = new Float64Array(points);
        for (let i = 0; i < noisyPoints.length; i++) noisyPoints[i] += (random() - 0.5) * 0.01;
        const poses = truth.map(clonePose);
        bundleAdjust(
            {
                intrinsics: truth.map(() => K),
                poses,
                priors,
                points: noisyPoints,
                ...observe(truth, points, K, random),
                breaks: new Set([8]),
            },
            { iterations: 30 },
        );
        for (const i of [7, 8, 15]) {
            expect(
                Math.hypot(...sub3(cameraCenter(poses[i]), cameraCenter(truth[i]))),
            ).toBeLessThan(0.003);
        }
    });

    it('removes drift from the phone poses while keeping metric scale', () => {
        const random = seededRandom(11);
        const truth = orbit(16, 0.9);
        const points = cloud(400, random);
        // Phone poses drift slowly: a growing rotation and translation error.
        const priors: Pose[] = truth.map((pose, i) => {
            const drift = expSO3([0.0012 * i, -0.0008 * i, 0.0005 * i]);
            const R = multiply3(drift, pose.R);
            const t = new Float64Array([
                pose.t[0] + 0.0015 * i,
                pose.t[1] - 0.001 * i,
                pose.t[2] + 0.0008 * i,
            ]);
            return { R, t };
        });
        // Odometry is consistent between neighbours, so make the priors' relative motion exact
        // except for the drift above and start BA from them.
        const camera: number[] = [];
        const point: number[] = [];
        const u: number[] = [];
        const v: number[] = [];
        for (let p = 0; p < points.length / 3; p++) {
            const x = points.subarray(p * 3, p * 3 + 3);
            truth.forEach((pose, c) => {
                const [pu, pv, z] = project(K, pose, x);
                if (z <= 0 || pu < 0 || pu > 640 || pv < 0 || pv > 480) return;
                camera.push(c);
                point.push(p);
                u.push(pu + (random() - 0.5) * 0.6);
                v.push(pv + (random() - 0.5) * 0.6);
            });
        }
        const poses = priors.map(clonePose);
        const noisyPoints = new Float64Array(points);
        for (let i = 0; i < noisyPoints.length; i++) noisyPoints[i] += (random() - 0.5) * 0.01;
        // Pin the first camera to the truth so that errors are measured in the same gauge.
        poses[0] = clonePose(truth[0]);
        priors[0] = clonePose(truth[0]);
        const report = bundleAdjust(
            {
                intrinsics: truth.map(() => K),
                poses,
                priors,
                points: noisyPoints,
                camera: Int32Array.from(camera),
                point: Int32Array.from(point),
                u: Float64Array.from(u),
                v: Float64Array.from(v),
            },
            { iterations: 30 },
        );
        expect(report.finalRms).toBeLessThan(0.5);
        expect(report.finalRms).toBeLessThan(report.initialRms / 4);
        const before = Math.hypot(...sub3(cameraCenter(priors[15]), cameraCenter(truth[15])));
        const after = Math.hypot(...sub3(cameraCenter(poses[15]), cameraCenter(truth[15])));
        expect(after).toBeLessThan(before / 3);
        // Scale comes only from the (drifting) odometry, so allow ~1 % of the orbit radius.
        expect(after).toBeLessThan(0.01);
        expect(rotationAngle(poses[15].R, truth[15].R)).toBeLessThan(
            rotationAngle(priors[15].R, truth[15].R) / 5,
        );
    });
});

describe('guided matching', () => {
    function featureSet(
        pose: Pose,
        points: Float64Array,
        words: Uint32Array,
        random: () => number,
        distractors: number,
    ): { set: FeatureSet; ids: number[] } {
        const xs: number[] = [];
        const ys: number[] = [];
        const ids: number[] = [];
        const descriptors: number[] = [];
        for (let p = 0; p < points.length / 3; p++) {
            const [u, v, z] = project(K, pose, points.subarray(p * 3, p * 3 + 3));
            if (z <= 0 || u < 0 || u >= 640 || v < 0 || v >= 480) continue;
            xs.push(u + (random() - 0.5) * 0.8);
            ys.push(v + (random() - 0.5) * 0.8);
            ids.push(p);
            for (let w = 0; w < DESCRIPTOR_WORDS; w++) {
                // Flip a couple of bits per view to mimic appearance change.
                const noise = random() < 0.5 ? 1 << Math.floor(random() * 32) : 0;
                descriptors.push((words[p * DESCRIPTOR_WORDS + w] ^ noise) >>> 0);
            }
        }
        for (let d = 0; d < distractors; d++) {
            xs.push(random() * 640);
            ys.push(random() * 480);
            ids.push(-1);
            for (let w = 0; w < DESCRIPTOR_WORDS; w++) descriptors.push((random() * 2 ** 32) >>> 0);
        }
        const count = xs.length;
        return {
            ids,
            set: {
                count,
                x: Float32Array.from(xs),
                y: Float32Array.from(ys),
                level: new Uint8Array(count),
                angle: new Float32Array(count),
                descriptors: Uint32Array.from(descriptors),
                metric: ORB_METRIC,
            },
        };
    }

    it('finds correct correspondences along the epipolar band', () => {
        const random = seededRandom(5);
        const points = cloud(600, random);
        const words = new Uint32Array(600 * DESCRIPTOR_WORDS);
        for (let i = 0; i < words.length; i++) words[i] = (random() * 2 ** 32) >>> 0;
        const [a, , b] = orbit(12, 0.9);
        const fa = featureSet(a, points, words, random, 300);
        const fb = featureSet(b, points, words, random, 300);
        const matches = guidedMatch(fa.set, fb.set, K, K, a, b, 640, 480, {
            band: 3,
            minDepth: 0.1,
            maxDepth: 5,
        });
        let correct = 0;
        for (let m = 0; m < matches.length; m += 2) {
            const id = fa.ids[matches[m]];
            if (id >= 0 && id === fb.ids[matches[m + 1]]) correct++;
        }
        expect(matches.length / 2).toBeGreaterThan(400);
        expect(correct / (matches.length / 2)).toBeGreaterThan(0.98);

        const robust = fundamentalInliers(fa.set, fb.set, matches, 1.5);
        expect(robust.length / 2).toBeGreaterThan(0.9 * correct);
    });

    it('matches without poses (all against all) when the pose cannot be trusted', () => {
        const random = seededRandom(6);
        const points = cloud(600, random);
        const words = new Uint32Array(600 * DESCRIPTOR_WORDS);
        for (let i = 0; i < words.length; i++) words[i] = (random() * 2 ** 32) >>> 0;
        const [a, , b] = orbit(12, 0.9);
        const fa = featureSet(a, points, words, random, 300);
        const fb = featureSet(b, points, words, random, 300);
        const matches = fundamentalInliers(fa.set, fb.set, bruteForceMatch(fa.set, fb.set), 2);
        let correct = 0;
        for (let m = 0; m < matches.length; m += 2) {
            const id = fa.ids[matches[m]];
            if (id >= 0 && id === fb.ids[matches[m + 1]]) correct++;
        }
        expect(matches.length / 2).toBeGreaterThan(300);
        expect(correct / (matches.length / 2)).toBeGreaterThan(0.98);
    });
});
