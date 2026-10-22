/**
 * Numerical check of the whole reconstruction pipeline on the synthetic scene, run inside
 * headless Chrome (WebGPU) by tools/verify.mjs; the pieces live in tools/verify/. Keyframes are rendered from a two-ring orbit,
 * the "phone" poses get a slow drift, then capture + reconstruction run exactly as in the app
 * and the outputs are scored against the analytic ground truth.
 */
import { QUALITY, Quality } from '../src/app/core/models/quality';
import { TEXTURE_METHODS, atlasMethod } from '../src/app/vision/texture/methods';
import { requestGpu } from '../src/app/vision/gpu/gpu-device';
import { seededRandom } from '../src/app/vision/math/dense';
import { SceneRenderer } from './scene-renderer';
import { Scenario, capture } from './verify/capture';
import { Check, check, checks, log } from './verify/report';
import { LIMITS, Score, score } from './verify/score';
import { siftCheck } from './verify/sift-check';
import { stereoCheck } from './verify/stereo-check';
import { HEIGHT, WIDTH } from './verify/trajectory';

declare global {
    interface Window {
        __verify?: { checks: Check[]; log: string[]; error?: string };
    }
}

/**
 * A hard capture: the phone reports a focal length 2 % off, its poses jump 4 cm / 1.5° in the
 * middle of the first ring without saying so, auto exposure changes the brightness ±15 % and
 * the lens darkens the corners by 25 %.
 */
const HARD: Scenario = {
    name: 'hard',
    focalError: 0.02,
    jumpAt: 18,
    exposure: 0.15,
    vignetting: 0.25,
};

async function run(): Promise<void> {
    const random = seededRandom(7);

    // WebGPU first, as in the app (the worker owns the device before any WebGL context exists).
    const gpu = await requestGpu();
    log.push(`gpu: ${gpu.label}`);
    const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    const gl = canvas.getContext('webgl2', {
        preserveDrawingBuffer: true,
    }) as WebGL2RenderingContext;
    const renderer = new SceneRenderer(gl);
    // `?quick`: only the High builds of the orbit and the hard capture (for tuning).
    if (new URLSearchParams(location.search).has('quick')) {
        await siftCheck(renderer, gpu);
        (globalThis as { __probe?: string[] }).__probe = log;
        const orbit = await capture(renderer, { name: 'orbit', focalError: 0, jumpAt: -1 }, 7);
        const hard = await capture(renderer, HARD, 11);
        const query = new URLSearchParams(location.search);
        if (query.has('truth')) {
            // The pipeline's floor: no bundle adjustment, the true poses.
            const settings = { ...QUALITY.high.bundle };
            QUALITY.high.bundle = { ...settings, features: 'orb', rounds: 0 };
            for (const [name, capture] of [['orbit', orbit]] as const) {
                const snapshot = capture.session.map.snapshot();
                snapshot.poses = capture.truth.map((pose) => ({
                    R: new Float64Array(pose.R),
                    t: new Float64Array(
                        [0, 1, 2].map(
                            (r) =>
                                -(
                                    pose.R[r * 3] * pose.center[0] +
                                    pose.R[r * 3 + 1] * pose.center[1] +
                                    pose.R[r * 3 + 2] * pose.center[2]
                                ),
                        ),
                    ),
                }));
                capture.session.baseline = snapshot;
                await score(gpu, capture, 'high', `${name} true poses`, LIMITS.high, random);
                capture.session.baseline = null;
            }
            QUALITY.high.bundle = settings;
        }
        // Variants of High: ?vary=geometric:1,3,5;filter.views:8,12 (dotted paths into the
        // profile; mrf.smoothness rebuilds the MRF texture method).
        const axes = (query.get('vary') ?? '')
            .split(';')
            .filter(Boolean)
            .map((axis) => {
                const [path, values] = axis.split(':');
                return { path, values: values.split(',') };
            });
        const high = structuredClone(QUALITY.high);
        const variants = axes.reduce<Record<string, string>[]>(
            (all, axis) =>
                all.flatMap((v) => axis.values.map((value) => ({ ...v, [axis.path]: value }))),
            [{}],
        );
        for (const variant of variants) {
            Object.assign(QUALITY.high, structuredClone(high));
            for (const [path, raw] of Object.entries(variant)) {
                const value =
                    raw === 'true'
                        ? true
                        : raw === 'false'
                          ? false
                          : isNaN(Number(raw))
                            ? raw
                            : Number(raw);
                if (path === 'mrf.seams') {
                    TEXTURE_METHODS.mrf = atlasMethod({
                        seams: value === true,
                        selection: { kind: 'mrf', smoothness: 1 },
                    });
                    continue;
                }
                const keys = path.split('.');
                let target = QUALITY.high as unknown as Record<string, unknown>;
                for (const key of keys.slice(0, -1))
                    target = target[key] as Record<string, unknown>;
                target[keys.at(-1) as string] = value;
            }
            const name =
                Object.entries(variant)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(' ') || 'default';
            await score(gpu, orbit, 'high', `orbit ${name}`, LIMITS.high, random);
            await score(
                gpu,
                hard,
                'high',
                `hard ${name}`,
                { ...LIMITS.high, dense: [2, 6], mesh: [2, 5] },
                random,
            );
        }
        Object.assign(QUALITY.high, high);
        return;
    }
    await stereoCheck(renderer, gpu);
    await siftCheck(renderer, gpu);

    // Every quality level on the same capture (also checks that rebuilds start from scratch).
    const clean = await capture(renderer, { name: 'orbit', focalError: 0, jumpAt: -1 }, 7);
    const results: Record<string, Score> = {};
    for (const quality of ['high', 'medium', 'low'] as Quality[]) {
        results[quality] = await score(gpu, clean, quality, 'orbit', LIMITS[quality], random);
    }
    check(
        'orbit: rebuilding High after the other levels gives the same model',
        Math.abs(
            (await score(gpu, clean, 'high', 'orbit (again)', LIMITS.high, random)).meshMedian -
                results['high'].meshMedian,
        ) < 0.05,
        'mesh error identical within 0.05 mm',
    );
    check(
        'orbit: slower levels give a more accurate shape',
        results['high'].shape < results['medium'].shape &&
            results['medium'].shape <= results['low'].shape + 0.01 &&
            results['high'].seconds > results['low'].seconds,
        ['high', 'medium', 'low']
            .map(
                (q) =>
                    `${q} shape ${results[q].shape.toFixed(3)} mm / ${results[q].seconds.toFixed(1)} s`,
            )
            .join(', '),
    );

    const hard = await capture(renderer, HARD, 11);
    check(
        'hard: the pose jump is detected and the segment re-anchored',
        hard.session.realigned >= 1 &&
            hard.session.map.breaks.has(18) &&
            hard.session.map.breaks.size <= 2,
        `breaks at ${[...hard.session.map.breaks].join(', ') || 'none'}, ${hard.session.fallbacks} rescued, ${hard.session.realigned} re-anchored`,
    );
    const hardHigh = await score(
        gpu,
        hard,
        'high',
        'hard',
        { ...LIMITS.high, dense: [2, 6], mesh: [2, 5] },
        random,
    );
    check(
        'hard: High recovers the focal-length error',
        Math.abs(hardHigh.focal - (1 / 1.02 - 1)) < 0.006,
        `correction ${(hardHigh.focal * 100).toFixed(2)} % (true ${((1 / 1.02 - 1) * 100).toFixed(2)} %)`,
    );
    const hardMedium = await score(
        gpu,
        hard,
        'medium',
        'hard',
        {
            ...LIMITS.medium,
            drift: null,
            reprojection: null,
            dense: [5, 15],
            mesh: [5, 15],
            box: 12,
            completeness: 0.7,
            color: 25,
        },
        random,
    );
    check(
        'hard: focal self-calibration (High) beats fixed intrinsics (Medium)',
        hardHigh.box < hardMedium.box && hardHigh.shape < hardMedium.shape,
        `box error ${hardHigh.box.toFixed(1)} vs ${hardMedium.box.toFixed(1)} mm, shape ${hardHigh.shape.toFixed(3)} vs ${hardMedium.shape.toFixed(3)} mm`,
    );
    check(
        'hard: MRF views + seam levelling (High) hide the vignetting seams of Medium',
        hardHigh.seams < hardMedium.seams * 0.6,
        `seam colour step p90 ${hardHigh.seams.toFixed(1)} vs ${hardMedium.seams.toFixed(1)} (0–255)`,
    );

    // Tracking lost and recovered with a jump, as the phone reports it (new segment).
    const lost = await capture(
        renderer,
        { name: 'lost', focalError: 0, jumpAt: 42, reported: true },
        13,
    );
    check(
        'lost: the reported tracking loss breaks the odometry and re-anchors the segment',
        lost.session.map.breaks.has(42) && lost.session.realigned >= 1,
        `breaks at ${[...lost.session.map.breaks].join(', ')}, ${lost.session.realigned} re-anchored`,
    );
    await score(gpu, lost, 'high', 'lost', LIMITS.high, random);
}

run().then(
    () => (window.__verify = { checks, log }),
    (error: unknown) =>
        (window.__verify = {
            checks,
            log,
            error: error instanceof Error ? `${error.message}\n${error.stack}` : String(error),
        }),
);
