import { seededRandom } from '../../src/app/vision/math/dense';
import { JpegImageStore } from '../../src/app/vision/pipeline/jpeg-image-store';
import { ScanSession } from '../../src/app/vision/pipeline/scan-session';
import { CameraState, SceneRenderer, depthGrid, lookAt, xrMatrix } from '../scene-renderer';
import { check, log } from './report';
import { HEIGHT, K, RINGS, TruePose, WIDTH, drifted, jumped } from './trajectory';

export interface Scenario {
    name: string;
    /** Relative error of the focal length the "phone" reports. */
    focalError: number;
    /** Keyframe from which the phone poses jump (relocalisation), or −1. */
    jumpAt: number;
    /** Whether the phone reports the jump as a tracking loss (new segment). */
    reported?: boolean;
    /** Amplitude of the auto-exposure changes between keyframes (0.15 = ±15 % brightness). */
    exposure?: number;
    /** Lens vignetting: brightness loss in the image corners (0.25 = 25 % darker). */
    vignetting?: number;
}

export interface Capture {
    session: ScanSession;
    truth: TruePose[];
    priors: TruePose[];
}

/** Records the two-ring orbit through the app's ScanSession, with drifting phone poses. */
export async function capture(
    renderer: SceneRenderer,
    scenario: Scenario,
    seed: number,
): Promise<Capture> {
    const random = seededRandom(seed);
    const session = new ScanSession(new JpegImageStore());
    const truth: TruePose[] = [];
    const priors: TruePose[] = [];
    const reported = {
        ...K,
        fx: K.fx * (1 + scenario.focalError),
        fy: K.fy * (1 + scenario.focalError),
    };
    let index = 0;
    const started = performance.now();
    for (const ring of RINGS) {
        for (let i = 0; i < ring.count; i++, index++) {
            const angle = (i / ring.count) * Math.PI * 2;
            const pose = lookAt([
                Math.sin(angle) * ring.radius,
                ring.height,
                Math.cos(angle) * ring.radius,
            ]);
            truth.push(pose);
            const camera: CameraState = { ...pose, k: K };
            const pixels = renderer.capture(camera, WIDTH, HEIGHT);
            if (scenario.exposure || scenario.vignetting) {
                // Auto exposure drifting slowly up and down as the phone turns, and a radial
                // falloff towards the corners that no per-view gain can undo.
                const gain = 1 + (scenario.exposure ?? 0) * Math.sin((index / 9) * Math.PI + 0.4);
                const falloff = scenario.vignetting ?? 0;
                const half = Math.hypot(WIDTH / 2, HEIGHT / 2);
                for (let y = 0; y < HEIGHT; y++) {
                    for (let x = 0; x < WIDTH; x++) {
                        const r2 = ((x - WIDTH / 2) ** 2 + (y - HEIGHT / 2) ** 2) / (half * half);
                        const g = gain * (1 - falloff * r2);
                        const i = (y * WIDTH + x) * 4;
                        for (let c = 0; c < 3; c++) {
                            pixels[i + c] = Math.min(255, Math.round(pixels[i + c] * g));
                        }
                    }
                }
            }
            const grid = depthGrid(camera, WIDTH, HEIGHT, 64, 36, () => (random() - 0.5) * 0.02);
            let prior = drifted(pose, index, random);
            if (scenario.jumpAt >= 0 && index >= scenario.jumpAt) prior = jumped(prior);
            priors.push(prior);
            const verdict = await session.addKeyframe({
                time: index * 0.5,
                image: { width: WIDTH, height: HEIGHT, data: pixels },
                intrinsics: reported,
                xrTransform: xrMatrix(prior.R, prior.center),
                depth: { width: 64, height: 36, meters: grid },
                segment: scenario.reported && index >= scenario.jumpAt ? 1 : 0,
            });
            if (!verdict.accepted)
                log.push(`${scenario.name}: keyframe ${index} rejected: ${verdict.reason}`);
        }
    }
    check(
        `${scenario.name}: capture keeps every sharp keyframe`,
        session.keyframes.length === index,
        `${session.keyframes.length}/${index} keyframes, ${session.map.points(2).ids.length} live sparse points, ${((performance.now() - started) / index).toFixed(0)} ms per keyframe, ${session.fallbacks} rescued / ${session.realigned} re-anchored`,
    );
    return { session, truth, priors };
}
