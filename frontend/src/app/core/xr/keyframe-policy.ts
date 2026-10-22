/** A keyframe every ~6° of view change or a baseline of 10 % of the scene depth. */
const KEYFRAME_ANGLE = (6 * Math.PI) / 180;
const KEYFRAME_BASELINE = 0.1;
/**
 * Keyframes are taken while the phone moves slowly (little motion blur and rolling-shutter
 * skew); when the view has changed a lot already, a moderately fast frame is accepted too.
 */
const SLOW_ANGULAR_SPEED = 0.25;
const SLOW_LINEAR_SPEED = 0.2;
const MAX_ANGULAR_SPEED = 0.5;
const MAX_LINEAR_SPEED = 0.35;
const FORCED_NOVELTY = 1.6;
/** Novelty past which a skipped keyframe is overdue because of the motion. */
const OVERDUE_NOVELTY = 1.3;
/** A pose step this large within one frame is a tracking jump (new segment). */
const JUMP_PER_FRAME = 0.1;

export interface KeyPose {
    id: number;
    /** Camera → world, column-major (WebXR view transform). */
    matrix: Float32Array;
}

export interface CaptureDecision {
    take: boolean;
    /** The user should slow down. */
    tooFast: boolean;
}

/**
 * When to take a keyframe: once the view is new enough compared with every keyframe so far,
 * and the phone is moving slowly enough for a sharp image. Tracks the smoothed speeds.
 */
export class KeyframePolicy {
    poses: KeyPose[] = [];
    private previous: { matrix: Float32Array; time: number } | null = null;
    private angularSpeed = 0;
    private linearSpeed = 0;

    reset(): void {
        this.poses = [];
    }

    forget(id: number): void {
        this.poses = this.poses.filter((pose) => pose.id !== id);
    }

    /** Updates the speeds with a new viewer pose; true if the pose jumped (tracking glitch). */
    move(matrix: Float32Array, time: number): boolean {
        const previous = this.previous;
        this.previous = { matrix: new Float32Array(matrix), time };
        if (!previous) return false;
        const dt = Math.max(1e-3, (time - previous.time) / 1000);
        const step = Math.hypot(
            matrix[12] - previous.matrix[12],
            matrix[13] - previous.matrix[13],
            matrix[14] - previous.matrix[14],
        );
        const angular = rotationBetween(matrix, previous.matrix) / dt;
        this.linearSpeed += 0.3 * (step / dt - this.linearSpeed);
        this.angularSpeed += 0.3 * (angular - this.angularSpeed);
        return dt < 0.2 && step > JUMP_PER_FRAME;
    }

    /** Forgets the motion history (a new AR session). */
    stop(): void {
        this.previous = null;
    }

    decide(matrix: Float32Array, sceneDepth: number): CaptureDecision {
        const novelty = this.novelty(matrix, sceneDepth);
        if (novelty < 1) return { take: false, tooFast: false };
        const slow = this.angularSpeed < SLOW_ANGULAR_SPEED && this.linearSpeed < SLOW_LINEAR_SPEED;
        const acceptable =
            this.angularSpeed < MAX_ANGULAR_SPEED && this.linearSpeed < MAX_LINEAR_SPEED;
        // Wait for a steady moment; only when far past the spacing settle for a faster one.
        const take = slow || (novelty >= FORCED_NOVELTY && acceptable);
        return { take, tooFast: !take && novelty >= OVERDUE_NOVELTY };
    }

    /** Novelty of a pose w.r.t. the keyframes so far (≥ 1 means "take a keyframe"). */
    private novelty(matrix: Float32Array, sceneDepth: number): number {
        const depth = Math.max(0.15, sceneDepth);
        let nearest = Infinity;
        for (const key of this.poses) {
            const m = key.matrix;
            const baseline = Math.hypot(matrix[12] - m[12], matrix[13] - m[13], matrix[14] - m[14]);
            const change = Math.max(
                baseline / (KEYFRAME_BASELINE * depth),
                rotationBetween(matrix, m) / KEYFRAME_ANGLE,
            );
            nearest = Math.min(nearest, change);
        }
        return nearest;
    }
}

/** Angle (rad) between the rotation parts of two column-major rigid transforms. */
export function rotationBetween(a: Float32Array, b: Float32Array): number {
    let trace = 0;
    for (let c = 0; c < 3; c++) {
        trace += a[c * 4] * b[c * 4] + a[c * 4 + 1] * b[c * 4 + 1] + a[c * 4 + 2] * b[c * 4 + 2];
    }
    return Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2)));
}
