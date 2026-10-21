import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    OnDestroy,
    afterNextRender,
    computed,
    effect,
    inject,
    output,
    signal,
    viewChild,
} from '@angular/core';
import { CAMERA_STRIDE } from '../../core/models/scan-results';
import { ScanService } from '../../core/services/scan-service';
import { cameraLines } from '../../vision/export/camera-lines';
import { GlScene } from '../../render/gl-scene';
import { OrbitCamera, pickMesh } from '../../render/orbit-camera';

type Mode = 'mesh' | 'dense' | 'sparse';

interface Pointer {
    x: number;
    y: number;
}

const TAP_PIXELS = 8;
const TAP_MS = 350;

@Component({
    selector: 'app-model-viewer',
    templateUrl: './model-viewer.html',
    styleUrl: './model-viewer.scss',
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { '(window:resize)': 'resize()' },
})
export class ModelViewer implements OnDestroy {
    private readonly scan = inject(ScanService);
    private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
    private scene: GlScene | null = null;
    private readonly camera = new OrbitCamera();
    private readonly pointers = new Map<number, Pointer>();
    private gesture: { start: number; x: number; y: number; moved: boolean } | null = null;
    private frame: number | null = null;
    private framedWith: 'none' | 'sparse' | 'dense' | 'mesh' = 'none';
    private interacted = false;
    private extent = 1;

    readonly openInsights = output<void>();
    readonly openExport = output<void>();
    readonly requestReset = output<void>();

    readonly chosen = signal<Mode | null>(null);
    readonly showCameras = signal(true);
    readonly measuring = signal(false);
    readonly marks = signal<number[][]>([]);
    readonly labelAt = signal<{ x: number; y: number } | null>(null);

    readonly hasMesh = computed(() => this.scan.mesh() !== null || this.scan.textured() !== null);
    readonly hasDense = computed(() => this.scan.dense() !== null);
    readonly hasSparse = computed(() => this.scan.sparse() !== null);
    readonly mode = computed<Mode>(() => {
        const chosen = this.chosen();
        if (chosen) return chosen;
        if (this.hasMesh()) return 'mesh';
        if (this.hasDense()) return 'dense';
        return 'sparse';
    });
    readonly canExport = computed(() => this.scan.phase() === 'done');
    readonly error = this.scan.error;
    readonly summary = computed(() => {
        const mode = this.mode();
        if (mode === 'mesh') {
            const mesh = this.scan.textured() ?? this.scan.mesh();
            return mesh ? `${formatCount(mesh.indices.length / 3)} triangles` : 'meshing…';
        }
        if (mode === 'dense') {
            const dense = this.scan.dense();
            return dense
                ? `${formatCount(dense.positions.length / 3)} dense points`
                : 'densifying…';
        }
        const sparse = this.scan.sparse();
        const cameras = (this.scan.cameras()?.length ?? 0) / CAMERA_STRIDE;
        return sparse
            ? `${formatCount(sparse.positions.length / 3)} points · ${cameras} cameras`
            : 'adjusting…';
    });
    readonly distance = computed(() => {
        const [a, b] = this.marks();
        if (!a || !b) return null;
        const meters = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        return meters < 0.1
            ? `${(meters * 1000).toFixed(1)} mm`
            : `${(meters * 100).toFixed(1)} cm`;
    });

    constructor() {
        afterNextRender(() => {
            const gl = this.canvas().nativeElement.getContext('webgl2', {
                antialias: true,
                alpha: false,
            });
            if (!gl) {
                this.scan.error.set('WebGL 2 is not available for the viewer.');
                return;
            }
            this.scene = new GlScene(gl);
            this.canvas().nativeElement.addEventListener('webglcontextlost', (event) =>
                event.preventDefault(),
            );
            this.resize();
            this.uploadAll();
        });
        effect(() => {
            this.scan.sparse();
            this.scan.dense();
            this.scan.mesh();
            this.scan.textured();
            this.scan.cameras();
            this.uploadAll();
        });
        effect(() => {
            this.mode();
            this.showCameras();
            this.marks();
            this.requestRender();
        });
    }

    private uploadAll(): void {
        const scene = this.scene;
        if (!scene) return;
        const sparse = this.scan.sparse();
        const dense = this.scan.dense();
        const mesh = this.scan.mesh();
        const textured = this.scan.textured();
        const cameras = this.scan.cameras();
        scene.setColored('sparse', sparse && { ...sparse, channels: 3 });
        scene.setColored('dense', dense && { ...dense, channels: 3 });
        if (textured) {
            scene.setMesh('mesh', textured.positions, textured.indices, textured.uvs);
            scene.setAtlas(textured.atlas);
        } else if (mesh) {
            scene.setMesh('mesh', mesh.positions, mesh.indices, null);
        } else {
            scene.remove('mesh');
        }
        // Frame the best geometry so far until the user moves the view.
        const best = mesh ? 'mesh' : dense ? 'dense' : sparse ? 'sparse' : 'none';
        if (!this.interacted && best !== this.framedWith && best !== 'none') {
            const positions = (mesh ?? dense ?? sparse)!.positions;
            this.camera.frame(positions);
            this.extent = this.camera.distance * Math.sin(this.camera.fov / 2);
            this.framedWith = best;
        }
        scene.setColored(
            'cameras',
            cameras && { ...cameraLines(cameras, this.extent * 0.05), channels: 4 },
        );
        this.requestRender();
    }

    resize(): void {
        const canvas = this.canvas().nativeElement;
        const ratio = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.round(canvas.clientWidth * ratio);
        canvas.height = Math.round(canvas.clientHeight * ratio);
        this.requestRender();
    }

    private requestRender(): void {
        if (this.frame !== null || typeof requestAnimationFrame === 'undefined') return;
        this.frame = requestAnimationFrame(() => {
            this.frame = null;
            this.render();
        });
    }

    private render(): void {
        const scene = this.scene;
        if (!scene) return;
        const gl = scene.gl;
        const canvas = this.canvas().nativeElement;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(27 / 255, 18 / 255, 53 / 255, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        const aspect = canvas.width / Math.max(1, canvas.height);
        const viewProjection = this.camera.viewProjection(aspect);
        const scale = canvas.height / (2 * Math.tan(this.camera.fov / 2));
        const mode = this.mode();
        if (mode === 'mesh') scene.drawMesh('mesh', viewProjection, this.camera.eye(), true);
        else if (mode === 'dense')
            scene.drawPoints('dense', viewProjection, this.extent * 0.005, scale);
        else scene.drawPoints('sparse', viewProjection, this.extent * 0.012, scale);
        if (this.showCameras()) scene.drawLines('cameras', viewProjection);
        if (this.marks().length > 0) {
            gl.disable(gl.DEPTH_TEST);
            scene.drawLines('measure-line', viewProjection);
            scene.drawPoints('measure', viewProjection, this.extent * 0.025, scale);
        }
        this.placeLabel(viewProjection, canvas);
    }

    private placeLabel(viewProjection: Float32Array, canvas: HTMLCanvasElement): void {
        const [a, b] = this.marks();
        if (!a || !b) {
            if (this.labelAt()) this.labelAt.set(null);
            return;
        }
        const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
        const m = viewProjection;
        const x = m[0] * mid[0] + m[4] * mid[1] + m[8] * mid[2] + m[12];
        const y = m[1] * mid[0] + m[5] * mid[1] + m[9] * mid[2] + m[13];
        const w = m[3] * mid[0] + m[7] * mid[1] + m[11] * mid[2] + m[15];
        if (w <= 0) return;
        this.labelAt.set({
            x: ((x / w + 1) / 2) * canvas.clientWidth,
            y: ((1 - y / w) / 2) * canvas.clientHeight,
        });
    }

    choose(mode: Mode): void {
        this.chosen.set(mode);
    }

    toggleCameras(): void {
        this.showCameras.update((show) => !show);
    }

    toggleMeasure(): void {
        this.measuring.update((on) => !on);
        this.setMarks([]);
    }

    reframe(): void {
        const geometry =
            this.scan.textured() ?? this.scan.mesh() ?? this.scan.dense() ?? this.scan.sparse();
        if (geometry) this.camera.frame(geometry.positions);
        this.requestRender();
    }

    down(event: PointerEvent): void {
        this.canvas().nativeElement.setPointerCapture(event.pointerId);
        this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        this.gesture =
            this.pointers.size === 1
                ? { start: performance.now(), x: event.clientX, y: event.clientY, moved: false }
                : null;
    }

    move(event: PointerEvent): void {
        const previous = this.pointers.get(event.pointerId);
        if (!previous) return;
        const height = this.canvas().nativeElement.clientHeight;
        if (this.pointers.size === 1) {
            const dx = event.clientX - previous.x;
            const dy = event.clientY - previous.y;
            if (
                this.gesture &&
                Math.hypot(event.clientX - this.gesture.x, event.clientY - this.gesture.y) >
                    TAP_PIXELS
            ) {
                this.gesture.moved = true;
            }
            if (event.buttons & 6 || event.shiftKey) this.camera.pan(dx, dy, height);
            else this.camera.rotate(dx, dy);
        } else if (this.pointers.size === 2) {
            const [a, b] = [...this.pointers.values()];
            const before = Math.hypot(a.x - b.x, a.y - b.y);
            const midBefore = [(a.x + b.x) / 2, (a.y + b.y) / 2];
            this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
            const [c, d] = [...this.pointers.values()];
            const after = Math.hypot(c.x - d.x, c.y - d.y);
            const midAfter = [(c.x + d.x) / 2, (c.y + d.y) / 2];
            if (after > 1) this.camera.zoom(before / after);
            this.camera.pan(midAfter[0] - midBefore[0], midAfter[1] - midBefore[1], height);
        }
        this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        this.interacted = true;
        this.requestRender();
    }

    up(event: PointerEvent): void {
        this.pointers.delete(event.pointerId);
        const gesture = this.gesture;
        this.gesture = null;
        if (!gesture || gesture.moved || performance.now() - gesture.start > TAP_MS) return;
        if (this.measuring()) this.pick(event);
    }

    wheel(event: WheelEvent): void {
        event.preventDefault();
        this.camera.zoom(Math.exp(event.deltaY * 0.0012));
        this.interacted = true;
        this.requestRender();
    }

    private pick(event: PointerEvent): void {
        const mesh = this.scan.textured() ?? this.scan.mesh();
        if (!mesh) return;
        const canvas = this.canvas().nativeElement;
        const box = canvas.getBoundingClientRect();
        const ndcX = ((event.clientX - box.left) / box.width) * 2 - 1;
        const ndcY = 1 - ((event.clientY - box.top) / box.height) * 2;
        const ray = this.camera.ray(ndcX, ndcY, canvas.width / Math.max(1, canvas.height));
        if (!ray) return;
        const hit = pickMesh(mesh.positions, mesh.indices, ray.origin, ray.direction);
        if (!hit) return;
        const marks = this.marks().length >= 2 ? [hit] : [...this.marks(), hit];
        this.setMarks(marks);
    }

    private setMarks(marks: number[][]): void {
        this.marks.set(marks);
        const scene = this.scene;
        if (!scene) return;
        const accent = [255, 201, 60, 255];
        scene.setColored(
            'measure',
            marks.length
                ? {
                      positions: Float32Array.from(marks.flat()),
                      colors: Uint8Array.from(marks.flatMap(() => accent)),
                      channels: 4,
                  }
                : null,
        );
        scene.setColored(
            'measure-line',
            marks.length === 2
                ? {
                      positions: Float32Array.from(marks.flat()),
                      colors: Uint8Array.from([...accent, ...accent]),
                      channels: 4,
                  }
                : null,
        );
    }

    ngOnDestroy(): void {
        if (this.frame !== null) cancelAnimationFrame(this.frame);
        this.scene?.dispose();
    }
}

function formatCount(value: number): string {
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e4) return `${Math.round(value / 1e3)}k`;
    return String(Math.round(value));
}
