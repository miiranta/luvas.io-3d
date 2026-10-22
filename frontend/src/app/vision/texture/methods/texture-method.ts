import { Mesh } from '../../fusion/mesh';
import { RgbaImage, TextureView } from '../texture-types';
import { FaceCandidates, faceCandidates } from '../face-visibility';
import { scaleIntrinsics } from '../../geometry/camera';

/** Everything a texturing method may use. */
export interface TextureInput {
    device: GPUDevice;
    mesh: Mesh;
    /** Per view: intrinsics at the texture resolution, pose, thumbnail and a quality weight. */
    views: (TextureView & { quality: number })[];
    /** Full-resolution image of a view. */
    image(view: number): Promise<RgbaImage>;
    jpeg(image: RgbaImage, quality: number): Promise<Uint8Array>;
    /** Resolution of the visibility z-buffer relative to the texture images. */
    visibility: number;
    /** Far plane of the z-buffer (m). */
    far: number;
}

export interface ViewSelection {
    candidates: FaceCandidates;
    /** Face adjacency and one view per face, for methods that build charts. */
    adjacency: Int32Array | null;
    labels: Int16Array | null;
}

/** The final model: a textured mesh (uvs + JPEG atlas) or a vertex-coloured one. */
export interface ModelResult {
    positions: Float32Array;
    indices: Uint32Array;
    uvs: Float32Array | null;
    atlasJpeg: Uint8Array | null;
    /** RGBA8 per vertex when there is no atlas. */
    colors: Uint8Array | null;
}

export interface TextureResult {
    model: ModelResult;
    charts: number;
    untextured: number;
    atlas: [number, number] | null;
}

export type Progress = (fraction: number, detail?: string | null) => void;

/** A texturing strategy; registered in `methods/index.ts`. */
export interface TextureMethod {
    selectViews(input: TextureInput, progress: Progress): Promise<ViewSelection>;
    bake(input: TextureInput, selection: ViewSelection, progress: Progress): Promise<TextureResult>;
}

/** Visible, front-facing candidate views per face from z-buffers at `input.visibility` scale. */
export function visibleCandidates(
    input: TextureInput,
    progress: Progress,
): Promise<FaceCandidates> {
    const first = input.views[0];
    const width = Math.round(first.width * input.visibility);
    const height = Math.round(first.height * input.visibility);
    return faceCandidates(
        input.device,
        input.mesh.positions,
        input.mesh.indices,
        input.views.map((view) => ({
            ...view,
            intrinsics: scaleIntrinsics(view.intrinsics, width / view.width, height / view.height),
        })),
        width,
        height,
        0.02,
        input.far,
        progress,
    );
}
