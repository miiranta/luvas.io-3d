import { Intrinsics, Pose } from '../geometry/camera';

export interface RgbaImage {
    width: number;
    height: number;
    data: Uint8Array | Uint8ClampedArray;
}

export interface TextureView {
    /** Intrinsics at the resolution of the full texture image. */
    intrinsics: Intrinsics;
    pose: Pose;
    width: number;
    height: number;
    /** Small RGB copy used for colour statistics. */
    thumb: { rgb: Uint8Array; width: number; height: number; intrinsics: Intrinsics };
}

export interface TexturedMesh {
    positions: Float32Array;
    uvs: Float32Array;
    indices: Uint32Array;
    atlas: { width: number; height: number; pixels: Uint8ClampedArray };
    charts: number;
    /** Faces no camera could texture (painted neutral grey). */
    untextured: number;
}
