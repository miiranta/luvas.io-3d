import { RgbaImage } from '../texture/atlas-baker';
import { ImageStore } from './scan-session';

/** Keyframe images kept as JPEG blobs (~10× smaller than RGBA), decoded on demand. */
export class JpegImageStore implements ImageStore {
    private readonly blobs: Blob[] = [];

    async put(index: number, image: RgbaImage): Promise<void> {
        this.blobs[index] = await encode(image, 0.93);
    }

    async get(index: number): Promise<RgbaImage> {
        const bitmap = await createImageBitmap(this.blobs[index], {
            colorSpaceConversion: 'none',
            premultiplyAlpha: 'none',
        });
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext('2d', { willReadFrequently: true });
        if (!context) throw new Error('2D canvas unavailable in the worker');
        context.drawImage(bitmap, 0, 0);
        bitmap.close();
        const { data, width, height } = context.getImageData(0, 0, canvas.width, canvas.height);
        return { width, height, data };
    }

    async jpeg(image: RgbaImage, quality: number): Promise<Uint8Array> {
        return new Uint8Array(await (await encode(image, quality)).arrayBuffer());
    }
}

export async function encode(image: RgbaImage, quality: number): Promise<Blob> {
    const canvas = new OffscreenCanvas(image.width, image.height);
    const context = canvas.getContext('2d');
    if (!context) throw new Error('2D canvas unavailable in the worker');
    const pixels = new Uint8ClampedArray(
        image.data.buffer,
        image.data.byteOffset,
        image.data.byteLength,
    );
    context.putImageData(
        new ImageData(pixels as Uint8ClampedArray<ArrayBuffer>, image.width, image.height),
        0,
        0,
    );
    return canvas.convertToBlob({ type: 'image/jpeg', quality });
}
