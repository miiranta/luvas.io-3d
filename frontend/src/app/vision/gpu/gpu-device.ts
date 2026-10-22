/** WebGPU flag values (from the spec; TypeScript's lib ships the types but not the constants). */
export const BufferUsage = {
    MAP_READ: 0x0001,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
} as const;
export const TextureUsage = {
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    RENDER_ATTACHMENT: 0x10,
} as const;
export const ShaderStage = { COMPUTE: 0x4 } as const;
export const MapMode = { READ: 0x0001 } as const;

export interface GpuContext {
    device: GPUDevice;
    /** Adapter description for the diagnostics panel. */
    label: string;
    /** Set when the system took the device away (driver reset, memory pressure). */
    lost: string | null;
    /** First uncaptured validation / out-of-memory error since it was last cleared. */
    error: string | null;
}

/** Requests a WebGPU device with the largest storage bindings the adapter allows. */
export async function requestGpu(): Promise<GpuContext> {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu) throw new Error('WebGPU is not available in this browser');
    // The adapter can be briefly unavailable (GPU process busy or restarting): retry a few times.
    let adapter: GPUAdapter | null = null;
    for (let attempt = 0; attempt < 4 && !adapter; attempt++) {
        if (attempt > 0) await new Promise((resolve) => setTimeout(resolve, 400 * attempt));
        adapter = await gpu.requestAdapter(
            attempt % 2 === 0 ? { powerPreference: 'high-performance' } : undefined,
        );
    }
    if (!adapter) throw new Error('no WebGPU adapter found');
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBuffersPerShaderStage: Math.min(
                10,
                adapter.limits.maxStorageBuffersPerShaderStage,
            ),
        },
    });
    const info = adapter.info;
    const label = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ');
    const context: GpuContext = { device, label: label || 'webgpu', lost: null, error: null };
    void device.lost.then((reason) => (context.lost = reason.message || reason.reason || 'lost'));
    // Without this a failed submit is silently skipped and a stage just produces nothing.
    device.addEventListener('uncapturederror', (event) => {
        context.error ??= (event as GPUUncapturedErrorEvent).error.message;
    });
    return context;
}

/** Copies a GPU buffer region into a new CPU array. */
export async function readBuffer(
    device: GPUDevice,
    source: GPUBuffer,
    bytes: number,
    offset = 0,
): Promise<ArrayBuffer> {
    const staging = device.createBuffer({
        size: bytes,
        usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, offset, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
}

export function storageBuffer(device: GPUDevice, bytes: number, label: string): GPUBuffer {
    return device.createBuffer({
        label,
        size: Math.max(16, Math.ceil(bytes / 4) * 4),
        usage: BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
    });
}

/** Turns asynchronous validation / out-of-memory errors into exceptions. */
export async function checked<T>(device: GPUDevice, work: () => T | Promise<T>): Promise<T> {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    const result = await work();
    const validation = await device.popErrorScope();
    const memory = await device.popErrorScope();
    const error = validation ?? memory;
    if (error) throw new Error(`GPU: ${error.message}`);
    return result;
}
