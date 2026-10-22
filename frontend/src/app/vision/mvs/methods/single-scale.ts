import { DenseStereo } from '../dense-stereo';
import { StereoMethod, uploadImages } from './stereo-method';

/** PatchMatch at one resolution, started from random planes and the phone depth. */
export const singleScale: StereoMethod = {
    async create(device, input, settings) {
        const { width, height } = input;
        const stereo = new DenseStereo(device, width, height, input.views);
        await uploadImages(stereo, input).catch((error: unknown) => {
            stereo.destroy();
            throw error;
        });
        return {
            stereo,
            solve: (layer) =>
                stereo.solve(layer, {
                    sources: input.sources[layer],
                    minDepth: input.ranges[layer][0],
                    maxDepth: input.ranges[layer][1],
                    prior: input.prior(layer, width, height),
                    geometric: false,
                    iterations: settings.iterations,
                }),
            destroy: () => stereo.destroy(),
        };
    },
};
