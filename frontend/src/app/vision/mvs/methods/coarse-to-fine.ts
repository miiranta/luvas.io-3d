import { scaleIntrinsics } from '../../geometry/camera';
import { DenseStereo } from '../dense-stereo';
import { StereoMethod, uploadImages } from './stereo-method';

/**
 * PatchMatch at half resolution first (a window covers twice the scene, which helps weakly
 * textured areas), then at full resolution starting from the coarse planes (ACMMP-style).
 */
export const coarseToFine: StereoMethod = {
    async create(device, input, settings) {
        const { width, height } = input;
        const coarseWidth = Math.round(width / 2);
        const coarseHeight = Math.round(height / 2);
        const coarse = new DenseStereo(
            device,
            coarseWidth,
            coarseHeight,
            input.views.map((view) => ({
                pose: view.pose,
                intrinsics: scaleIntrinsics(
                    view.intrinsics,
                    coarseWidth / width,
                    coarseHeight / height,
                ),
            })),
            { depthMaps: false },
        );
        const stereo = new DenseStereo(device, width, height, input.views, {
            coarse: { hypotheses: coarse.hypotheses, width: coarseWidth, height: coarseHeight },
        });
        try {
            await uploadImages(coarse, input);
            await uploadImages(stereo, input);
        } catch (error) {
            stereo.destroy();
            coarse.destroy();
            throw error;
        }
        return {
            stereo,
            async solve(layer) {
                const range = {
                    sources: input.sources[layer],
                    minDepth: input.ranges[layer][0],
                    maxDepth: input.ranges[layer][1],
                    geometric: false,
                };
                await coarse.solve(layer, {
                    ...range,
                    prior: input.prior(layer, coarseWidth, coarseHeight),
                    iterations: settings.iterations,
                    pack: false,
                });
                await stereo.solve(layer, {
                    ...range,
                    prior: null,
                    iterations: settings.fineIterations,
                    fromCoarse: true,
                });
            },
            // The fine level's bind group references the coarse planes: release them together.
            destroy() {
                stereo.destroy();
                coarse.destroy();
            },
        };
    },
};
