import { StereoMethodId } from '../../../core/models/quality';
import { coarseToFine } from './coarse-to-fine';
import { singleScale } from './single-scale';
import { StereoMethod } from './stereo-method';

/** Stereo strategies by id (the quality profile picks one). */
export const STEREO_METHODS: Record<StereoMethodId, StereoMethod> = {
    'single-scale': singleScale,
    'coarse-to-fine': coarseToFine,
};

export type { StereoInput, StereoMethod, StereoRun, StereoSettings } from './stereo-method';
