import { TextureMethodId } from '../../../core/models/quality';
import { atlasMethod } from './atlas-method';

export { atlasMethod } from './atlas-method';
import { TextureMethod } from './texture-method';
import { vertexColours } from './vertex-method';

/** Texturing strategies by id (the quality profile picks one). */
export const TEXTURE_METHODS: Record<TextureMethodId, TextureMethod> = {
    vertex: vertexColours,
    atlas: atlasMethod({ seams: false, selection: { kind: 'charts', mergeSmall: false } }),
    levelled: atlasMethod({ seams: true, selection: { kind: 'charts', mergeSmall: true } }),
    mrf: atlasMethod({ seams: true, selection: { kind: 'mrf', smoothness: 1 } }),
};

export type {
    ModelResult,
    TextureInput,
    TextureMethod,
    TextureResult,
    ViewSelection,
} from './texture-method';
