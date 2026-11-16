import type {SceneManifest} from '../scene-package';
import type {BlockTrial} from '../block-trial';

type RoutePackage = SceneManifest & {world: string};
const localJson = (value: unknown): value is string => typeof value === 'string' &&
  /^\/(?:[\w.-]+\/)*[\w.-]+\.json$/.test(value) && !value.split('/').some(part => part === '.' || part === '..');

/** A compact comparison must belong to this exact frozen world and drive. */
export function routePackage(value: unknown, archive: SceneManifest, trial: Pick<BlockTrial, 'routeEN'>): RoutePackage | null {
  if (!value || typeof value !== 'object') return null;
  const candidate = value as Record<string, unknown>;
  const hash = candidate.contentHash;
  const layerAvailable = (key: 'cars' | 'overhead') =>
    candidate[key] === undefined ? archive[key] === undefined : localJson(candidate[key]);
  if (candidate.scope !== 'route-block' || candidate.sourceWorldHash !== archive.contentHash ||
      typeof archive.contentHash !== 'string' || !/^[0-9a-f]{12}$/.test(archive.contentHash) ||
      typeof hash !== 'string' || !/^[0-9a-f]{12}$/.test(hash) || candidate.format !== 'gzip-chunks' ||
      typeof candidate.compressedBytes !== 'number' || !Number.isSafeInteger(candidate.compressedBytes) || candidate.compressedBytes <= 0 ||
      typeof candidate.uncompressedBytes !== 'number' || !Number.isSafeInteger(candidate.uncompressedBytes) || candidate.uncompressedBytes < 28 ||
      !Array.isArray(candidate.chunks) || candidate.chunks.length !== Math.ceil(candidate.uncompressedBytes / (8 * 1024 * 1024)) ||
      !candidate.chunks.every((url, index) => url === '/world-' + hash + '-' + String(index).padStart(2, '0') + '.bin') ||
      !localJson(candidate.world) || !layerAvailable('cars') || !layerAvailable('overhead') ||
      !Array.isArray(candidate.routeEN) || candidate.routeEN.length !== trial.routeEN.length ||
      !candidate.routeEN.every((point, index) => Array.isArray(point) && point.length === 2 &&
        trial.routeEN[index].length === 2 && point.every((coordinate, axis) => typeof coordinate === 'number' &&
          Number.isFinite(coordinate) && coordinate === trial.routeEN[index][axis]))) return null;
  return value as RoutePackage;
}
