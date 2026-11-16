import type {BlockVariant} from './block-trial';

type Overlay = Exclude<BlockVariant, 'baseline'>;
type Layer = {show(visible: boolean): void; dispose(): void};

/** Only the last requested representation may become visible. Each overlay is
 * fetched once on demand; failed loads are removed so the same button retries. */
export function lazyBlockVariants<T extends Layer>(load: (variant: Overlay, signal: AbortSignal) => Promise<T>) {
  const layers = new Map<Overlay, T>(), pending = new Map<Overlay, Promise<T>>();
  const abort = new AbortController();let generation=0, disposed=false;
  return {
    async set(variant: BlockVariant): Promise<boolean> {
      const request=++generation;
      for (const layer of layers.values()) layer.show(false);
      if(disposed)return false;
      if(variant==='baseline')return true;
      let waiting=pending.get(variant);
      if(!layers.has(variant)&&!waiting){
        waiting=load(variant,abort.signal).then(layer=>{
          if(disposed)layer.dispose();else{layer.show(false);layers.set(variant,layer);}
          return layer;
        }).finally(()=>pending.delete(variant));
        pending.set(variant,waiting);
      }
      try {
        const layer=layers.get(variant)??await waiting!;
        if(disposed||request!==generation)return false;
        layer.show(true);return true;
      } catch(error) {
        if(disposed||request!==generation)return false;
        throw error;
      }
    },
    forEach(callback: (layer: T) => void) {for(const layer of layers.values())callback(layer);},
    dispose() {if(disposed)return;disposed=true;generation++;abort.abort();for(const layer of layers.values())layer.dispose();layers.clear();},
  };
}
