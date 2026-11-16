// One material primitive at a time. Material and open boundaries never cross.
import {readSync,writeSync} from 'node:fs';
import {MeshoptSimplifier} from './vendor/meshoptimizer/simplifier.mjs';
await MeshoptSimplifier.ready;
function take(n,eof=false){const b=Buffer.allocUnsafeSlow(n);let p=0;while(p<n){const k=readSync(0,b,p,n-p);if(!k){if(eof&&!p)return null;throw Error('Truncated facade request');}p+=k;}return b;}
function send(a){const b=Buffer.from(a.buffer,a.byteOffset,a.byteLength);for(let p=0;p<b.length;)p+=writeSync(1,b,p,b.length-p);}
for(;;){
 const h=take(8,true);if(!h)break;const [v,n]=new Uint32Array(h.buffer,h.byteOffset,2);
 const pb=take(v*12),ib=take(n*4);const p=new Float32Array(pb.buffer,pb.byteOffset,v*3),i=new Uint32Array(ib.buffer,ib.byteOffset,n);
 const [out]=MeshoptSimplifier.simplify(i,p,3,Math.max(3,Math.floor(n/30)*3),.05,['LockBorder','ErrorAbsolute']);
 send(new Uint32Array([out.length]));send(out);
}
