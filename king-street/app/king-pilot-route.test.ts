import test from 'node:test';
import assert from 'node:assert/strict';
import {pilotStreamRequest,pilotAreaHref,pilotStreamMatches,pilotStreamIndex} from './king-pilot-route.ts';
const route=(path:string)=>pilotStreamRequest(new URL(path,'https://viewer.example'));
test('normal launch and legacy streaming URLs open the full neighborhood',()=>{
 for(const url of ['/','/?drive=1','/?stream=1','/?stream=1&drive=1','/?stream=neighborhood','/?viewFrom=1,2,3#viewFrom=4,5,6'])assert.equal(route(url),'neighborhood',url);
 assert.equal(route('/?stream=corridor&drive=1'),'corridor');
});
test('archival routes survive and explicit manifests require a source hash match',()=>{
 assert.equal(route('/previous'),null);
 assert.equal(route('/previous?drive=1&stream=1'),null);
 assert.equal(route('/?stream=0'),null);
 for(const name of ['current','pacific-heights-r31-street','king-street-local'])
  assert.equal(route('/?manifest=/compare/'+name+'.json'),'match-source');
 assert.equal(route('/block-drive'),null);
 assert.equal(route('/car-studio'),null);
 assert.equal(pilotStreamMatches('match-source','old-source','new-source'),false);
 assert.equal(pilotStreamMatches('match-source','same-source','same-source'),true);
 assert.equal(pilotStreamMatches('match-source',undefined,undefined),false);
 assert.equal(pilotStreamMatches(null,'same-source','same-source'),false);
});
test('area menu offers independent full, corridor and archive destinations',()=>{
 assert.equal(pilotAreaHref('@neighborhood'),'/');
 assert.equal(pilotAreaHref('@corridor'),'/?stream=corridor&drive=1');
 assert.equal(pilotAreaHref(''),'/previous');
 assert.equal(new URL(pilotAreaHref('/compare/king-street-local.json'),'https://viewer.example').searchParams.get('manifest'),'/compare/king-street-local.json');
});

test('named archives select their own spatial package and retain source matching',()=>{
 const index='/compare/area-stream-1234.json';
 assert.equal(pilotStreamIndex('match-source',index),index);
 assert.equal(pilotStreamIndex('match-source','https://invalid.example/x.json'),'/compare/neighborhood-stream.json');
 assert.equal(pilotStreamIndex('match-source','/compare/../x.json'),'/compare/neighborhood-stream.json');
 assert.equal(pilotStreamIndex('neighborhood',index),'/compare/neighborhood-stream.json');
 assert.equal(pilotStreamIndex('corridor',index),'/compare/drive-stream.json');
});
