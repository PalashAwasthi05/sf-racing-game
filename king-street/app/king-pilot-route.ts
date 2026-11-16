// URL policy stays separate from stream residency and immutable scene packages.
export function pilotStreamRequest(url:URL):'neighborhood'|'corridor'|'match-source'|null {
 if(url.pathname.replace(/\/$/,'')==='/previous'||url.searchParams.get('stream')==='0')return null;
 const requested=url.searchParams.get('manifest');
 if(requested&&/^\/compare\/[\w.-]+\.json$/.test(requested))return 'match-source';
 if(url.searchParams.get('stream')==='corridor')return 'corridor';
 return url.pathname==='/'?'neighborhood':null;
}
export function pilotAreaHref(value:string) {
 if(value==='@neighborhood')return '/';
 if(value==='@corridor')return '/?stream=corridor&drive=1';
 return value?'/?manifest='+encodeURIComponent(value):'/previous';
}
export function pilotStreamMatches(request:ReturnType<typeof pilotStreamRequest>,archiveHash?:string,sourceHash?:string) {
 return !!request&&(request!=='match-source'||!!archiveHash&&archiveHash===sourceHash);
}
