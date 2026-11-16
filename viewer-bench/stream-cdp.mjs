// Small Chrome DevTools Protocol client for the isolated stream benchmark.
// Uses Node's built-in WebSocket so the memory guarded runner need not load Playwright.
import {EventEmitter} from 'node:events';
import {writeFile} from 'node:fs/promises';

const encode = value => JSON.stringify(value, (_key, item) => item === undefined ? {$streamCdpUndefined:true} : item);
const jsValue = value => value?.unserializableValue !== undefined ? Number(value.unserializableValue) : value?.value;
const argumentExpression = args => `JSON.parse(${JSON.stringify(encode(args))},(_k,v)=>v&&v.$streamCdpUndefined===true?undefined:v)`;
const expression = (fn,args=[]) => `(${fn.toString()})(...${argumentExpression(args)})`;

class Connection extends EventEmitter {
  constructor(url){super();this.ws=new WebSocket(url);this.next=0;this.pending=new Map();this.closed=false;
    this.ready=new Promise((resolve,reject)=>{this.ws.addEventListener('open',resolve,{once:true});this.ws.addEventListener('error',e=>reject(e.error??new Error('DevTools WebSocket failed')),{once:true});});
    this.ws.addEventListener('message',event=>this.message(JSON.parse(event.data)));
    this.ws.addEventListener('close',()=>{this.closed=true;const error=new Error('DevTools connection closed');for(const p of this.pending.values())p.reject(error);this.pending.clear();this.emit('close');});
  }
  message(message){
    if(!message.id){this.emit('protocol-event',message);return;}
    const item=this.pending.get(message.id);
    if(!item)return;
    this.pending.delete(message.id);
    if(message.error)item.reject(new Error(message.error.message));
    else item.resolve(message.result);
  }
  async send(method,params={},sessionId){await this.ready;if(this.closed)throw new Error('DevTools connection is closed');const id=++this.next;
    return new Promise((resolve,reject)=>{this.pending.set(id,{resolve,reject});this.ws.send(JSON.stringify({id,method,params,sessionId}));});}
  close(){if(!this.closed)this.ws.close();}
}

class CDPSession extends EventEmitter {
  constructor(connection,targetId){super();this.connection=connection;this.targetId=targetId;this.sessionId=null;this.closed=false;}
  async attach(){const result=await this.connection.send('Target.attachToTarget',{targetId:this.targetId,flatten:true});this.sessionId=result.sessionId;
    this.dispatcher=message=>{if(message.sessionId===this.sessionId)this.emit(message.method,message.params);};
    this.connection.on('protocol-event',this.dispatcher);return this;}
  send(method,params={}){if(this.closed)return Promise.reject(new Error('DevTools session is closed'));return this.connection.send(method,params,this.sessionId);}
  detach(){if(this.closed)return Promise.resolve();this.closed=true;this.connection.off('protocol-event',this.dispatcher);return this.connection.send('Target.detachFromTarget',{sessionId:this.sessionId}).catch(()=>{});}
}

class Request {
  constructor(url,method,headers,postData){this._url=url;this._method=method;this._headers=headers??{};this._postData=postData;}
  url(){return this._url;} method(){return this._method;} headers(){return this._headers;} postData(){return this._postData;}
  failure(){return this._failure?{errorText:this._failure}:null;}
}
class Route {
  constructor(session,requestId,request){this.session=session;this.requestId=requestId;this._request=request;this.done=false;}
  request(){return this._request;}
  async continue(){if(this.done)return;this.done=true;await this.session.send('Fetch.continueRequest',{requestId:this.requestId});}
  async abort(){if(this.done)return;this.done=true;await this.session.send('Fetch.failRequest',{requestId:this.requestId,errorReason:'Aborted'});}
}
class Response {constructor(url,status){this._url=url;this._status=status;}url(){return this._url;}status(){return this._status;}ok(){return this._status>=200&&this._status<300;}}

class Page extends EventEmitter {
  constructor(context,target){super();this.context=context;this.target=target;this.session=new CDPSession(context.connection,target.targetId);this.ready=this.session.attach().then(async()=>{
    await this.session.send('Page.enable');await this.session.send('Page.setLifecycleEventsEnabled',{enabled:true});await this.session.send('Runtime.enable');await this.session.send('Network.enable');
    this.session.on('Runtime.exceptionThrown',e=>this.emit('pageerror',new Error(e.exceptionDetails?.exception?.description??e.exceptionDetails?.text??'Page exception')));
    this.session.on('Runtime.bindingCalled',e=>this.binding(e));
    this.session.on('Network.requestWillBeSent',e=>{const req=new Request(e.request.url,e.request.method,e.request.headers,e.request.postData);this.requests.set(e.requestId,req);this.emit('request',req);});
    this.session.on('Network.responseReceived',e=>{const r=new Response(e.response.url,e.response.status);this.responses.set(e.requestId,r);this.emit('response',r);});
    this.session.on('Network.loadingFailed',e=>{const req=this.requests.get(e.requestId);if(req){req._failure=e.errorText;this.emit('requestfailed',req);}this.requests.delete(e.requestId);this.responses.delete(e.requestId);});
    this.session.on('Network.loadingFinished',e=>{this.requests.delete(e.requestId);this.responses.delete(e.requestId);});
    this.session.on('Runtime.consoleAPICalled',e=>{if(e.type==='error')this.emit('console',{type:()=>e.type,text:()=>e.args.map(a=>a.value??a.description??'').join(' ')});});
    this.session.on('Fetch.requestPaused',e=>this.intercept(e));
    this.session.on('Page.lifecycleEvent',e=>this.emit('lifecycle',e));
  });this.requests=new Map();this.responses=new Map();this.routes=[];this.bindings=new Map();}
  async ensure(){await this.ready;}
  async evaluate(fn,...args){await this.ensure();const out=await this.session.send('Runtime.evaluate',{expression:expression(fn,args),awaitPromise:true,returnByValue:true,userGesture:true});
    if(out.exceptionDetails)throw new Error(out.exceptionDetails.exception?.description??out.exceptionDetails.text??'Page evaluation failed');return jsValue(out.result);}
  async addInitScript(fn,arg){await this.ensure();const code=`(()=>{(${fn.toString()})(...${argumentExpression([arg])});})()`;
    await this.session.send('Page.addScriptToEvaluateOnNewDocument',{source:code});}
  async exposeFunction(name,callback){
    await this.ensure();
    const invoke=name+'__invoke';
    await this.session.send('Runtime.addBinding',{name:invoke});this.bindings.set(invoke,callback);
    const install=expression((name,invoke)=>{
      let sequence=0;
      const pending=new Map();
      window[invoke].__settle=(id,value,error)=>{
        const entry=pending.get(id);
        if(!entry)return;
        pending.delete(id);
        if(error)entry.reject(new Error(error));else entry.resolve(value);
      };
      window[name]=(...args)=>new Promise((resolve,reject)=>{
        const id=String(++sequence);
        pending.set(id,{resolve,reject});
        window[invoke](JSON.stringify({id,args}));
      });
    },[name,invoke]);
    await this.session.send('Page.addScriptToEvaluateOnNewDocument',{source:install});await this.session.send('Runtime.evaluate',{expression:install});
  }
  async binding(e){const callback=this.bindings.get(e.name);if(!callback)return;let packet;try{packet=JSON.parse(e.payload);const value=await callback(...packet.args);await this.session.send('Runtime.evaluate',{expression:`window[${JSON.stringify(e.name)}].__settle(${JSON.stringify(packet.id)},${value===undefined?'undefined':JSON.stringify(value)},null)`});}
    catch(error){if(packet?.id)await this.session.send('Runtime.evaluate',{expression:`window[${JSON.stringify(e.name)}].__settle(${JSON.stringify(packet.id)},null,${JSON.stringify(String(error))})`}).catch(()=>{});else this.emit('pageerror',error);}}
  async route(_pattern,handler){await this.ensure();this.routes.push(handler);if(!this.fetchEnabled){this.fetchEnabled=true;await this.session.send('Fetch.enable',{patterns:[{urlPattern:'*',requestStage:'Request'}]});}}
  async intercept(e){const request=new Request(e.request.url,e.request.method,e.request.headers,e.request.postData),route=new Route(this.session,e.requestId,request);
    try{for(const handler of this.routes)await handler(route);if(!route.done)await route.continue();}catch(error){this.emit('pageerror',error);await route.abort().catch(()=>{});}}
  async goto(url,{waitUntil='domcontentloaded',timeout=30000}={}){await this.ensure();const ready=this.waitLifecycle(waitUntil,timeout);await this.session.send('Page.navigate',{url});await ready;}
  waitLifecycle(name,timeout){return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.off('lifecycle',check);reject(new Error('Navigation timeout'));},timeout);const check=e=>{if(e.name.toLowerCase()===name.toLowerCase()){clearTimeout(timer);this.off('lifecycle',check);resolve();}};this.on('lifecycle',check);});}
  async waitForFunction(fn,arg,_options={}){const timeout=_options.timeout??30000,start=Date.now();while(Date.now()-start<timeout){if(await this.evaluate(fn,...(arg===undefined?[]:[arg])))return;await new Promise(r=>setTimeout(r,100));}throw new Error('waitForFunction timed out');}
  async bringToFront(){await this.ensure();await this.session.send('Page.bringToFront');}
  async screenshot({path,type='png',quality,fullPage=false}={}){await this.ensure();const result=await this.session.send('Page.captureScreenshot',{format:type==='jpeg'?'jpeg':'png',quality, captureBeyondViewport:fullPage});await writeFile(path,Buffer.from(result.data,'base64'));}
  locator(selector){return {innerText:async()=>this.evaluate(s=>document.querySelector(s)?.innerText??'',selector)};}
}

class Context {
  constructor(connection,target){this.connection=connection;this.target=target;this._page=null;}
  pages(){return this.target?[this.pageFor(this.target)]:[];}
  pageFor(target){if(!this._page||this._page.target.targetId!==target.targetId)this._page=new Page(this,target);return this._page;}
  async newCDPSession(page){await page.ensure();return page.session;}
}

export const browser={async connect(debugUrl){const connection=new Connection(debugUrl);await connection.ready;
  const version=await connection.send('Browser.getVersion');const targetInfo=await connection.send('Target.getTargets');
  const target=targetInfo.targetInfos.find(t=>t.type==='page'&&t.url==='about:blank');
  if(!target){connection.close();throw Error('The isolated blank tab was not found');}
  const context=new Context(connection,target);
  return {version:()=>version.product,contexts:()=>[context],close:async()=>{await context._page?.session.detach();connection.close();}};
}};
export default browser;
