"""Forward IPv4 127.0.0.1:3001 to the dev server, which binds only [::1]:3001 whatever --host says."""
import asyncio
async def pipe(r,w):
    try:
        while data:=await r.read(65536):w.write(data);await w.drain()
    except Exception:pass
    finally:w.close()
async def handle(cr,cw):
    try:sr,sw=await asyncio.open_connection('::1',3001)
    except Exception:cw.close();return
    await asyncio.gather(pipe(cr,sw),pipe(sr,cw))
async def main():
    server=await asyncio.start_server(handle,'0.0.0.0',3001);await server.serve_forever()
asyncio.run(main())
