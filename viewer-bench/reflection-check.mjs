// Controlled GPU reflection fixture; no production geometry is modified.
// Run with two local dev servers, frozen baseline then revised:
//   node reflection-check.mjs BASELINE_PORT_OR_URL CANDIDATE_PORT_OR_URL OUTPUT_DIR
// Add --occlusion for a compact two-pose regression with a dim wall behind the source.
// Requires Playwright resolvable from cwd/package.json or PLAYWRIGHT_FROM.
// Optional CHROME_EXECUTABLE selects system Chrome; GPU_ARGS is a JSON array.
// Imports the servers' actual Three, TemporalResolve, reflectStreet and DepthProbeCapture.
// Outputs full images, analytical-reference crops, GPU/errors and numerical measurements.
// This isolates material/probe registration, not the viewer's capture scheduling/culling,
// road shaders, glass, foliage, dynamic sources or performance. The reference is an
// ideal planar mirror; PMREM/BRDF blur and energy differences are expected.
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
const { chromium } = createRequire(
  path.join(process.env.PLAYWRIGHT_FROM || process.cwd(), "package.json"),
)("playwright");
const occlusion = process.argv.includes("--occlusion");
const [base = "3017", revised = "3001", output = "reflection-check-output"] = process.argv
    .slice(2)
    .filter((v) => !v.startsWith("--")),
  out = path.resolve(output);
mkdirSync(out, { recursive: true });
const origin = (value) =>
  value.startsWith("http") ? value.replace(/\/$/, "") : `http://127.0.0.1:${value}`;
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : {}),
  args: process.env.GPU_ARGS
    ? JSON.parse(process.env.GPU_ARGS)
    : [
        ...(process.platform === "darwin" ? ["--use-angle=metal"] : []),
        "--enable-gpu",
        "--ignore-gpu-blocklist",
      ],
});
const reports = {};
try {
  for (const [tag, port] of [
    ["baseline", base],
    ["revised", revised],
    ["same-probe-no-correction", revised],
  ]) {
    const page = await browser.newPage({
        viewport: { width: 960, height: 720 },
        deviceScaleFactor: 1,
      }),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });
    await page.route("**/__reflection-check", (r) =>
      r.fulfill({ contentType: "text/html", body: '<html><body style="margin:0"></body></html>' }),
    );
    await page.goto(`${origin(port)}/__reflection-check`);
    const result = await page.evaluate(
      async ({ tag, occlusion }) => {
        const text = await fetch("/app/temporal-resolve.ts").then((r) => r.text());
        const threePath = text.match(/from\s*["']([^"']*three[^"']*)["']/)?.[1];
        if (!threePath) throw Error("Cannot locate actual three module import");
        const T = await import(threePath),
          { TemporalResolve } = await import("/app/temporal-resolve.ts"),
          { reflectStreet } = await import("/app/street-reflections.ts");
        const W = 960,
          H = 720,
          renderer = new T.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
        renderer.setSize(W, H);
        renderer.toneMapping = T.NoToneMapping;
        renderer.outputColorSpace = T.LinearSRGBColorSpace;
        document.body.append(renderer.domElement);
        const scene = new T.Scene();
        scene.background = new T.Color(0.008, 0.008, 0.008);
        const camera = new T.PerspectiveCamera(44, W / H, 0.1, 100);
        camera.position.set(0, 2, 8);
        const target = new T.WebGLRenderTarget(W, H, {
            type: T.HalfFloatType,
            depthTexture: new T.DepthTexture(W, H, T.UnsignedIntType),
          }),
          temporal = new TemporalResolve(target, camera);
        temporal.setSize(W, H);
        // Two broad bands and asymmetric end caps make image identity and registration measurable.
        const group = new T.Group();
        for (let i = 0; i < 8; i++) {
          const color = [
            0xf51b08, 0x0ae72b, 0x1468ff, 0xffe609, 0xf613ab, 0x16e8ed, 0xfe6722, 0x7e37ff,
          ][i];
          const m = new T.MeshStandardMaterial({
            color: 0x000000,
            emissive: color,
            emissiveIntensity: 1,
            roughness: 1,
            side: T.DoubleSide,
          });
          const mesh = new T.Mesh(new T.PlaneGeometry(0.49, 1.8), m);
          mesh.position.set((i - 3.5) * 0.5, 3, -4);
          group.add(mesh);
        }
        scene.add(group);
        const wall = occlusion
          ? new T.Mesh(
              new T.PlaneGeometry(40, 16),
              new T.MeshStandardMaterial({
                color: 0x000000,
                emissive: new T.Color(0.035, 0.035, 0.035),
                roughness: 1,
                side: T.DoubleSide,
              }),
            )
          : null;
        if (wall) {
          wall.position.set(0, 8, -12);
          scene.add(wall);
        }
        const floorMat = new T.MeshStandardMaterial({
          color: 0xffffff,
          metalness: 1,
          roughness: 0.035,
          envMapIntensity: 1,
        });
        const floor = new T.Mesh(new T.PlaneGeometry(50, 50), floorMat);
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);
        const probeState = {
          origin: { value: new T.Vector3(0, 2.5, 8) },
          enabled: { value: tag === "same-probe-no-correction" ? 0 : 1 },
          capture: { value: 0 },
        };
        reflectStreet([floorMat], temporal, tag === "baseline" ? undefined : probeState);
        const depthProbe =
          tag === "baseline"
            ? null
            : new (await import("/app/street-probe.ts")).DepthProbeCapture(512);
        const cube =
            depthProbe?.cube ??
            new T.WebGLCubeRenderTarget(128, {
              type: T.HalfFloatType,
              generateMipmaps: false,
              minFilter: T.LinearFilter,
            }),
          probe = new T.CubeCamera(0.35, 2000, cube),
          filter = new T.PMREMGenerator(renderer);
        probe.position.copy(probeState.origin.value);
        floor.visible = false;
        if (depthProbe) {
          if (probe.coordinateSystem !== renderer.coordinateSystem) {
            probe.coordinateSystem = renderer.coordinateSystem;
            probe.updateCoordinateSystem();
          }
          probe.updateMatrixWorld();
          probeState.capture.value = 1;
          for (let f = 0; f < 6; f++) depthProbe.renderFace(renderer, scene, probe.children[f], f);
          probeState.capture.value = 0;
        } else probe.update(renderer, scene);
        floor.visible = true;
        const pmrem = filter.fromCubemap(cube.texture);
        floorMat.envMap = pmrem.texture;
        floorMat.needsUpdate = true;
        const diagnostics = [];
        for (let i = 0; i < 8; i++) {
          const x = (i - 3.5) * 0.5,
            dir = new T.Vector3(x, 0.5, -12),
            size = pmrem.height / 4,
            px = Math.floor(2 * size + 1 + (0.5 + x / 24) * (size - 2)),
            py = Math.floor(size + 1 + (0.5 + 0.5 / 24) * (size - 2)),
            a = new Uint16Array(4);
          renderer.readRenderTargetPixels(pmrem, px, py, 1, 1, a);
          diagnostics.push({
            stripe: i,
            direction: dir.toArray(),
            expectedRange: dir.length(),
            atlasPixel: [px, py],
            rgba: Array.from(a, T.DataUtils.fromHalfFloat),
          });
        }
        const originalCompile = floorMat.onBeforeCompile;
        let shaderPatch = null;
        floorMat.onBeforeCompile = (s, r) => {
          originalCompile(s, r);
          shaderPatch = {
            hasCorrection: s.fragmentShader.includes("reflectVec=correctStreetDirection"),
            hasUniform: !!s.uniforms.uStreetProbeEnabled,
            enabled: s.uniforms.uStreetProbeEnabled?.value,
          };
        };
        const gl = renderer.getContext(),
          gpu = gl.getParameter(
            gl.getExtension("WEBGL_debug_renderer_info").UNMASKED_RENDERER_WEBGL,
          );
        const read = () => {
          const a = new Uint8Array(W * H * 4);
          gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, a);
          return a;
        };
        const render = () => {
          temporal.begin();
          renderer.setRenderTarget(target);
          renderer.render(scene, camera);
          temporal.restore();
          temporal.render(renderer);
          renderer.setRenderTarget(null);
          const old = scene.background; // Copy the actual linear beauty render, without a second scene draw.
          copy.uniforms.t.value = target.texture;
          renderer.render(copyScene, copyCamera);
          scene.background = old;
        };
        const copyScene = new T.Scene(),
          copyCamera = new T.Camera(),
          copy = new T.ShaderMaterial({
            uniforms: { t: { value: target.texture } },
            vertexShader: "varying vec2 v;void main(){v=uv;gl_Position=vec4(position.xy,0.,1.);}",
            fragmentShader:
              "uniform sampler2D t;varying vec2 v;void main(){gl_FragColor=vec4(texture2D(t,v).rgb,1.);}",
            depthTest: false,
            depthWrite: false,
          });
        copyScene.add(new T.Mesh(new T.PlaneGeometry(2, 2), copy));
        const snapshots = [],
          metrics = [];
        for (const offset of [0, 3]) {
          temporal.reset();
          camera.position.x = offset;
          for (const s of occlusion ? [0, 30] : Array.from({ length: 31 }, (_, i) => i)) {
            const pitch = ((-7 - s * 0.8) * Math.PI) / 180;
            camera.lookAt(0, 2 + Math.tan(pitch) * 12, -4);
            camera.updateMatrixWorld();
            for (let k = 0; k < (occlusion ? 24 : s === 0 ? 20 : 3); k++) render();
            if (![0, 10, 20, 30].includes(s)) continue;
            const actual = read(),
              png = renderer.domElement.toDataURL("image/png").split(",")[1];
            // Ideal planar reflection: mirror the source in world space, hide the floor and real source.
            floor.visible = false;
            group.visible = false;
            const ideal = group.clone();
            ideal.visible = true;
            ideal.scale.y = -1;
            scene.add(ideal);
            const idealWall = wall?.clone();
            if (wall) {
              wall.visible = false;
              idealWall.position.y *= -1;
              idealWall.scale.y = -1;
              scene.add(idealWall);
            }
            renderer.setRenderTarget(null);
            renderer.render(scene, camera);
            const reference = read(),
              refPng = renderer.domElement.toDataURL("image/png").split(",")[1];
            scene.remove(ideal);
            if (wall) {
              scene.remove(idealWall);
              wall.visible = true;
            }
            floor.visible = true;
            group.visible = true;
            const corners = [
                [-2, -3.9, -4],
                [2, -2.1, -4],
              ].map((p) => new T.Vector3(...p).project(camera)),
              directCorners = [
                [-2, 2.1, -4],
                [2, 3.9, -4],
              ].map((p) => new T.Vector3(...p).project(camera));
            let x0 = Math.max(0, Math.floor((corners[0].x * 0.5 + 0.5) * W)),
              x1 = Math.min(W, Math.ceil((corners[1].x * 0.5 + 0.5) * W)),
              y0 = Math.max(0, Math.floor((corners[0].y * 0.5 + 0.5) * H)),
              y1 = Math.min(H, Math.ceil((corners[1].y * 0.5 + 0.5) * H));
            let n = 0,
              dot = 0,
              aa = 0,
              bb = 0,
              err = 0,
              sumA = 0,
              sumB = 0,
              weight = 0,
              cx = 0,
              cy = 0;
            for (let y = y0; y < y1; y++)
              for (let x = x0; x < x1; x++) {
                const i = (y * W + x) * 4;
                for (let c = 0; c < 3; c++) {
                  const a = actual[i + c] / 255,
                    b = reference[i + c] / 255;
                  dot += a * b;
                  aa += a * a;
                  bb += b * b;
                  err += Math.abs(a - b);
                  sumA += a;
                  sumB += b;
                  n++;
                }
                const w =
                  Math.max(actual[i], actual[i + 1], actual[i + 2]) -
                  Math.min(actual[i], actual[i + 1], actual[i + 2]);
                weight += w;
                cx += w * x;
                cy += w * y;
              }
            metrics.push({
              offset,
              step: s,
              pitchDeg: -7 - s * 0.8,
              sourceEntirelyOffscreen: directCorners.every((p) => p.y > 1),
              sourceNdcY: directCorners.map((p) => p.y),
              reflectionCrop: { x: x0, y: H - y1, width: x1 - x0, height: y1 - y0 },
              cosine: dot / Math.sqrt(aa * bb || 1),
              relativeEnergy: sumA / sumB,
              pearson:
                (dot - (sumA * sumB) / n) /
                Math.sqrt((aa - (sumA * sumA) / n) * (bb - (sumB * sumB) / n) || 1),
              mae: err / n,
              chromaCentroid: weight ? [cx / weight, H - cy / weight] : null,
            });
            const crop = async (data) => {
              const img = await createImageBitmap(
                  await (await fetch("data:image/png;base64," + data)).blob(),
                ),
                c = document.createElement("canvas");
              c.width = x1 - x0;
              c.height = y1 - y0;
              c.getContext("2d").drawImage(
                img,
                x0,
                H - y1,
                c.width,
                c.height,
                0,
                0,
                c.width,
                c.height,
              );
              return c.toDataURL("image/png").split(",")[1];
            };
            snapshots.push({
              step: `x${offset}-${s}`,
              png,
              refPng,
              crop: await crop(png),
              refCrop: await crop(refPng),
            });
          }
        }
        renderer.dispose();
        return {
          fixture: occlusion ? "source-with-far-wall" : "isolated-source",
          wall: occlusion
            ? { planeZ: -12, width: 40, height: 16, centerY: 8, linearRadiance: 0.035 }
            : null,
          gpu,
          diagnostics,
          shaderPatch,
          metrics,
          snapshots,
        };
      },
      { tag, occlusion },
    );
    for (const s of result.snapshots) {
      writeFileSync(`${out}/${tag}-${s.step}.png`, Buffer.from(s.png, "base64"));
      writeFileSync(`${out}/${tag}-${s.step}-ideal.png`, Buffer.from(s.refPng, "base64"));
      writeFileSync(`${out}/${tag}-${s.step}-crop.png`, Buffer.from(s.crop, "base64"));
      writeFileSync(`${out}/${tag}-${s.step}-ideal-crop.png`, Buffer.from(s.refCrop, "base64"));
    }
    delete result.snapshots;
    reports[tag] = { ...result, errors };
    await page.close();
  }
  // A source must retain its registered color structure after leaving the image.
  // Pearson tolerates expected PMREM/BRDF brightness loss and blur, while .80
  // rejects missing/incorrect stripes. The matched 512px ablation guards against
  // attributing an existing probe response to the direction correction itself.
  const failures = [];
  for (const [tag, report] of Object.entries(reports))
    if (report.errors.length) failures.push(`${tag}: ${report.errors.join("; ")}`);
  for (const offset of [0, 3]) {
    const actual = reports.revised.metrics.find((m) => m.offset === offset && m.step === 30),
      ablation = reports["same-probe-no-correction"].metrics.find(
        (m) => m.offset === offset && m.step === 30,
      );
    if (!actual?.sourceEntirelyOffscreen)
      failures.push(`x=${offset}: final source is not wholly offscreen`);
    if (!Number.isFinite(actual?.pearson) || actual.pearson < 0.8)
      failures.push(`x=${offset}: final pattern correlation ${actual?.pearson} is below .80`);
    if (actual?.pearson < ablation?.pearson)
      failures.push(
        `x=${offset}: final pattern correlation is below matched no-correction ablation`,
      );
  }
  reports.validation = {
    passed: failures.length === 0,
    failures,
    minimumOffscreenPearson: 0.8,
    brightnessThreshold: null,
    rationale:
      "Registered palette structure must survive source departure. Pearson tolerates expected PMREM/BRDF energy loss; absolute brightness is diagnostic only. Matched no-correction ablation isolates direction correction.",
  };
  writeFileSync(out + "/metrics.json", JSON.stringify(reports, null, 2));
  console.log(JSON.stringify(reports, null, 2));
  if (failures.length) throw Error("Reflection regression failed: " + failures.join("; "));
} finally {
  await browser.close();
}
