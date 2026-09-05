// A small point-cloud viewer on three.js for the depth outputs: a depth map plus the colour frame it came
// from become a cloud you can orbit. Loaded on demand as an ES module; three.js comes from jsdelivr, the same
// URL the robot page maps, so the two share one copy.
const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.181.0/build/three.module.js";
let THREE = null;

// depth (H×W, camera-frame distance along the view axis) + colours (RGBA bytes of a W'×H' frame) -> positions, colours
export function cloudFromDepth(depth, w, h, rgba, rw, rh, fovyDeg, stride, maxDepth) {
  stride = stride || 1;
  maxDepth = maxDepth || Infinity;   // points at or beyond it (a far plane, the sky) are dropped
  const f = (h / 2) / Math.tan((fovyDeg || 60) * Math.PI / 360);
  const n = Math.ceil(w / stride) * Math.ceil(h / stride);
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let k = 0;
  for (let v = 0; v < h; v += stride) {
    for (let u = 0; u < w; u += stride) {
      const z = depth[v * w + u];
      if (!(z > 0) || z >= maxDepth) continue;
      pos[k * 3] = (u + 0.5 - w / 2) / f * z;
      pos[k * 3 + 1] = -(v + 0.5 - h / 2) / f * z;
      pos[k * 3 + 2] = -z;
      if (rgba) {
        const ru = Math.min(rw - 1, Math.floor(u * rw / w)), rv = Math.min(rh - 1, Math.floor(v * rh / h)), i = (rv * rw + ru) * 4;
        col[k * 3] = rgba[i] / 255; col[k * 3 + 1] = rgba[i + 1] / 255; col[k * 3 + 2] = rgba[i + 2] / 255;
      } else { col[k * 3] = col[k * 3 + 1] = col[k * 3 + 2] = 0.8; }
      k++;
    }
  }
  return { positions: pos.subarray(0, k * 3), colors: col.subarray(0, k * 3) };
}

// relative inverse depth (Depth Anything: larger is closer) -> a plausible metric-looking depth in [near, far]
export function inverseToDepth(rel, near, far) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < rel.length; i++) { if (rel[i] < mn) mn = rel[i]; if (rel[i] > mx) mx = rel[i]; }
  const out = new Float32Array(rel.length), s = mx > mn ? 1 / (mx - mn) : 1;
  for (let i = 0; i < rel.length; i++) { const t = (rel[i] - mn) * s; out[i] = 1 / (1 / far + t * (1 / near - 1 / far)); }
  return out;
}

export async function createViewer(canvas) {
  THREE = THREE || await import(THREE_URL);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true });
  const points = new THREE.Points(geometry, material);
  scene.add(points);
  // a plain orbit: drag to turn, wheel to zoom, around the cloud's centre
  const target = new THREE.Vector3(0, 0, -2);
  let yaw = 0, pitch = 0, dist = 3, dragging = false, lastX = 0, lastY = 0, dirty = true, alive = true;
  function place() {
    camera.position.set(target.x + dist * Math.sin(yaw) * Math.cos(pitch), target.y + dist * Math.sin(pitch), target.z + dist * Math.cos(yaw) * Math.cos(pitch));
    camera.lookAt(target);
  }
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => { if (!dragging) return; yaw -= (e.clientX - lastX) * 0.008; pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - lastY) * 0.008)); lastX = e.clientX; lastY = e.clientY; dirty = true; });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); dist = Math.max(0.2, Math.min(50, dist * (e.deltaY > 0 ? 1.1 : 0.9))); dirty = true; }, { passive: false });
  function resize() {
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
    }
  }
  function frame() {
    if (!alive) return;
    resize();
    if (dirty) { place(); renderer.render(scene, camera); dirty = false; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return {
    // positions and colours as flat arrays; the camera frames the cloud's centre at a distance that fits it
    setCloud(positions, colors, opts) {
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geometry.computeBoundingSphere();
      // the caller may frame a region of interest instead of the whole cloud (a far wall would otherwise shrink a near table)
      const bs = opts && opts.frame ? { center: new THREE.Vector3().fromArray(opts.frame.center), radius: opts.frame.radius } : geometry.boundingSphere;
      if (bs && isFinite(bs.radius) && (!opts || !opts.keepView)) {
        target.copy(bs.center); dist = bs.radius * 1.7; material.size = bs.radius * ((opts && opts.pointScale) || 0.012);
        if (opts && opts.view) { yaw = opts.view.yaw || 0; pitch = opts.view.pitch || 0; }   // start from an angle that shows the relief
      }
      else if (bs && isFinite(bs.radius)) { material.size = bs.radius * ((opts && opts.pointScale) || 0.012); }
      dirty = true;
    },
    dispose() { alive = false; geometry.dispose(); material.dispose(); renderer.dispose(); },
  };
}

// ---- volumes: a ray-marched view of a (D, H, W) scalar field inside a wireframe cube -------------------------
export async function createVolumeViewer(canvas) {
  THREE = THREE || await import(THREE_URL);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 20);
  const uniforms = {
    map: { value: null },
    cameraPos: { value: new THREE.Vector3() },
    steps: { value: 120.0 },
    density: { value: 1.6 },
    lo: { value: 0.0 },
    hi: { value: 1.0 },
  };
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms,
    transparent: true,
    side: THREE.BackSide,
    depthWrite: false,
    vertexShader: `
      out vec3 vLocal;
      void main() { vLocal = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `
      precision highp float; precision highp sampler3D;
      uniform sampler3D map; uniform vec3 cameraPos; uniform float steps, density, lo, hi;
      in vec3 vLocal; out vec4 outColor;
      // a compact turbo-like ramp: dark blue -> cyan -> yellow -> red
      vec3 ramp(float t) {
        return clamp(vec3(1.7 * t - 0.6, 1.0 - abs(2.2 * t - 1.1), 1.2 - 2.0 * t) + vec3(0.0, 0.15, 0.3) * (1.0 - t), 0.0, 1.0);
      }
      vec2 box(vec3 o, vec3 d) {   // entry/exit distances of the unit cube centred at the origin
        vec3 inv = 1.0 / d, t0 = (-0.5 - o) * inv, t1 = (0.5 - o) * inv;
        vec3 tmin = min(t0, t1), tmax = max(t0, t1);
        return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
      }
      void main() {
        vec3 dir = normalize(vLocal - cameraPos);
        vec2 t = box(cameraPos, dir);
        float tn = max(t.x, 0.0), tf = t.y;
        if (tf <= tn) { outColor = vec4(0.0); return; }
        float dt = (tf - tn) / steps;
        vec3 p = cameraPos + dir * tn;
        vec4 acc = vec4(0.0);
        for (float i = 0.0; i < 200.0; i += 1.0) {
          if (i >= steps || acc.a > 0.97) break;
          float v = texture(map, p + 0.5).r;
          v = clamp((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
          float a = pow(v, 1.6) * density * dt * 6.0;   // faint values stay transparent
          acc.rgb += (1.0 - acc.a) * a * ramp(v);
          acc.a += (1.0 - acc.a) * a;
          p += dir * dt;
        }
        outColor = acc;
      }`,
  });
  const cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  scene.add(cube);
  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.6 }));
  scene.add(frame);
  let yaw = 0.7, pitch = 0.45, dist = 2.4, dragging = false, lastX = 0, lastY = 0, dirty = true, alive = true, texture = null;
  function place() {
    camera.position.set(dist * Math.sin(yaw) * Math.cos(pitch), dist * Math.sin(pitch), dist * Math.cos(yaw) * Math.cos(pitch));
    camera.lookAt(0, 0, 0);
    uniforms.cameraPos.value.copy(camera.position);
  }
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => { if (!dragging) return; yaw -= (e.clientX - lastX) * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - lastY) * 0.008)); lastX = e.clientX; lastY = e.clientY; dirty = true; });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); dist = Math.max(1.2, Math.min(6, dist * (e.deltaY > 0 ? 1.1 : 0.9))); dirty = true; }, { passive: false });
  function resize() {
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300, r = renderer.getPixelRatio();
    if (canvas.width !== Math.round(w * r) || canvas.height !== Math.round(h * r)) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true; }
  }
  function tick() {
    if (!alive) return;
    resize();
    if (dirty && texture) { place(); renderer.render(scene, camera); dirty = false; }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  return {
    // data is (D, H, W) row-major; the volume is shown with depth along the viewer's z, height up
    setVolume(data, d, h, w, opts) {
      const arr = data instanceof Float32Array ? data : Float32Array.from(data);
      let lo = 0, hi = 1;
      if (opts && opts.normalize) { lo = Infinity; hi = -Infinity; for (let i = 0; i < arr.length; i++) { if (arr[i] < lo) lo = arr[i]; if (arr[i] > hi) hi = arr[i]; } if (!(hi > lo)) { lo = 0; hi = 1; } }
      uniforms.lo.value = lo; uniforms.hi.value = hi;
      if (texture) texture.dispose();
      texture = new THREE.Data3DTexture(arr, w, h, d);
      texture.format = THREE.RedFormat; texture.type = THREE.FloatType;
      texture.minFilter = texture.magFilter = THREE.LinearFilter;
      texture.unpackAlignment = 1; texture.needsUpdate = true;
      uniforms.map.value = texture;
      // a non-cubic output (a crop) keeps its proportions
      const m = Math.max(d, h, w); cube.scale.set(w / m, h / m, d / m); frame.scale.copy(cube.scale);
      dirty = true;
    },
    // the two viewers of a page share one orbit so input and output stay comparable
    view() { return { yaw, pitch, dist }; },
    setView(v) { yaw = v.yaw; pitch = v.pitch; dist = v.dist; dirty = true; },
    onOrbit(fn) { canvas.addEventListener("pointermove", () => { if (dragging) fn(); }); canvas.addEventListener("wheel", () => fn()); },
    dispose() { alive = false; if (texture) texture.dispose(); material.dispose(); renderer.dispose(); },
  };
}
