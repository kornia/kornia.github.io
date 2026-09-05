// kornia.org/playground/robot: a MuJoCo car with a camera, driving in a small textured room, with the
// camera feed running through the playground's ONNX graphs. Physics: the official MuJoCo WebAssembly
// bindings (@mujoco/mujoco). Rendering: three.js, mirroring MuJoCo's geoms every frame (the WASM build
// has no renderer of its own; the geom-to-three mapping follows zalo/mujoco_wasm, MIT).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "@mujoco/mujoco";

const ROOT = new URL(document.body.dataset.base || "../", location.href).href;   // the playground root
const CAM = 256;                    // the robot camera renders at the playground's graph size
const el = (tag, cls, attrs) => { const n = document.createElement(tag); if (cls) n.className = cls; for (const k in attrs || {}) n.setAttribute(k, attrs[k]); return n; };
const $ = (id) => document.getElementById(id);
const status = (text, error) => { const s = $("rb-status"); s.textContent = text; s.classList.toggle("pg-error", !!error); };

// MuJoCo is z-up, three.js is y-up: (x, y, z) -> (x, z, -y)
const mjPos = (buf, i, target) => target.set(buf[i * 3], buf[i * 3 + 2], -buf[i * 3 + 1]);
const mjQuat = (buf, i, target) => target.set(-buf[i * 4 + 1], -buf[i * 4 + 3], buf[i * 4 + 2], -buf[i * 4 + 0]);
const SWZ = new THREE.Matrix3().set(1, 0, 0, 0, 0, 1, 0, -1, 0);   // MuJoCo world -> three world
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// The scenes. Each is a MuJoCo model with exactly one camera (the one the page reads), a body the
// third-person view follows, the controls that map keys to actuators, and an autopilot that gets the
// red-mask centroid of the camera frame (dx, dy in -0.5..0.5, n pixels, frac of the frame) every other frame.
const SCENES = {
  car: {
    title: "Car in a room", xml: "scene.xml", follow: "car", viewPos: [-0.6, 0.5, 0.6],
    summary: "MuJoCo's example car in a textured room, with a forward camera.",
    hints: "<kbd>\u2191</kbd><kbd>\u2193</kbd> drive <kbd>\u2190</kbd><kbd>\u2192</kbd> steer <span class=\"pg-note\">drag to orbit</span>",
    autopilotLabel: "Autopilot: follow the red ball",
    details: [["Scene", "a 6 m room with textured walls, a crate, a pillar and two balls that roll when pushed"], ["Camera", "on the chassis looking forward, 65\u00b0, 256\u00d7256"], ["Controls", "two motors on the wheels through forward and turn tendons"], ["Autopilot", "kornia rgb_to_hsv on the frame, a red mask, and its centroid steers the car"]],
    setup(ctx) {
      let steer = 0, drive = 0;
      return {
        control(dt, auto) {
          const k = ctx.keys, d = ctx.data;
          if (auto) { d.ctrl[0] = drive; d.ctrl[1] = steer; return; }
          d.ctrl[0] = (k.ArrowUp || k.w ? 1 : 0) - (k.ArrowDown || k.s ? 1 : 0);
          d.ctrl[1] = ((k.ArrowLeft || k.a ? 1 : 0) - (k.ArrowRight || k.d ? 1 : 0)) * 0.6;
        },
        autopilot(m) {
          if (m.n > 12) { steer = -m.dx * 2.2; drive = m.frac > 0.12 ? 0 : 0.9; return "red ball at " + Math.round((m.dx + 0.5) * 100) + " % of the frame width"; }
          steer = 0.45; drive = 0.25;
          return "searching for the red ball\u2026";
        },
      };
    },
  },
  arm: {
    title: "Arm on a table", xml: "arm.xml", follow: "base", viewPos: [-0.75, 0.85, 0.7],
    summary: "A four-joint arm with a wrist camera above a table of blocks.",
    hints: "<kbd>\u2190</kbd><kbd>\u2192</kbd> yaw <kbd>\u2191</kbd><kbd>\u2193</kbd> shoulder <kbd>W</kbd><kbd>S</kbd> elbow <kbd>A</kbd><kbd>D</kbd> wrist <span class=\"pg-note\">drag to orbit</span>",
    autopilotLabel: "Autopilot: look at the red block",
    sliders: [{ name: "yaw", min: -2.6, max: 2.6 }, { name: "shoulder", min: -1.3, max: 1.3 }, { name: "elbow", min: -1.6, max: 1.6 }, { name: "wrist", min: -1.7, max: 1.7 }],
    targets: [0, 0.3, 0.3, 0.15],   // the camera ends up about 13 cm above the table looking 45° down at the blocks
    details: [["Scene", "a table with three blocks and a ball, in the same room"], ["Camera", "on the wrist, looking along the last link, 70\u00b0, 256\u00d7256"], ["Controls", "four position actuators: base yaw, shoulder, elbow and wrist pitch"], ["Autopilot", "the red mask centroid drives yaw and wrist pitch until the block is centred"]],
    setup(ctx) {
      const t = this.targets.slice(), lim = this.sliders;
      let sweep = 1;
      const rate = 1.4;   // rad/s from the keys
      const home = this.targets;
      return {
        targets: t,
        engage() { sweep = t[0] >= 0 ? -1 : 1; },   // start the search by turning back toward the centre
        control(dt, auto) {
          const k = ctx.keys;
          if (!auto) {
            t[0] += ((k.ArrowLeft ? 1 : 0) - (k.ArrowRight ? 1 : 0)) * rate * dt;
            t[1] += ((k.ArrowUp ? 1 : 0) - (k.ArrowDown ? 1 : 0)) * rate * dt;
            t[2] += ((k.w ? 1 : 0) - (k.s ? 1 : 0)) * rate * dt;
            t[3] += ((k.d ? 1 : 0) - (k.a ? 1 : 0)) * rate * dt;
          }
          for (let i = 0; i < 4; i++) { t[i] = clamp(t[i], lim[i].min, lim[i].max); ctx.data.ctrl[i] = t[i]; }
          ctx.onTargets(t);
        },
        autopilot(m) {
          if (m.n > 12) {
            t[0] -= m.dx * 0.12;   // to the right in the frame: yaw right
            t[3] += m.dy * 0.12;   // below the centre: pitch down
            return "red block at " + Math.round((m.dx + 0.5) * 100) + " %, " + Math.round((m.dy + 0.5) * 100) + " % of the frame";
          }
          // not in view: bring the shoulder, elbow and wrist back to the pose that looks at the table, and sweep the whole yaw range
          for (let i = 1; i < 4; i++) t[i] += (home[i] - t[i]) * 0.08;
          t[0] += 0.025 * sweep;
          if (t[0] > 2.4 || t[0] < -2.4) sweep = -sweep;
          return "sweeping for the red block\u2026";
        },
      };
    },
  },
};
SCENES.humanoid = {
  title: "Humanoid", xml: "humanoid.xml", follow: "torso", viewPos: [-2.2, 4.2, 3.0],
  summary: "A 1.6 m humanoid walking around the room, with a camera in its head.",
  hints: "<kbd>\u2191</kbd><kbd>\u2193</kbd> walk <kbd>\u2190</kbd><kbd>\u2192</kbd> turn <span class=\"pg-note\">drag to orbit</span>",
  autopilotLabel: "Autopilot: walk to the red ball",
  details: [["Scene", "the 6 m room with 2 m walls, a crate, a pillar, a cone and two balls"], ["Camera", "in the head, 1.5 m up, looking 35\u00b0 down, 75\u00b0, 256\u00d7256"], ["Body", "the pelvis moves on two slides and a yaw hinge that the page drives, so the figure cannot fall; ten position actuators run the gait"], ["Autopilot", "the red mask centroid steers the heading; it stops once the ball sits low in the frame, about 0.8 m away"]],
  setup(ctx) {
    const { data, model, mujoco } = ctx;
    const qadr = (name) => model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, name)];
    const root = [qadr("root_x"), qadr("root_y"), qadr("root_yaw")];
    const pose = { x: data.qpos[root[0]], y: data.qpos[root[1]], yaw: 0 };
    let phase = 0, speed = 0, turn = 0, arrived = false;     // speed in m/s, turn in rad/s
    const SPEED = 0.55, TURN = 1.1, STEP_HZ = 1.1;
    function gait(dt) {
      const walking = Math.abs(speed) > 0.01;
      phase = walking ? (phase + 2 * Math.PI * STEP_HZ * dt * Math.sign(speed) + 2 * Math.PI) % (2 * Math.PI) : phase * 0.9;   // backwards runs the cycle in reverse
      const a = walking ? 1 : 0;
      const sw = Math.sin(phase), swR = Math.sin(phase + Math.PI);
      const c = data.ctrl;
      // hips: negative swings the leg forward; knees flex while the leg swings forward; ankles follow lightly
      c[0] = -0.45 * sw * a;            c[1] = 0.9 * Math.max(0, sw) * a;  c[2] = 0.15 * sw * a;
      c[3] = -0.45 * swR * a;           c[4] = 0.9 * Math.max(0, swR) * a; c[5] = 0.15 * swR * a;
      // arms swing opposite to the leg on the same side, elbows slightly bent
      c[6] = 0.35 * sw * a;  c[7] = -0.4 - 0.2 * Math.max(0, -sw) * a;
      c[8] = 0.35 * swR * a; c[9] = -0.4 - 0.2 * Math.max(0, -swR) * a;
    }
    return {
      control(dt, auto) {
        const k = ctx.keys;
        if (!auto) {
          speed = (k.ArrowUp || k.w) ? SPEED : (k.ArrowDown || k.s) ? -0.6 * SPEED : 0;
          turn = ((k.ArrowLeft || k.a ? 1 : 0) - (k.ArrowRight || k.d ? 1 : 0)) * TURN;
          if (speed || turn) arrived = false;
        }
        pose.yaw += turn * dt;
        pose.x = clamp(pose.x + speed * Math.cos(pose.yaw) * dt, -2.7, 2.7);
        pose.y = clamp(pose.y + speed * Math.sin(pose.yaw) * dt, -2.7, 2.7);
        data.qpos[root[0]] = pose.x; data.qpos[root[1]] = pose.y; data.qpos[root[2]] = pose.yaw;
        data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, "root_x")]] = 0;
        gait(dt);
      },
      autopilot(m) {
        // the ball sinks toward the bottom of the frame as the figure closes in; low enough means it has arrived
        if (m.n > 12) {
          arrived = m.dy > 0.32;
          turn = arrived ? 0 : -m.dx * 1.6;
          speed = arrived ? 0 : SPEED;
          return arrived ? "at the red ball" : "red ball at " + Math.round((m.dx + 0.5) * 100) + " % of the frame width";
        }
        if (arrived) { turn = 0; speed = 0; return "at the red ball"; }   // it left the frame below the camera: stay
        turn = 0.5; speed = 0;
        return "looking for the red ball\u2026";
      },
      reset() { pose.x = -2; pose.y = 0; pose.yaw = 0; phase = 0; speed = 0; turn = 0; arrived = false; },
    };
  },
};
const sceneId = SCENES[new URLSearchParams(location.search).get("scene")] ? new URLSearchParams(location.search).get("scene") : "car";
const sceneDef = SCENES[sceneId];   // `scene` is the three.js scene inside main()

function pageChrome() {
  document.title = sceneDef.title + " - Kornia robot simulator";
  $("rb-title").textContent = sceneDef.title;
  $("rb-summary").textContent = sceneDef.summary;
  $("rb-hints").innerHTML = sceneDef.hints;
  $("rb-autopilot-label").textContent = sceneDef.autopilotLabel;
  // the sidebar: the scenes, in the playground's list markup
  const side = $("rb-scenes");
  const pkg = el("section", "pg-pkg");
  const h2 = el("h2"); h2.textContent = "Scenes";
  const count = el("span", "pg-pkg-count"); count.textContent = String(Object.keys(SCENES).length);
  h2.appendChild(count); pkg.appendChild(h2);
  const list = el("ul", "pg-list");
  Object.keys(SCENES).forEach((id) => {
    const li = el("li", "pg-row pg-row-live");
    const a = el("a", "pg-row-name pg-row-model" + (id === sceneId ? " pg-row-current" : ""), { href: "?scene=" + id });
    if (id === sceneId) a.setAttribute("aria-current", "page");
    const t = el("span", "pg-model-title"); t.textContent = SCENES[id].title;
    const sub = el("span", "pg-model-sub"); sub.textContent = SCENES[id].autopilotLabel.replace("Autopilot: ", "");
    a.appendChild(t); a.appendChild(sub); li.appendChild(a); list.appendChild(li);
  });
  pkg.appendChild(list); side.appendChild(pkg);
}

async function main() {
  pageChrome();
  status("loading MuJoCo (10 MB) and three.js…");
  const [mujoco, registry, xml] = await Promise.all([
    load_mujoco(),
    fetch(ROOT + "registry.json").then((r) => r.json()),
    fetch(sceneDef.xml).then((r) => r.text()),
  ]);
  mujoco.FS.mkdir("/working");
  mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
  mujoco.FS.writeFile("/working/scene.xml", xml);
  const model = mujoco.MjModel.mj_loadXML("/working/scene.xml");
  const data = new mujoco.MjData(model);
  const followBody = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, sceneDef.follow);

  // ------------------------------------------------------------------ three.js scene from the MuJoCo model
  const viewport = $("rb-view");
  const renderer = new THREE.WebGLRenderer({ canvas: viewport, antialias: true, preserveDrawingBuffer: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1220);
  scene.add(new THREE.HemisphereLight(0xdfe7f5, 0x3a3a3a, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(2, 4, 1.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 12;
  sun.shadow.camera.left = sun.shadow.camera.bottom = -3; sun.shadow.camera.right = sun.shadow.camera.top = 3;
  scene.add(sun);

  const names = new Uint8Array(model.names);
  const dec = new TextDecoder();
  const bodyName = (b) => { let s = model.name_bodyadr[b], e = s; while (e < names.length && names[e] !== 0) e++; return dec.decode(names.subarray(s, e)); };
  const bodies = {};
  for (let b = 0; b < model.nbody; b++) { bodies[b] = new THREE.Group(); bodies[b].name = bodyName(b); }
  const textures = {};
  function textureFor(matId) {
    const texId = model.mat_texid[matId * 10 + 1];   // mjNTEXROLE = 10, RGB role = 1
    if (texId < 0) return null;
    if (textures[texId]) return textures[texId];
    const w = model.tex_width[texId], h = model.tex_height[texId], ch = model.tex_nchannel[texId], off = Number(model.tex_adr[texId]);
    const rgba = new Uint8Array(w * h * 4);
    const src = model.tex_data;
    for (let p = 0; p < w * h; p++) {
      rgba[p * 4] = src[off + p * ch];
      rgba[p * 4 + 1] = ch > 1 ? src[off + p * ch + 1] : rgba[p * 4];
      rgba[p * 4 + 2] = ch > 2 ? src[off + p * ch + 2] : rgba[p * 4];
      rgba[p * 4 + 3] = 255;
    }
    const t = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.colorSpace = THREE.SRGBColorSpace;
    t.needsUpdate = true;
    textures[texId] = t;
    return t;
  }
  const G = mujoco.mjtGeom;
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_group[g] >= 3) continue;
    const type = model.geom_type[g], b = model.geom_bodyid[g];
    const s = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];
    let geometry = null, isPlane = false;
    if (type === G.mjGEOM_PLANE.value) { geometry = new THREE.PlaneGeometry(s[0] * 2 || 40, s[1] * 2 || 40); isPlane = true; }
    else if (type === G.mjGEOM_SPHERE.value) geometry = new THREE.SphereGeometry(s[0], 32, 24);
    else if (type === G.mjGEOM_CAPSULE.value) geometry = new THREE.CapsuleGeometry(s[0], s[1] * 2, 8, 20);
    else if (type === G.mjGEOM_CYLINDER.value) geometry = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 32);
    else if (type === G.mjGEOM_BOX.value) geometry = new THREE.BoxGeometry(s[0] * 2, s[2] * 2, s[1] * 2);
    else if (type === G.mjGEOM_ELLIPSOID.value) geometry = new THREE.SphereGeometry(1, 24, 16);
    else if (type === G.mjGEOM_MESH.value) {
      const m = model.geom_dataid[g];
      const v = model.mesh_vert.slice(model.mesh_vertadr[m] * 3, (model.mesh_vertadr[m] + model.mesh_vertnum[m]) * 3);
      for (let i = 0; i < v.length; i += 3) { const y = v[i + 1]; v[i + 1] = v[i + 2]; v[i + 2] = -y; }
      const f = model.mesh_face.slice(model.mesh_faceadr[m] * 3, (model.mesh_faceadr[m] + model.mesh_facenum[m]) * 3);
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(v, 3));
      geometry.setIndex(Array.from(f));
      geometry.computeVertexNormals();
    } else continue;
    let color = [model.geom_rgba[g * 4], model.geom_rgba[g * 4 + 1], model.geom_rgba[g * 4 + 2], model.geom_rgba[g * 4 + 3]];
    let map = null, repeat = [1, 1];
    const matId = model.geom_matid[g];
    if (matId >= 0) {
      color = [model.mat_rgba[matId * 4], model.mat_rgba[matId * 4 + 1], model.mat_rgba[matId * 4 + 2], model.mat_rgba[matId * 4 + 3]];
      map = textureFor(matId);
      repeat = [model.mat_texrepeat[matId * 2], model.mat_texrepeat[matId * 2 + 1]];
    }
    if (map) { map = map.clone(); map.repeat.set(repeat[0] * (isPlane ? 1 : 1), repeat[1]); map.needsUpdate = true; }
    const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(color[0], color[1], color[2]), map: map, roughness: 0.85, metalness: 0.05, transparent: color[3] < 1, opacity: color[3] });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = !isPlane;
    mesh.receiveShadow = true;
    mjPos(model.geom_pos, g, mesh.position);
    if (isPlane) mesh.rotation.x = -Math.PI / 2; else mjQuat(model.geom_quat, g, mesh.quaternion);
    if (type === G.mjGEOM_ELLIPSOID.value) mesh.scale.set(s[0], s[2], s[1]);
    bodies[b].add(mesh);
  }
  for (let b = 0; b < model.nbody; b++) scene.add(bodies[b]);

  // third-person camera that follows the car
  const viewCam = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
  viewCam.position.set(sceneDef.viewPos[0], sceneDef.viewPos[1], sceneDef.viewPos[2]);
  const orbit = new OrbitControls(viewCam, viewport);
  orbit.enableDamping = true;
  orbit.maxPolarAngle = Math.PI / 2 - 0.02;
  function resize() {
    // the view fills its box; the box is as tall as the three feeds beside it (CSS), or 2:1 when they wrap below
    const box = viewport.parentElement;
    const w = box.clientWidth || 640, h = box.clientHeight || Math.round(w / 2);
    renderer.setSize(w, h, false);
    viewCam.aspect = w / h;
    viewCam.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // ------------------------------------------------------------------ the robot's camera: colour and depth
  const fovy = model.cam_fovy[0];
  const robotCam = new THREE.PerspectiveCamera(fovy, 1, 0.02, 12);
  const colorTarget = new THREE.WebGLRenderTarget(CAM, CAM, { depthBuffer: true });
  const depthTarget = new THREE.WebGLRenderTarget(CAM, CAM);
  const depthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  const pixels = new Uint8Array(CAM * CAM * 4);
  const camCanvas = $("rb-cam"), outCanvas = $("rb-out"), depthCanvas = $("rb-depth");
  const camImage = camCanvas.getContext("2d").createImageData(CAM, CAM);
  const depthImage = depthCanvas.getContext("2d").createImageData(CAM, CAM);
  const trueDepth = new Float32Array(CAM * CAM);   // metres, row 0 at the top
  const camXmat = new THREE.Matrix3(), camRot = new THREE.Matrix4();
  function placeRobotCam() {
    mjPos(data.cam_xpos, 0, robotCam.position);
    const m = data.cam_xmat;
    camXmat.set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);   // world <- camera, MuJoCo axes
    camXmat.premultiply(SWZ);                                             // world axes to three
    camRot.setFromMatrix3(camXmat);
    robotCam.quaternion.setFromRotationMatrix(camRot);
  }
  // the point-cloud toggle on the depth feed
  let cloudOn = false, cloudViewer = null, cloudMod = null, cloudFramed = false;
  const cloudBtn = $("rb-depth-3d"), cloudCanvas = $("rb-cloud");
  cloudBtn.addEventListener("click", async () => {
    cloudOn = !cloudOn;
    cloudBtn.setAttribute("aria-pressed", cloudOn ? "true" : "false");
    depthCanvas.hidden = cloudOn; cloudCanvas.hidden = !cloudOn;
    if (cloudOn && !cloudViewer) {
      try { cloudMod = await import("../viewer3d.js"); cloudViewer = await cloudMod.createViewer(cloudCanvas); cloudFramed = false; }
      catch (e) { status("point cloud unavailable: " + (e.message || e), true); cloudOn = false; depthCanvas.hidden = false; cloudCanvas.hidden = true; }
    }
  });
  const TURBO = [[48, 18, 59], [70, 107, 227], [40, 187, 236], [31, 233, 168], [135, 253, 78], [225, 220, 55], [253, 149, 39], [223, 64, 17], [122, 4, 3]];
  const turbo = (t) => { const x = Math.max(0, Math.min(1, t)) * 8, i = Math.min(7, Math.floor(x)), f = x - i, a = TURBO[i], b = TURBO[i + 1]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; };
  function renderRobotCam() {
    placeRobotCam();
    // colour
    renderer.setRenderTarget(colorTarget);
    renderer.render(scene, robotCam);
    renderer.readRenderTargetPixels(colorTarget, 0, 0, CAM, CAM, pixels);
    for (let y = 0; y < CAM; y++) camImage.data.set(pixels.subarray((CAM - 1 - y) * CAM * 4, (CAM - y) * CAM * 4), y * CAM * 4);   // GL rows are bottom-up
    camCanvas.getContext("2d").putImageData(camImage, 0, 0);
    // depth: packed window-space depth -> metres along the view axis
    scene.overrideMaterial = depthMaterial;
    renderer.setRenderTarget(depthTarget);
    renderer.render(scene, robotCam);
    scene.overrideMaterial = null;
    renderer.setRenderTarget(null);
    renderer.readRenderTargetPixels(depthTarget, 0, 0, CAM, CAM, pixels);
    const near = robotCam.near, far = robotCam.far;
    let dmin = Infinity, dmax = 0;
    for (let y = 0; y < CAM; y++) for (let x = 0; x < CAM; x++) {
      const i = ((CAM - 1 - y) * CAM + x) * 4;
      const z = (pixels[i] + pixels[i + 1] / 255 + pixels[i + 2] / 65025 + pixels[i + 3] / 16581375) / 255;   // unpackRGBAToDepth
      const viewZ = (near * far) / ((far - near) * z - far);   // perspectiveDepthToViewZ (negative)
      const d = Math.min(far, -viewZ);
      trueDepth[y * CAM + x] = d;
      if (d < dmin) dmin = d; if (d > dmax && d < far) dmax = d;
    }
    for (let i = 0; i < CAM * CAM; i++) {
      const t = 1 - Math.min(1, Math.max(0, (trueDepth[i] - dmin) / Math.max(1e-3, dmax - dmin)));   // near = bright
      const c = turbo(t);
      depthImage.data[i * 4] = c[0]; depthImage.data[i * 4 + 1] = c[1]; depthImage.data[i * 4 + 2] = c[2]; depthImage.data[i * 4 + 3] = 255;
    }
    depthCanvas.getContext("2d").putImageData(depthImage, 0, 0);
    $("rb-depth-cap").textContent = "true depth · " + dmin.toFixed(2) + " to " + dmax.toFixed(2) + " m";
    if (cloudOn && cloudViewer && (frameNo % 2 === 0)) {   // the metric depth and the colour frame as a live point cloud
      const pc = cloudMod.cloudFromDepth(trueDepth, CAM, CAM, camImage.data, CAM, CAM, fovy, 1, robotCam.far * 0.98);
      // frame around the median depth, so a near table is not dwarfed by the room behind it
      const sample = []; for (let i = 0; i < trueDepth.length; i += 37) if (trueDepth[i] < robotCam.far * 0.98) sample.push(trueDepth[i]);
      sample.sort((a, b) => a - b);
      const med = sample.length ? sample[Math.floor(sample.length / 2)] : 2;
      cloudViewer.setCloud(pc.positions, pc.colors, { keepView: cloudFramed, view: { yaw: 0.55, pitch: 0.5 }, pointScale: 0.022, frame: { center: [0, 0, -med], radius: Math.max(0.15, med * 1.3) } });
      cloudFramed = true;
    }
  }

  // ------------------------------------------------------------------ perception on the camera feed
  const sessions = {};
  const session = (url) => (sessions[url] = sessions[url] || ort.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }));
  function canvasToTensor(canvas, input) {
    const d = canvas.getContext("2d").getImageData(0, 0, CAM, CAM).data, plane = CAM * CAM;
    const mean = (input && input.mean) || [0, 0, 0], std = (input && input.std) || [1, 1, 1], scale = (input && input.scale) || 1;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) out[c * plane + i] = ((d[i * 4 + c] / 255) * scale - mean[c]) / std[c];
    return new ort.Tensor("float32", out, [1, 3, CAM, CAM]);
  }
  function tensorToCanvas(tensor, canvas, normalize) {
    const [, ch, h, w] = tensor.dims, ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h), plane = h * w, src = tensor.data;
    canvas.width = w; canvas.height = h;
    let lo = 0, k = 1;
    if (normalize) { let mn = Infinity, mx = -Infinity; for (let i = 0; i < Math.min(ch, 3) * plane; i++) { if (src[i] < mn) mn = src[i]; if (src[i] > mx) mx = src[i]; } lo = mn; k = mx > mn ? 1 / (mx - mn) : 1; }
    for (let i = 0; i < plane; i++) {
      const r = (src[i] - lo) * k, g = ch === 1 ? r : (src[plane + i] - lo) * k, b = ch === 1 ? r : (src[2 * plane + i] - lo) * k;
      img.data[i * 4] = Math.max(0, Math.min(255, r * 255)); img.data[i * 4 + 1] = Math.max(0, Math.min(255, g * 255)); img.data[i * 4 + 2] = Math.max(0, Math.min(255, b * 255)); img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // the perception menu: a few operators that run on every frame, the models that fit a frame, XFeat tracking
  // perception: one kornia operator on every camera frame, in the browser through onnxruntime-web; the neural
  // models live in the Experiment and run on kornia's server, which is not a per-frame path
  const PERCEPTION = [
    ["kornia.filters.sobel", "Sobel edges"], ["kornia.filters.laplacian", "Laplacian"], ["kornia.filters.gaussian_blur2d", "Gaussian blur"],
    ["kornia.color.rgb_to_grayscale", "Grayscale"], ["kornia.color.rgb_to_hsv", "HSV"], ["kornia.enhance.equalize", "Equalize (low light)"],
    ["kornia.morphology.dilation", "Dilation"],
  ];
  const perceptionSel = $("rb-perception");
  const ops = PERCEPTION.map(([id, label]) => ({ op: registry.ops.find((o) => o.id === id && o.mode === "onnx"), label })).filter((x) => x.op);
  ops.forEach(({ op, label }) => { const o = el("option", "", { value: "op:" + op.id }); o.textContent = label; perceptionSel.appendChild(o); });
  const maskOpt = el("option", "", { value: "mask" }); maskOpt.textContent = "Red mask (what the autopilot sees)"; perceptionSel.appendChild(maskOpt);
  const noneOpt = el("option", "", { value: "none" }); noneOpt.textContent = "Off"; perceptionSel.appendChild(noneOpt);
  let current = { kind: "none" };
  let lastRun = 0, runBusy = false;
  function selectPerception(value) {
    perceptionSel.value = value;
    if (value === "none") { current = { kind: "none" }; outCanvas.getContext("2d").clearRect(0, 0, CAM, CAM); return; }
    if (value === "mask") { current = { kind: "mask" }; return; }
    const op = ops.find((x) => "op:" + x.op.id === value);
    current = op ? { kind: "op", op: op.op } : { kind: "none" };
  }
  perceptionSel.addEventListener("change", () => selectPerception(perceptionSel.value));
  const maskImage = outCanvas.getContext("2d").createImageData(CAM, CAM);
  function drawMask(hsv) {
    const plane = CAM * CAM, d = maskImage.data;
    for (let i = 0; i < plane; i++) {
      const on = isRed(hsv[i], hsv[plane + i], hsv[2 * plane + i]);
      d[i * 4] = on ? 235 : 20; d[i * 4 + 1] = on ? 60 : 24; d[i * 4 + 2] = on ? 60 : 32; d[i * 4 + 3] = 255;
    }
    outCanvas.getContext("2d").putImageData(maskImage, 0, 0);
  }
  const isRed = (h, s, v) => s > 0.6 && v > 0.25 && (h < 0.2 || h > 6.1);   // kornia: hue in radians; the poster's orange sits at 0.5

  async function runPerception() {
    if (!current || current.kind === "none" || runBusy) return;
    runBusy = true;
    const t0 = performance.now();
    try {
      if (current.kind === "op") {
        const op = current.op;
        const key = op.select_order && op.select_order.length ? op.select_order.map((n) => { const p = op.params.find((q) => q.name === n); const v = p.default; return p.type === "int" ? String(Math.round(v)) : String(v); }).join("|") : "default";
        const s = await session(ROOT + op.graphs[key]);
        const feeds = {}; feeds[op.inputs[0]] = canvasToTensor(camCanvas);
        let k = 1;
        op.params.filter((p) => p.kind === "live").forEach((p) => { feeds[op.inputs[k++]] = new ort.Tensor("float32", new Float32Array([p.default]), [1]); });
        const r = await s.run(feeds);
        tensorToCanvas(r[Object.keys(r)[0]], outCanvas, op.output.display === "normalize");
      } else if (current.kind === "mask" && hsvOp) {
        hsvSession = hsvSession || await session(ROOT + hsvOp.graphs[Object.keys(hsvOp.graphs)[0]]);
        const feeds = {}; feeds[hsvOp.inputs[0]] = canvasToTensor(camCanvas);
        const r = await hsvSession.run(feeds);
        drawMask(r[Object.keys(r)[0]].data);
      }
      lastRun = performance.now() - t0;
    } catch (e) {
      status("perception failed: " + (e.message || e), true);
      current = { kind: "none" };
    }
    runBusy = false;
  }

  // ------------------------------------------------------------------ controls: keys, sliders, and the red-mask autopilot
  const keys = {};
  const keyOf = (e) => (e.key.length === 1 ? e.key.toLowerCase() : e.key);   // Shift or Caps Lock must not break WASD
  window.addEventListener("keydown", (e) => { const k = keyOf(e); if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "w", "a", "s", "d", " "].indexOf(k) !== -1) { keys[k] = true; if (e.target === document.body) e.preventDefault(); } });
  window.addEventListener("keyup", (e) => { keys[keyOf(e)] = false; });
  const autopilot = $("rb-autopilot");
  const sliders = [];
  const ctl = sceneDef.setup({ keys, data, model, mujoco, onTargets: (t) => { sliders.forEach((sl, i) => { if (document.activeElement !== sl.input) { sl.input.value = t[i]; sl.out.textContent = t[i].toFixed(2); } }); } });
  if (sceneDef.sliders) {
    const box = $("rb-sliders");
    sceneDef.sliders.forEach((sp, i) => {
      const row = el("div", "pg-param");
      const label = el("label", "", { for: "rb-" + sp.name }); label.textContent = sp.name;
      const input = el("input", "", { type: "range", id: "rb-" + sp.name, min: sp.min, max: sp.max, step: 0.01, value: ctl.targets[i] });
      const out = el("output"); out.textContent = ctl.targets[i].toFixed(2);
      input.addEventListener("input", () => { ctl.targets[i] = Number(input.value); out.textContent = ctl.targets[i].toFixed(2); autopilot.checked = false; });
      row.appendChild(label); row.appendChild(input); row.appendChild(out); box.appendChild(row);
      sliders.push({ input, out });
    });
    $("rb-panel").hidden = false;
  }
  let hsvSession = null;
  const hsvOp = registry.ops.find((o) => o.id === "kornia.color.rgb_to_hsv" && o.mode === "onnx");
  let autoBusy = false, autoText = "";
  async function autopilotStep() {
    // kornia's rgb_to_hsv on the frame, then a red mask (hue near 0, saturated) and its centroid; the scene decides what to do with it
    if (autoBusy || !hsvOp) return;
    autoBusy = true;
    try {
      hsvSession = hsvSession || await session(ROOT + hsvOp.graphs[Object.keys(hsvOp.graphs)[0]]);
      const feeds = {}; feeds[hsvOp.inputs[0]] = canvasToTensor(camCanvas);
      const r = await hsvSession.run(feeds);
      const hsv = r[Object.keys(r)[0]].data, plane = CAM * CAM;
      let cx = 0, cy = 0, n = 0;
      for (let i = 0; i < plane; i++) {
        const h = hsv[i], s = hsv[plane + i], v = hsv[2 * plane + i];   // kornia: hue in radians [0, 2pi]
        if (isRed(h, s, v)) { cx += i % CAM; cy += Math.floor(i / CAM); n++; }
      }
      autoText = ctl.autopilot({ n, frac: n / plane, dx: n ? (cx / n) / CAM - 0.5 : 0, dy: n ? (cy / n) / CAM - 0.5 : 0 });
    } catch (e) { autoText = "autopilot failed: " + (e.message || e); autopilot.checked = false; }
    autoBusy = false;
  }
  $("rb-reset").addEventListener("click", () => { mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data); if (ctl.targets) sceneDef.targets.forEach((v, i) => { ctl.targets[i] = v; }); if (ctl.reset) ctl.reset(); });
  autopilot.addEventListener("change", () => { autoText = ""; if (autopilot.checked && ctl.engage) ctl.engage(); });

  // ------------------------------------------------------------------ the loop
  const tmpV = new THREE.Vector3(), lookAt = new THREE.Vector3();
  let last = performance.now(), simTime = 0, frames = 0, fpsAt = performance.now(), frameNo = 0;
  const readyAt = performance.now();
  mujoco.mj_forward(model, data);
  status("ready");
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    // control
    ctl.control(dt, autopilot.checked);
    // physics at its own timestep, catching up with wall time
    simTime += dt;
    const step = model.opt.timestep;
    let n = 0;
    while (data.time < simTime && n < 40) { mujoco.mj_step(model, data); n++; }
    // mirror bodies
    for (let b = 1; b < model.nbody; b++) { mjPos(data.xpos, b, bodies[b].position); mjQuat(data.xquat, b, bodies[b].quaternion); }
    // camera follows the car
    mjPos(data.xpos, followBody, lookAt);
    orbit.target.lerp(lookAt, 0.2);
    orbit.update();
    renderer.setRenderTarget(null);
    renderer.render(scene, viewCam);
    renderRobotCam();
    frameNo++;
    if (autopilot.checked && frameNo % 2 === 0) autopilotStep();
    runPerception();
    frames++;
    if (now - fpsAt > 1000) {
      status((frames / ((now - fpsAt) / 1000)).toFixed(0) + " fps" + (current && current.kind !== "none" && lastRun ? " · perception " + lastRun.toFixed(0) + " ms" : "") + (autopilot.checked && autoText ? " · " + autoText : ""));
      frames = 0; fpsAt = now;
    }
    requestAnimationFrame(frame);
  }
  selectPerception("op:kornia.filters.sobel");
  requestAnimationFrame(frame);
  window.__robot = { model, data, mujoco, followBody, scene: sceneId, ctl };   // for probes
}

main().catch((e) => { console.error(e); status("the simulator could not start: " + (e.message || e), true); });
