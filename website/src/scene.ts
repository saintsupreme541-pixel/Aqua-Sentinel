import * as THREE from "three";

export type NodeShape =
  | "cluster"
  | "slab"
  | "swarm"
  | "knot"
  | "beacon"
  | "monolith"
  | "torus"
  | "shard"
  | "orb"
  | "lattice"
  | "helix";

export interface SceneNodeDef {
  id: string;
  label: string;
  sub: string;
  hue: number;
  shape: NodeShape;
  group: "section" | "project";
}

interface SceneNode {
  def: SceneNodeDef;
  group: THREE.Group;
  materials: THREE.Material[];
  basePos: THREE.Vector3;
  phase: number;
  spin: number;
  showT: number;
  showTarget: number;
  hoverT: number;
  focusT: number; // 1 = normal, 0.25 = receded, 1.35 = focused
  focusTarget: number;
  opTarget: number;
}

const BG = 0x070a12;
const EASE = {
  outCubic: (t: number) => 1 - Math.pow(1 - t, 3),
  inOutCubic: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  outBack: (t: number) => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2),
};

export interface EngineOptions {
  container: HTMLElement;
  nodes: SceneNodeDef[];
  onSelect: (id: string) => void;
  onHover: (id: string | null) => void;
  isMobile: boolean;
}

function hueColor(h: number, s: number, l: number): THREE.Color {
  return new THREE.Color(`hsl(${h}, ${s}%, ${l}%)`);
}

function spriteTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.4, "rgba(255,255,255,0.5)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export class SceneEngine {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private raycaster: THREE.Raycaster;
  private pointer: THREE.Vector2;
  private pointerSmooth = new THREE.Vector2(0, 0);
  private pointerActive = false;
  private nodes: SceneNode[] = [];
  private hitboxes: THREE.Mesh[] = [];
  private core!: THREE.Group;
  private coreShell!: THREE.Mesh;
  private particles!: THREE.Points;
  private particleBoost = 0;
  private clock = new THREE.Clock();
  private raf = 0;
  private disposed = false;
  private hoveredId: string | null = null;
  private downPos: { x: number; y: number } | null = null;
  private timeouts: number[] = [];
  private showTimeouts: number[] = [];
  private opts: EngineOptions;

  // camera rig state
  private camBase = new THREE.Vector3(0, 7.5, 27);
  private lookBase = new THREE.Vector3(0, 0.4, 0);
  private camFrom = new THREE.Vector3();
  private camTo = new THREE.Vector3();
  private lookFrom = new THREE.Vector3();
  private lookTo = new THREE.Vector3();
  private tweenT = 1;
  private tweenDur = 1;
  private tweenEase: (t: number) => number = EASE.inOutCubic;

  private homePose = { cam: new THREE.Vector3(0, 2.4, 12.4), look: new THREE.Vector3(0, 0.2, 0) };
  private workPose = { cam: new THREE.Vector3(0, 2.0, 13.6), look: new THREE.Vector3(0, 0.4, 0) };
  // narrow viewports: compress the orbital ring so every node stays on-screen
  private ringScale = 1;

  constructor(opts: EngineOptions) {
    this.opts = opts;
    const { container, isMobile } = opts;

    // Horizontal FOV shrinks with aspect; compress ring + pull camera so the
    // full orbit fits on narrow windows/phones instead of spilling off-screen.
    const aspect = container.clientWidth / Math.max(1, container.clientHeight);
    this.ringScale = THREE.MathUtils.clamp(aspect / 1.45, 0.55, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 1.5 : 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setClearColor(BG, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(BG, 0.028);

    this.camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 120);
    this.camera.position.copy(this.camBase);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(-10, -10);

    this.buildLights();
    this.buildCore();
    this.buildParticles(isMobile);
    for (const def of opts.nodes) this.addNode(def);

    if (isMobile) {
      this.homePose.cam.multiplyScalar(1.18);
      this.workPose.cam.multiplyScalar(1.18);
      this.homePose.cam.y = 3.4;
      this.workPose.cam.y = 3.0;
    }
    const pull = aspect < 1.0 ? 1.3 : aspect < 1.45 ? 1.12 : 1;
    this.homePose.cam.multiplyScalar(pull);
    this.workPose.cam.multiplyScalar(pull);

    window.addEventListener("resize", this.onResize);
    window.addEventListener("pointermove", this.onPointerMove);
    container.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("pointerup", this.onPointerUp);

    this.loop();
  }

  /* ---------------- build ---------------- */

  private buildLights() {
    this.scene.add(new THREE.AmbientLight(0x223038, 1.4));
    const key = new THREE.PointLight(0xf2ead8, 60, 60, 1.8);
    key.position.set(0, 2.5, 0);
    this.scene.add(key);
    const cool = new THREE.DirectionalLight(0x6d86a8, 1.1);
    cool.position.set(-8, 6, -6);
    this.scene.add(cool);
    const warm = new THREE.DirectionalLight(0xc9b896, 0.5);
    warm.position.set(9, -3, 7);
    this.scene.add(warm);
  }

  private buildCore() {
    this.core = new THREE.Group();

    const inner = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.35, 3),
      new THREE.MeshStandardMaterial({
        color: hueColor(205, 18, 16),
        roughness: 0.28,
        metalness: 0.55,
        emissive: hueColor(205, 30, 10),
        emissiveIntensity: 0.35,
      })
    );
    this.core.add(inner);

    this.coreShell = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.85, 1),
      new THREE.MeshBasicMaterial({ color: hueColor(190, 30, 70), wireframe: true, transparent: true, opacity: 0.14 })
    );
    this.core.add(this.coreShell);

    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(2.4, 32, 24),
      new THREE.MeshBasicMaterial({
        color: hueColor(200, 40, 60),
        transparent: true,
        opacity: 0.045,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.BackSide,
      })
    );
    this.core.add(halo);

    this.scene.add(this.core);
  }

  private buildParticles(isMobile: boolean) {
    const count = isMobile ? 320 : 900;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 13 + Math.random() * 17;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = (Math.random() - 0.5) * 24;
      positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.085,
      map: spriteTexture(),
      color: 0xbfd2d8,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    this.particles = new THREE.Points(geo, mat);
    this.scene.add(this.particles);
  }

  private buildShape(kind: NodeShape, hue: number): { group: THREE.Group; materials: THREE.Material[] } {
    const group = new THREE.Group();
    const materials: THREE.Material[] = [];
    const c = hueColor(hue, 42, 64);
    const mk = (geo: THREE.BufferGeometry, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) => {
      const mat = new THREE.MeshStandardMaterial({
        color: c,
        roughness: 0.34,
        metalness: 0.28,
        emissive: hueColor(hue, 45, 22),
        emissiveIntensity: 0.25,
        transparent: true,
        ...extra,
      });
      materials.push(mat);
      mat.userData.baseOpacity = 1;
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);
      return mesh;
    };

    switch (kind) {
      case "cluster": {
        mk(new THREE.IcosahedronGeometry(0.3, 0));
        for (let i = 0; i < 3; i++) {
          const a = (i / 3) * Math.PI * 2;
          const t = mk(new THREE.TetrahedronGeometry(0.26, 0));
          t.position.set(Math.cos(a) * 0.62, Math.sin(a * 2) * 0.2, Math.sin(a) * 0.62);
        }
        break;
      }
      case "slab": {
        mk(new THREE.BoxGeometry(0.5, 1.55, 0.5));
        const edgeMat = new THREE.LineBasicMaterial({ color: hueColor(hue, 30, 75), transparent: true, opacity: 0.35 });
        edgeMat.userData.baseOpacity = 0.35;
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.78, 1.85, 0.78)), edgeMat);
        materials.push(edgeMat);
        group.add(edges);
        break;
      }
      case "swarm": {
        const n = 200;
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
          const r = 0.35 + Math.random() * 0.55;
          const th = Math.random() * Math.PI * 2;
          const ph = Math.acos(2 * Math.random() - 1);
          pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
          pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
          pos[i * 3 + 2] = r * Math.cos(ph);
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        const m = new THREE.PointsMaterial({
          size: 0.055,
          map: spriteTexture(),
          color: hueColor(hue, 55, 75),
          transparent: true,
          opacity: 0.9,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });
        m.userData.baseOpacity = 0.9;
        materials.push(m);
        group.add(new THREE.Points(g, m));
        break;
      }
      case "knot":
        mk(new THREE.TorusKnotGeometry(0.4, 0.12, 110, 14));
        break;
      case "beacon": {
        const cone = mk(new THREE.ConeGeometry(0.4, 1.05, 4));
        cone.rotation.x = Math.PI;
        cone.position.y = 0.2;
        const tip = mk(new THREE.SphereGeometry(0.14, 20, 16), { emissiveIntensity: 1.4 });
        tip.position.y = -0.42;
        break;
      }
      case "monolith":
        mk(new THREE.BoxGeometry(0.58, 1.65, 0.34));
        break;
      case "torus":
        mk(new THREE.TorusGeometry(0.48, 0.17, 18, 40));
        break;
      case "shard": {
        const s = mk(new THREE.OctahedronGeometry(0.52, 0));
        s.scale.set(1, 1.75, 1);
        break;
      }
      case "orb":
        mk(new THREE.SphereGeometry(0.5, 32, 24), { roughness: 0.15, metalness: 0.4 });
        break;
      case "lattice": {
        mk(new THREE.IcosahedronGeometry(0.38, 0));
        const wireMat = new THREE.LineBasicMaterial({ color: hueColor(hue, 45, 70), transparent: true, opacity: 0.5 });
        wireMat.userData.baseOpacity = 0.5;
        const wire = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(0.58, 0)), wireMat);
        materials.push(wireMat);
        group.add(wire);
        break;
      }
      case "helix":
        mk(new THREE.TorusKnotGeometry(0.36, 0.1, 96, 12, 2, 5));
        break;
    }
    return { group, materials };
  }

  private addNode(def: SceneNodeDef) {
    const { group, materials } = this.buildShape(def.shape, def.hue);
    const isSection = def.group === "section";

    const hit = new THREE.Mesh(
      new THREE.SphereGeometry(isSection ? 1.0 : 0.95, 8, 8),
      new THREE.MeshBasicMaterial({ visible: false })
    );
    hit.userData.nodeId = def.id;
    group.add(hit);
    this.hitboxes.push(hit);

    const idx = this.nodes.length;
    const ringR = (isSection ? 6.3 : 7.7) * this.ringScale;
    const ys = isSection ? [0.5, -0.5, 0.95, -0.75, 0.35] : [-0.5, 0.75, -0.15, 0.95, -0.85, 0.45];
    const angle = (idx / (isSection ? 5 : 6)) * Math.PI * 2 - Math.PI / 2;
    const basePos = new THREE.Vector3(Math.cos(angle) * ringR, ys[idx % ys.length], Math.sin(angle) * ringR);
    group.position.copy(basePos);
    group.visible = false;

    const node: SceneNode = {
      def,
      group,
      materials,
      basePos,
      phase: Math.random() * Math.PI * 2,
      spin: (Math.random() * 0.16 + 0.05) * (Math.random() > 0.5 ? 1 : -1),
      showT: 0,
      showTarget: 0,
      hoverT: 0,
      focusT: 1,
      focusTarget: 1,
      opTarget: 1,
    };
    this.nodes.push(node);
    this.scene.add(group);
  }

  /* ---------------- public API ---------------- */

  startIntro() {
    this.camFrom.copy(this.camBase);
    this.camTo.copy(this.homePose.cam);
    this.lookFrom.copy(this.lookBase);
    this.lookTo.copy(this.homePose.look);
    this.tweenT = 0;
    this.tweenDur = 3.4;
    this.tweenEase = EASE.inOutCubic;

    this.core.scale.setScalar(0.001);
    const start = performance.now();
    const grow = () => {
      if (this.disposed) return;
      const t = Math.min(1, (performance.now() - start) / 2000);
      this.core.scale.setScalar(Math.max(0.001, EASE.outBack(t)));
      if (t < 1) requestAnimationFrame(grow);
    };
    requestAnimationFrame(grow);

    this.timeouts.push(
      window.setTimeout(() => {
        this.nodes.forEach((n, i) => {
          if (n.def.group === "section") {
            this.timeouts.push(window.setTimeout(() => (n.showTarget = 1), i * 110));
          }
        });
      }, 900)
    );
  }

  setHome() {
    this.flyTo(this.homePose.cam, this.homePose.look, 1.7);
    this.showOnly(null);
    this.nodes.forEach((n) => {
      n.focusTarget = 1;
      n.opTarget = 1;
    });
    this.coreDim(1);
  }

  setSection(id: string | null) {
    if (id === null) {
      this.setHome();
      return;
    }
    if (id === "work") {
      this.flyTo(this.workPose.cam, this.workPose.look, 1.9);
      this.showOnly("project");
      this.nodes.forEach((n) => {
        n.focusTarget = 1;
        n.opTarget = 1;
      });
      this.coreDim(0.72);
      return;
    }
    // side sections: camera drifts toward the active node
    const node = this.nodes.find((n) => n.def.id === id);
    if (!node) return;
    const dir = node.basePos.clone().normalize();
    const cam = dir.clone().multiplyScalar(10.6 * Math.max(this.ringScale, 0.75)).add(new THREE.Vector3(0, 1.6, 0));
    this.flyTo(cam, node.basePos.clone().multiplyScalar(0.25), 1.9);
    this.showOnly("section", id);
    this.coreDim(0.8);
  }

  focusProject(id: string) {
    const node = this.nodes.find((n) => n.def.id === id);
    if (!node) return;
    const p = node.basePos;
    const side = new THREE.Vector3().crossVectors(p, new THREE.Vector3(0, 1, 0)).normalize().multiplyScalar(1.4);
    const cam = p.clone().multiplyScalar(0.52).add(side).add(new THREE.Vector3(0, 0.8, 0));
    this.flyTo(cam, p.clone(), 1.6);
    this.nodes.forEach((n) => {
      if (n.def.group !== "project") return;
      const focused = n.def.id === id;
      n.focusTarget = focused ? 1.35 : 0.28;
      n.opTarget = focused ? 1 : 0.3;
    });
    this.coreDim(0.55);
  }

  focusOverview() {
    this.flyTo(this.workPose.cam, this.workPose.look, 1.6);
    this.nodes.forEach((n) => {
      if (n.def.group === "project") {
        n.focusTarget = 1;
        n.opTarget = 1;
      }
    });
    this.coreDim(0.72);
  }

  pulse() {
    this.particleBoost = 1;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.timeouts.forEach(clearTimeout);
    this.showTimeouts.forEach(clearTimeout);
    window.removeEventListener("resize", this.onResize);
    window.removeEventListener("pointermove", this.onPointerMove);
    this.opts.container.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("pointerup", this.onPointerUp);
    this.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = (mesh as unknown as { material?: THREE.Material | THREE.Material[] }).material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else if (mat) mat.dispose();
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  /* ---------------- transitions ---------------- */

  private coreDim(scale: number) {
    // animate core scale toward target via per-frame lerp; store target on group userData
    (this.core as unknown as { targetScale: number }).targetScale = scale;
  }

  private flyTo(cam: THREE.Vector3, look: THREE.Vector3, dur: number) {
    this.camFrom.copy(this.camBase);
    this.camTo.copy(cam);
    this.lookFrom.copy(this.lookBase);
    this.lookTo.copy(look);
    this.tweenT = 0;
    this.tweenDur = dur;
    this.tweenEase = EASE.inOutCubic;
  }

  private showOnly(group: "section" | "project" | null, exceptId?: string) {
    // cancel any pending reveals from a previous call so fast navigation
    // cannot re-show nodes that were just hidden
    this.showTimeouts.forEach(clearTimeout);
    this.showTimeouts = [];
    this.nodes.forEach((n) => {
      const shouldShow =
        group === null ? n.def.group === "section" : n.def.group === group && (!exceptId || n.def.id === exceptId);
      if (shouldShow) {
        const i = this.nodes.filter((x) => x.def.group === n.def.group).indexOf(n);
        this.showTimeouts.push(window.setTimeout(() => (n.showTarget = 1), 120 + i * 95));
      } else {
        n.showTarget = 0;
      }
      n.hoverT = 0;
    });
  }

  /* ---------------- interaction ---------------- */

  private onResize = () => {
    const { container } = this.opts;
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  };

  private onPointerMove = (e: PointerEvent) => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.pointerActive = true;
    this.updateHover();
  };

  private onPointerDown = (e: PointerEvent) => {
    this.downPos = { x: e.clientX, y: e.clientY };
  };

  private onPointerUp = (e: PointerEvent) => {
    if (!this.downPos) return;
    const dx = e.clientX - this.downPos.x;
    const dy = e.clientY - this.downPos.y;
    this.downPos = null;
    if (Math.hypot(dx, dy) > 8) return;
    // re-raycast at tap point (mobile: no prior hover)
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.updateHover();
    if (this.hoveredId) this.opts.onSelect(this.hoveredId);
  };

  private updateHover() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.hitboxes, false);
    const id = hits.length > 0 ? (hits[0].object.userData.nodeId as string) : null;
    if (id !== this.hoveredId) {
      this.hoveredId = id;
      this.opts.onHover(id);
    }
  }

  /* ---------------- loop ---------------- */

  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    // camera tween
    if (this.tweenT < 1) {
      this.tweenT = Math.min(1, this.tweenT + dt / this.tweenDur);
      const k = this.tweenEase(this.tweenT);
      this.camBase.lerpVectors(this.camFrom, this.camTo, k);
      this.lookBase.lerpVectors(this.lookFrom, this.lookTo, k);
    }

    // pointer parallax (smoothed toward raw pointer for inertia)
    this.pointerSmooth.lerp(this.pointer, Math.min(1, dt * 4));
    const px = this.pointerActive ? this.pointerSmooth.x * 0.5 : 0;
    const py = this.pointerActive ? this.pointerSmooth.y * 0.3 : 0;
    this.camera.position.set(this.camBase.x + px, this.camBase.y + py, this.camBase.z);
    this.camera.lookAt(this.lookBase.x - px * 0.25, this.lookBase.y - py * 0.15, this.lookBase.z);

    // core idle + dim
    this.core.rotation.y += dt * 0.07;
    this.coreShell.rotation.y -= dt * 0.16;
    this.coreShell.rotation.x += dt * 0.05;
    const coreTarget = (this.core as unknown as { targetScale?: number }).targetScale ?? 1;
    const s = this.core.scale.x + (coreTarget - this.core.scale.x) * Math.min(1, dt * 3.2);
    this.core.scale.setScalar(s);

    // nodes
    const camDir = new THREE.Vector3();
    for (const n of this.nodes) {
      n.showT += (n.showTarget - n.showT) * Math.min(1, dt * 4.5);
      n.hoverT += ((n.def.id === this.hoveredId ? 1 : 0) - n.hoverT) * Math.min(1, dt * 7);
      n.focusT += (n.focusTarget - n.focusT) * Math.min(1, dt * 3.5);

      n.group.visible = n.showT > 0.02;
      if (!n.group.visible) continue;

      const float = Math.sin(t * 0.6 + n.phase) * 0.22;
      camDir.subVectors(this.camera.position, n.basePos).normalize();
      const push = camDir.multiplyScalar(0.42 * n.hoverT);
      n.group.position.set(n.basePos.x + push.x, n.basePos.y + float + push.y, n.basePos.z + push.z);

      n.group.rotation.y += dt * n.spin * (1 + n.hoverT * 1.6);
      const scale = n.showT * n.focusT * (1 + 0.2 * n.hoverT);
      n.group.scale.setScalar(Math.max(0.001, scale));

      const op = n.showT * n.opTarget * (1 - 0.15 * n.hoverT);
      for (const m of n.materials) {
        (m as THREE.MeshStandardMaterial).opacity = op * (m.userData.baseOpacity ?? 1);
      }
    }

    // particles
    this.particleBoost *= 0.955;
    this.particles.rotation.y += dt * 0.008;
    const pmat = this.particles.material as THREE.PointsMaterial;
    pmat.size = 0.085 * (1 + this.particleBoost * 1.6);
    pmat.opacity = 0.5 + this.particleBoost * 0.4;
    this.particles.position.x = -px * 1.6;
    this.particles.position.y = py * 1.0;

    this.renderer.render(this.scene, this.camera);
  };
}
