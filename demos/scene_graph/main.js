import * as THREE from 'three';
import * as xb from 'xrblocks';

import {
  DEFAULT_MATCH_SCORE,
  ObjectFinder,
  int16ToFloat32,
} from './ObjectFinder.js';
import {RelationDetector} from './RelationDetector.js';
import {SceneGraph} from './SceneGraph.js';

// A real-world scene graph built entirely on device: an object detector
// supplies labelled boxes, the RelateAnything relation head supplies scored
// (subject, predicate, object) edges between them, the depth mesh grounds
// each node to a 3D point, and scans are merged as the user moves. The graph
// is the data structure a voice tool router queries ("what is on the table?").
// EmbeddingGemma 2 then answers spoken questions against it: the question is
// embedded straight from the microphone, each node from its label and image
// crop, and the closest node is highlighted in the room.

const params = new URLSearchParams(location.search);
const SPATIAL_COLOR = 0x5ba7ff;
const SEMANTIC_COLOR = 0xffb347;
const HIGHLIGHT_COLOR = 0xff4fd1;
const HIGHLIGHT_CSS = '#ff4fd1';
/** Scans kept for indexing while EmbeddingGemma 2 is still loading. */
const MAX_PENDING_SCANS = 6;
const MOVE_RESCAN_M = 0.5;
const TURN_RESCAN_RAD = 0.6;
const RESCAN_COOLDOWN_MS = 4000;

const MB = 1024 * 1024;

function makeLabelSprite(text, background) {
  const canvas = document.createElement('canvas');
  canvas.width = 384;
  canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = background;
  ctx.beginPath();
  ctx.roundRect(4, 4, canvas.width - 8, canvas.height - 8, 24);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.font = '600 44px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(
    text,
    canvas.width / 2,
    canvas.height / 2 + 2,
    canvas.width - 32
  );
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      depthTest: false,
      transparent: true,
    })
  );
  sprite.scale.set(0.32, 0.08, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function makeRingSprite() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.strokeStyle = HIGHLIGHT_CSS;
  ctx.lineWidth = 10;
  ctx.beginPath();
  ctx.arc(64, 64, 52, 0, Math.PI * 2);
  ctx.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      depthTest: false,
      transparent: true,
    })
  );
  sprite.renderOrder = 11;
  return sprite;
}

class SceneGraphDemo extends xb.Script {
  constructor() {
    super();
    this.graph = new SceneGraph();
    this.markers = new THREE.Group();
    this.add(this.markers);
    this.relations = new RelationDetector({
      modelUrl: params.get('model') ?? undefined,
      bankUrl: params.get('bank') ?? undefined,
      ortDistUrl: params.get('ort') ?? undefined,
      preferWebGPU: params.get('backend') !== 'wasm',
      threshold: Number(params.get('thr') ?? 0.4),
      usePerPredicateThresholds: params.has('bankthr'),
    });
    this.relationsError = null;
    this.finder =
      params.get('voice') === '0'
        ? null
        : new ObjectFinder({
            dtype: params.get('egdtype') ?? 'q4',
            preferWebGPU: params.get('backend') !== 'wasm',
            minScore: Number(params.get('match') ?? DEFAULT_MATCH_SCORE),
          });
    this.finderError = null;
    this.pendingIndex = [];
    this.highlightId = null;
    this.highlightRing = null;
    this.lastPreview = null;
    this.queries = [];
    this.listening = null;
    this.inXR = false;
    this.busy = false;
    this.auto = params.get('auto') !== '0';
    this.scans = [];
    this.lastScanAt = -Infinity;
    this.scanCamPos = new THREE.Vector3();
    this.scanCamQuat = new THREE.Quaternion();
    this.camPos = new THREE.Vector3();
    this.camQuat = new THREE.Quaternion();
    this.raycaster = new THREE.Raycaster();
    this.ndc = new THREE.Vector2();
    this.ui = {};
  }

  init() {
    this.ui = {
      status: document.getElementById('status'),
      metrics: document.getElementById('metrics'),
      graph: document.getElementById('graph'),
      preview: document.getElementById('preview'),
      scan: document.getElementById('scan'),
      auto: document.getElementById('auto'),
      talk: document.getElementById('talk'),
      askForm: document.getElementById('askForm'),
      askText: document.getElementById('askText'),
      answer: document.getElementById('answer'),
    };
    this.ui.scan?.addEventListener('click', () => this.scan());
    this.ui.auto?.addEventListener('click', () => {
      this.auto = !this.auto;
      this.ui.auto.classList.toggle('active', this.auto);
    });
    this.ui.auto?.classList.toggle('active', this.auto);
    const talk = this.ui.talk;
    talk?.addEventListener('pointerdown', (event) => {
      talk.setPointerCapture?.(event.pointerId);
      this.startListening();
    });
    talk?.addEventListener('pointerup', () => this.stopListening());
    talk?.addEventListener('pointercancel', () => this.stopListening());
    this.ui.askForm?.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = this.ui.askText?.value.trim();
      if (text) this.askText(text);
    });
    if (!this.finder) {
      this.ui.answer?.replaceChildren('voice search off (?voice=0)');
      talk?.setAttribute('disabled', '');
    }
    window.sceneGraphDemo = this;

    this.relations
      .load((text) => this.setStatus(text))
      .catch((error) => {
        console.warn('[scene_graph] relation model unavailable', error);
        this.relationsError = error?.message ?? String(error);
        this.setStatus(
          `relations off (${this.relationsError}); running detector + depth only`
        );
      })
      // Load the search model after the relation head so the two downloads
      // do not compete; scans made meanwhile are indexed once it is ready.
      .finally(() => this.loadFinder());
  }

  async loadFinder() {
    if (!this.finder) return;
    try {
      await this.finder.load((text) => this.setAnswer(text));
      this.setAnswer('hold “ask” and say what you are looking for');
      const pending = this.pendingIndex.splice(0);
      for (const scan of pending) await this.indexScan(scan);
    } catch (error) {
      console.warn('[scene_graph] EmbeddingGemma 2 unavailable', error);
      this.finderError = error?.message ?? String(error);
      this.pendingIndex = [];
      this.setAnswer(`voice search off (${this.finderError})`);
    }
  }

  setAnswer(text) {
    this.ui.answer?.replaceChildren(text);
    console.debug('[scene_graph:search]', text);
  }

  /** Embeds every detection of one scan into its scene-graph node. */
  async indexScan({ids, labels, boxes, imageData}) {
    for (let i = 0; i < ids.length; i++) {
      await this.finder.addObservation(ids[i], labels[i], imageData, boxes[i]);
    }
  }

  // Hold-to-talk on a headset: pinch (or trigger) anywhere while in XR.
  onXRSessionStarted() {
    this.inXR = true;
  }

  onXRSessionEnded() {
    this.inXR = false;
  }

  onSelectStart() {
    if (this.inXR) this.startListening();
  }

  onSelectEnd() {
    if (this.inXR) this.stopListening();
  }

  startListening() {
    if (this.listening) return;
    if (!this.finder?.ready) {
      this.setAnswer(
        this.finderError
          ? `voice search off (${this.finderError})`
          : 'EmbeddingGemma 2 is still loading…'
      );
      return;
    }
    this.ui.talk?.classList.add('active');
    if (this.ui.talk) this.ui.talk.textContent = 'listening… release';
    this.setAnswer('listening…');
    const sound = xb.core.sound;
    this.listening = sound
      .startRecording()
      .then(() => sound.getRecordingSampleRate());
  }

  async stopListening() {
    const started = this.listening;
    if (!started) return null;
    this.listening = null;
    this.ui.talk?.classList.remove('active');
    if (this.ui.talk) this.ui.talk.textContent = 'hold to ask';
    // Recording starts asynchronously; a quick tap must not leave it running.
    const rate = await started.catch(() => null);
    const buffer = xb.core.sound.stopRecording();
    if (!buffer || !rate) {
      this.setAnswer('no microphone audio');
      return null;
    }
    return this.askAudio(int16ToFloat32(buffer), rate);
  }

  /** Answers a spoken question given raw samples at any rate. */
  async askAudio(samples, sampleRate) {
    if (!this.finder?.ready) {
      this.setAnswer('EmbeddingGemma 2 is still loading…');
      return null;
    }
    const t0 = performance.now();
    this.setAnswer('searching…');
    const query = await this.finder.embedSpeech(samples, sampleRate);
    if (!query) {
      this.setAnswer('did not hear any speech');
      return null;
    }
    return this.answer_(query, {
      kind: 'speech',
      label: `${query.seconds.toFixed(1)} s of speech`,
      ms: performance.now() - t0,
    });
  }

  /** Answers a typed question. */
  async askText(text) {
    if (!this.finder?.ready) {
      this.setAnswer('EmbeddingGemma 2 is still loading…');
      return null;
    }
    const t0 = performance.now();
    const query = await this.finder.embedText(text);
    return this.answer_(query, {
      kind: 'text',
      label: `“${text}”`,
      ms: performance.now() - t0,
    });
  }

  answer_(query, meta) {
    const ranked = this.finder.search(query.vector);
    const match = this.finder.match(query.vector);
    this.highlightId = match?.id ?? null;
    this.renderGraph();
    if (this.lastPreview) this.drawPreview(...this.lastPreview);

    const nodes = this.graph.nodes;
    const top = ranked.slice(0, 3).map((r) => ({
      id: r.id,
      label: nodes.get(r.id)?.label,
      score: +r.score.toFixed(3),
    }));
    const answer = this.ui.answer;
    if (answer) {
      const head = document.createElement('div');
      head.className = 'query';
      head.textContent = `${meta.label} · ${meta.ms.toFixed(0)} ms`;
      const body = document.createElement('div');
      if (match) {
        body.className = 'hit';
        body.textContent = this.graph.describe(match.id);
      } else {
        body.textContent = top.length
          ? `not found (closest: ${top[0].label} ${top[0].score.toFixed(2)})`
          : 'nothing indexed yet: scan first';
      }
      const alts = document.createElement('div');
      alts.className = 'alts';
      alts.textContent = top
        .map((t) => `${t.label} ${t.score.toFixed(2)}`)
        .join(' · ');
      answer.replaceChildren(head, body, alts);
    }
    const record = {
      at: Date.now(),
      kind: meta.kind,
      query: meta.label,
      ms: +meta.ms.toFixed(0),
      match: match
        ? {
            id: match.id,
            label: nodes.get(match.id)?.label,
            score: +match.score.toFixed(3),
            margin: +match.margin.toFixed(3),
            description: this.graph.describe(match.id),
          }
        : null,
      top,
    };
    this.queries.push(record);
    return record;
  }

  setStatus(text) {
    if (this.ui.status) this.ui.status.textContent = text;
    console.debug('[scene_graph]', text);
  }

  /** Diagnostic snapshot for test drivers. */
  async debugInfo() {
    const cam = xb.core?.deviceCamera;
    let snapshot = null;
    try {
      const img = await cam?.captureSnapshot({outputFormat: 'imageData'});
      snapshot = img ? `${img.width}x${img.height}` : null;
    } catch (error) {
      snapshot = `error: ${error?.message ?? error}`;
    }
    return {
      status: this.ui.status?.textContent,
      simulator: !!xb.core?.simulator,
      xrSessionActive: !!xb.core?.renderer?.xr?.isPresenting,
      deviceCamera: cam
        ? {
            width: cam.width,
            height: cam.height,
            xrAccess: cam.isUsingXRCameraAccess,
          }
        : null,
      depthMesh: !!xb.core?.depth?.depthMesh,
      detector: !!xb.core?.world?.objects,
      snapshot,
    };
  }

  /** True once the simulator or XR session has a camera and a device camera. */
  get sceneReady() {
    return !!(
      xb.core?.camera &&
      xb.core?.deviceCamera &&
      xb.core?.world?.objects
    );
  }

  update() {
    if (this.highlightRing) {
      const pulse = 0.24 * (1 + 0.2 * Math.sin(performance.now() / 160));
      this.highlightRing.scale.set(pulse, pulse, 1);
    }
    // Billboarding is handled by sprites. Re-scan once the user has moved.
    if (!this.auto || this.busy) return;
    if (performance.now() - this.lastScanAt < RESCAN_COOLDOWN_MS) return;
    const camera = xb.core.camera;
    if (!camera) return;
    camera.getWorldPosition(this.camPos);
    camera.getWorldQuaternion(this.camQuat);
    const moved = this.camPos.distanceTo(this.scanCamPos);
    const turned = this.camQuat.angleTo(this.scanCamQuat);
    if (moved > MOVE_RESCAN_M || turned > TURN_RESCAN_RAD) this.scan();
  }

  /** One scan: snapshot, detect, relate, ground, merge into the graph. */
  async scan() {
    if (this.busy) return null;
    const detector = xb.core.world?.objects;
    const deviceCamera = xb.core.deviceCamera;
    const live = xb.core.camera;
    if (!detector || !deviceCamera || !live) return null;
    this.busy = true;
    const timings = {};
    const t0 = performance.now();
    try {
      this.lastScanAt = t0;
      live.getWorldPosition(this.scanCamPos);
      live.getWorldQuaternion(this.scanCamQuat);
      // Freeze the camera at snapshot time so grounding matches the pixels.
      const camera = live.clone();
      camera.matrixAutoUpdate = false;
      live.updateMatrixWorld();
      camera.matrixWorld.copy(live.matrixWorld);
      camera.matrixWorldInverse.copy(live.matrixWorld).invert();
      camera.projectionMatrix.copy(live.projectionMatrix);
      camera.projectionMatrixInverse.copy(live.projectionMatrixInverse);

      this.setStatus('capturing frame…');
      const imageData = await deviceCamera.captureSnapshot({
        outputFormat: 'imageData',
      });
      if (!imageData) {
        this.setStatus('no camera frame yet');
        return null;
      }
      timings.snapshot = performance.now() - t0;

      this.setStatus('detecting objects…');
      const t1 = performance.now();
      const detected =
        (await detector.runDetection({snapshot: {imageData}})) ?? [];
      timings.detect = performance.now() - t1;

      const boxes = detected.map((d) => {
        const b = d.detection2DBoundingBox;
        return [b.min.x, b.min.y, b.max.x, b.max.y];
      });
      const labels = detected.map((d) => d.label);

      let relation = {triplets: [], ms: {}};
      if (this.relations.ready && boxes.length >= 2) {
        this.setStatus(`relating ${boxes.length} objects…`);
        relation = await this.relations.relate(imageData, boxes, {labels});
      }
      timings.relate = relation.ms.total ?? 0;
      timings.relateRun = relation.ms.run ?? 0;

      const t2 = performance.now();
      const depthMesh = xb.core.depth?.depthMesh;
      const aspect = imageData.width / imageData.height;
      const grounded = detected.map((d, i) => ({
        label: d.label,
        box: boxes[i],
        point: this.groundBox(boxes[i], camera, aspect, depthMesh, d.position),
      }));
      const ids = this.graph.ingest(grounded, relation.triplets);
      timings.ground = performance.now() - t2;

      this.renderGraph();
      this.drawPreview(imageData, grounded, relation.triplets, ids);

      // Index each detection for voice search under its graph node.
      const scanIndex = {ids, labels, boxes, imageData};
      if (this.finder?.ready && ids.length) {
        this.setStatus(`indexing ${ids.length} objects for search…`);
        const t3 = performance.now();
        await this.indexScan(scanIndex);
        timings.embed = performance.now() - t3;
      } else if (this.finder && !this.finderError && ids.length) {
        this.pendingIndex.push(scanIndex);
        this.pendingIndex.splice(
          0,
          this.pendingIndex.length - MAX_PENDING_SCANS
        );
      }
      timings.total = performance.now() - t0;
      const record = {
        at: Date.now(),
        camera: {
          position: this.scanCamPos.toArray().map((v) => +v.toFixed(2)),
        },
        frame: {width: imageData.width, height: imageData.height},
        detections: grounded.map((g, i) => ({
          id: ids[i],
          label: g.label,
          box: g.box.map((v) => +v.toFixed(3)),
          point: g.point ? g.point.toArray().map((v) => +v.toFixed(2)) : null,
        })),
        triplets: relation.triplets.map((t) => ({
          subject: `${t.subjectLabel}#${t.subject}`,
          predicate: t.predicate,
          object: `${t.objectLabel}#${t.object}`,
          score: +t.score.toFixed(3),
          spatial: t.spatial,
        })),
        timings: Object.fromEntries(
          Object.entries(timings).map(([k, v]) => [k, +v.toFixed(1)])
        ),
      };
      this.scans.push(record);
      this.lastScan = record;
      this.updateMetrics(record);
      this.setStatus(
        `${detected.length} objects, ${relation.triplets.length} relations ` +
          `(${this.graph.nodes.size} nodes / ${this.graph.edges.size} edges in graph)`
      );
      return record;
    } catch (error) {
      console.error('[scene_graph] scan failed', error);
      this.setStatus(`scan failed: ${error?.message ?? error}`);
      return null;
    } finally {
      this.busy = false;
    }
  }

  /** Raycasts a box centre against the depth mesh; falls back to the detector's own estimate. */
  groundBox(box, camera, snapAspect, depthMesh, fallback) {
    const u = (box[0] + box[2]) / 2;
    const v = (box[1] + box[3]) / 2;
    let sx = 1;
    let sy = 1;
    if (snapAspect < camera.aspect) sx = snapAspect / camera.aspect;
    else if (snapAspect > camera.aspect) sy = camera.aspect / snapAspect;
    this.ndc.set((u * 2 - 1) * sx, (1 - v) * 2 * sy - sy);
    this.raycaster.setFromCamera(this.ndc, camera);
    if (depthMesh) {
      const restore = depthMesh.raycast;
      if (depthMesh.__origRaycast) depthMesh.raycast = depthMesh.__origRaycast;
      try {
        const hits = this.raycaster.intersectObject(depthMesh, true);
        if (hits.length) return hits[0].point.clone();
      } finally {
        depthMesh.raycast = restore;
      }
    }
    if (fallback && fallback.lengthSq() > 0) return fallback.clone();
    // Last resort: 2 m along the ray, so the node still has somewhere to be.
    return this.raycaster.ray.at(2, new THREE.Vector3());
  }

  renderGraph() {
    this.markers.clear();
    this.highlightRing = null;
    const focus = this.highlightId;
    const nodeSprites = new Map();
    for (const node of this.graph.nodes.values()) {
      if (!node.point) continue;
      const hit = node.id === focus;
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(hit ? 0.05 : 0.035, 12, 12),
        new THREE.MeshBasicMaterial({color: hit ? HIGHLIGHT_COLOR : 0xffffff})
      );
      dot.position.copy(node.point);
      this.markers.add(dot);
      const label = makeLabelSprite(
        node.label,
        hit ? 'rgba(200,40,165,0.95)' : 'rgba(20,28,40,0.85)'
      );
      label.position.copy(node.point).add(new THREE.Vector3(0, 0.1, 0));
      if (hit) {
        label.scale.multiplyScalar(1.4);
        label.position.y += 0.35;
        // A ring on the object and a beacon line up to its label.
        this.highlightRing = makeRingSprite();
        this.highlightRing.position.copy(node.point);
        this.markers.add(this.highlightRing);
        const beacon = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            node.point,
            label.position.clone(),
          ]),
          new THREE.LineBasicMaterial({
            color: HIGHLIGHT_COLOR,
            depthTest: false,
            transparent: true,
          })
        );
        beacon.renderOrder = 11;
        this.markers.add(beacon);
      }
      this.markers.add(label);
      nodeSprites.set(node.id, node.point);
    }
    for (const edge of this.graph.edges.values()) {
      const a = nodeSprites.get(edge.sub);
      const b = nodeSprites.get(edge.obj);
      if (!a || !b) continue;
      // With a highlighted node, its relations stay bright and the rest fade.
      const faded = focus && edge.sub !== focus && edge.obj !== focus;
      const color = edge.spatial ? SPATIAL_COLOR : SEMANTIC_COLOR;
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity: faded ? 0.15 : 0.9,
        })
      );
      this.markers.add(line);
      if (faded) continue;
      const mid = a.clone().lerp(b, 0.5);
      const tag = makeLabelSprite(
        edge.predicate,
        edge.spatial ? 'rgba(30,80,140,0.85)' : 'rgba(150,90,20,0.85)'
      );
      tag.scale.set(0.26, 0.065, 1);
      tag.position.copy(mid);
      this.markers.add(tag);
    }
    if (this.ui.graph) {
      const nodes = this.graph.nodes;
      this.ui.graph.innerHTML = [...this.graph.edges.values()]
        .sort((x, y) => y.score - x.score)
        .slice(0, 24)
        .map(
          (e) =>
            `<li class="${e.spatial ? 'spatial' : 'semantic'}` +
            `${focus && (e.sub === focus || e.obj === focus) ? ' hit' : ''}">` +
            `${nodes.get(e.sub)?.label} <b>${e.predicate}</b> ${nodes.get(e.obj)?.label}` +
            `<span class="score">${e.score.toFixed(2)}</span></li>`
        )
        .join('');
    }
  }

  drawPreview(imageData, grounded, triplets, ids = []) {
    this.lastPreview = [imageData, grounded, triplets, ids];
    const canvas = this.ui.preview;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.height = Math.round(
      (canvas.width * imageData.height) / imageData.width
    );
    const frame = document.createElement('canvas');
    frame.width = imageData.width;
    frame.height = imageData.height;
    frame.getContext('2d').putImageData(imageData, 0, 0);
    ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
    const W = canvas.width;
    const H = canvas.height;
    ctx.lineWidth = 2;
    ctx.font = '12px system-ui, sans-serif';
    grounded.forEach((g, i) => {
      const [x0, y0, x1, y1] = g.box;
      ctx.strokeStyle = '#4fe38a';
      ctx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(
        x0 * W,
        y0 * H - 14,
        ctx.measureText(`${i} ${g.label}`).width + 6,
        14
      );
      ctx.fillStyle = '#fff';
      ctx.fillText(`${i} ${g.label}`, x0 * W + 3, y0 * H - 3);
    });
    for (const t of triplets) {
      const a = grounded[t.subject]?.box;
      const b = grounded[t.object]?.box;
      if (!a || !b) continue;
      ctx.strokeStyle = t.spatial ? '#5ba7ff' : '#ffb347';
      ctx.beginPath();
      ctx.moveTo(((a[0] + a[2]) / 2) * W, ((a[1] + a[3]) / 2) * H);
      ctx.lineTo(((b[0] + b[2]) / 2) * W, ((b[1] + b[3]) / 2) * H);
      ctx.stroke();
    }
    grounded.forEach((g, i) => {
      if (!this.highlightId || ids[i] !== this.highlightId) return;
      const [x0, y0, x1, y1] = g.box;
      ctx.lineWidth = 4;
      ctx.strokeStyle = HIGHLIGHT_CSS;
      ctx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
    });
  }

  async updateMetrics(record) {
    const memory = await this.sampleMemory();
    const rows = [
      [
        'detector',
        xb.core.options?.world?.objects?.backendConfig?.activeBackend,
      ],
      [
        'relation backend',
        this.relations.backend ??
          (this.relationsError ? 'unavailable' : 'loading'),
      ],
      [
        'relation model',
        this.relations.modelBytes
          ? `${(this.relations.modelBytes / MB).toFixed(1)} MB`
          : '-',
      ],
      [
        'model load',
        this.relations.loadMs
          ? `${(this.relations.loadMs / 1000).toFixed(1)} s`
          : '-',
      ],
      ['frame', `${record.frame.width}×${record.frame.height}`],
      ['detect', `${record.timings.detect} ms`],
      [
        'relate (run)',
        `${record.timings.relate} ms (${record.timings.relateRun} ms)`,
      ],
      ['scan total', `${record.timings.total} ms`],
      [
        'search model',
        this.finder?.ready
          ? `EmbeddingGemma 2 ${this.finder.dtype} on ${this.finder.backend}, ` +
            `${(this.finder.modelBytes / MB).toFixed(0)} MB`
          : this.finderError
            ? 'unavailable'
            : this.finder
              ? 'loading'
              : 'off',
      ],
      [
        'index',
        this.finder
          ? `${this.finder.vectors.size} nodes` +
            (record.timings.embed ? ` (${record.timings.embed} ms)` : '')
          : '-',
      ],
      ['JS heap', memory.jsHeapMB ? `${memory.jsHeapMB.toFixed(0)} MB` : '-'],
      [
        'page memory',
        memory.uaTotalMB ? `${memory.uaTotalMB.toFixed(0)} MB` : '-',
      ],
    ];
    if (this.ui.metrics) {
      this.ui.metrics.innerHTML = rows
        .map(([k, v]) => `<span>${k}</span><span>${v ?? '-'}</span>`)
        .join('');
    }
    record.memory = memory;
  }

  async sampleMemory() {
    const memory = {};
    if (performance.memory) {
      memory.jsHeapMB = performance.memory.usedJSHeapSize / MB;
      memory.jsHeapLimitMB = performance.memory.jsHeapSizeLimit / MB;
    }
    if (performance.measureUserAgentSpecificMemory) {
      try {
        const result = await performance.measureUserAgentSpecificMemory();
        memory.uaTotalMB = result.bytes / MB;
        memory.breakdown = result.breakdown
          .filter((b) => b.bytes > MB / 2)
          .map((b) => ({types: b.types, mb: +(b.bytes / MB).toFixed(1)}));
      } catch (error) {
        memory.uaError = String(error?.message ?? error);
      }
    }
    memory.modelMB = +(this.relations.modelBytes / MB).toFixed(1);
    memory.backend = this.relations.backend;
    if (navigator.gpu && !memory.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        const info = adapter?.info;
        memory.gpu = info
          ? `${info.vendor} ${info.architecture} ${info.description}`.trim()
          : null;
      } catch {
        memory.gpu = null;
      }
    }
    return memory;
  }

  /** Moves the simulator user; yaw/pitch in degrees. */
  teleport(position, yawDeg = 0, pitchDeg = 0) {
    const camera = xb.core.camera;
    camera.position.set(position[0], position[1], position[2]);
    camera.quaternion.setFromEuler(
      new THREE.Euler(
        THREE.MathUtils.degToRad(pitchDeg),
        THREE.MathUtils.degToRad(yawDeg),
        0,
        'YXZ'
      )
    );
    camera.updateMatrixWorld();
  }

  getReport() {
    return {
      relationBackend: this.relations.backend,
      relationsError: this.relationsError,
      modelMB: +(this.relations.modelBytes / MB).toFixed(1),
      loadMs: +this.relations.loadMs.toFixed(0),
      vocabulary: this.relations.vocab?.names ?? [],
      graph: this.graph.toJSON(),
      scans: this.scans,
      descriptions: [...this.graph.nodes.keys()].map((id) =>
        this.graph.describe(id)
      ),
      search: this.finder
        ? {
            backend: this.finder.backend,
            error: this.finderError,
            dtype: this.finder.dtype,
            modelMB: +(this.finder.modelBytes / MB).toFixed(1),
            loadMs: +this.finder.loadMs.toFixed(0),
            minScore: this.finder.minScore,
            indexed: [...this.finder.vectors].map(([id, e]) => ({
              id,
              label: e.label,
              views: e.views,
            })),
            queries: this.queries,
          }
        : null,
    };
  }
}

function start() {
  const options = new xb.Options();
  options.deviceCamera.enabled = true;
  options.permissions.camera = true;
  // Voice search records from the microphone; ask before the session starts.
  options.permissions.microphone = params.get('voice') !== '0';
  // ObjectDetector declares the AI service as a dependency even when the
  // on-device MediaPipe backend is selected, so it has to be enabled. No key
  // is needed: without one the AI module only logs a warning.
  options.enableAI();
  options.world.enableObjectDetection();
  options.world.objects.backendConfig.activeBackend =
    params.get('detector') === 'gemini' ? 'gemini' : 'mediapipe';
  options.world.objects.backendConfig.mediapipe.scoreThreshold = Number(
    params.get('det') ?? 0.3
  );
  options.world.objects.showDebugVisualizations = false;
  options.depth.enabled = true;
  options.depth.depthMesh.enabled = true;
  options.xrButton.showEnterSimulatorButton = true;
  if (params.has('headless')) {
    options.enableAutomationMode({enableHands: false});
  }
  options.setAppTitle('Scene Graph');
  options.setAppDescription(
    'Objects and their relations, detected on device and grounded in the room.'
  );
  xb.add(new SceneGraphDemo());
  xb.init(options);
}

document.addEventListener('DOMContentLoaded', start);
