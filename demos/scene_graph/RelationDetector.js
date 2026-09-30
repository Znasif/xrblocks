import * as ort from 'onnxruntime-web';

/**
 * Where the RelateAnything web export lives. The authors publish the ONNX
 * graph, its predicate bank and the ONNX Runtime files next to their browser
 * demo; override with `?model=` / `?bank=` for a mirror.
 */
export const DEFAULT_MODEL_BASE =
  'https://maelic.github.io/RelateAnythingProject/demo/';
export const DEFAULT_MODEL_URL = `${DEFAULT_MODEL_BASE}models/relateanything_vits16plus_w16.onnx`;
export const DEFAULT_BANK_URL = `${DEFAULT_MODEL_BASE}assets/banks/predicate_bank.json`;
export const DEFAULT_ORT_DIST =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

/**
 * The 35 predicates the authors ship as the default operating vocabulary.
 * Any subset of the 243-entry bank can be activated at runtime; a smaller
 * vocabulary is both faster and less chatty.
 */
export const DEFAULT_PREDICATES = [
  'holding',
  'sitting on',
  'sitting at',
  'standing on',
  'looking at',
  'using',
  'leaning against',
  'part of',
  'resting on',
  'on',
  'covering',
  'inside',
  'on top of',
  'contained in',
  'hanging from',
  'attached to',
  'in front of',
  'beside',
  'to the left of',
  'to the right of',
  'behind',
  'above',
  'below',
];

function sigmoid(z) {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

/**
 * Runs the RelateAnything relation head in the browser through ONNX Runtime
 * Web. Feed it an image and boxes from any detector; get back scored
 * (subject, predicate, object) triplets. Object labels are never an input.
 */
export class RelationDetector {
  constructor({
    modelUrl = DEFAULT_MODEL_URL,
    bankUrl = DEFAULT_BANK_URL,
    ortDistUrl = DEFAULT_ORT_DIST,
    preferWebGPU = true,
    predicates = DEFAULT_PREDICATES,
    threshold = 0.4,
    pairWeight = 1.0,
    topk = 24,
    usePerPredicateThresholds = false,
  } = {}) {
    this.modelUrl = modelUrl;
    this.bankUrl = bankUrl;
    this.ortDistUrl = ortDistUrl;
    this.preferWebGPU = preferWebGPU;
    this.requestedPredicates = predicates;
    this.threshold = threshold;
    this.pairWeight = pairWeight;
    this.topk = topk;
    this.usePerPredicateThresholds = usePerPredicateThresholds;

    this.session = null;
    this.backend = null;
    this.bank = null;
    this.imgSize = 448;
    this.maxBoxes = 32;
    this.calib = {a: 1, b: 0};
    /** Bytes of model weights fetched, for the memory readout. */
    this.modelBytes = 0;
    this.loadMs = 0;
    this.vocab = null;
    this.canvas = null;
  }

  get ready() {
    return !!this.session;
  }

  async load(onStatus = () => {}) {
    const t0 = performance.now();
    onStatus('loading predicate bank…');
    const bank = await (await fetch(this.bankUrl)).json();
    this.bank = bank;
    this.imgSize = bank.img_size ?? bank.meta?.img_size ?? 448;
    this.maxBoxes = bank.max_boxes ?? bank.meta?.max_boxes ?? 32;
    const calib = bank.calibration ?? bank.meta?.calibration;
    if (calib) this.calib = {a: calib.a, b: calib.b};
    this.setVocabulary(this.requestedPredicates);

    onStatus('downloading relation model…');
    const modelResponse = await fetch(this.modelUrl);
    if (!modelResponse.ok) {
      throw new Error(`model fetch failed: ${modelResponse.status}`);
    }
    const modelBuffer = await modelResponse.arrayBuffer();
    this.modelBytes = modelBuffer.byteLength;
    // Large graphs are shipped as .onnx + .data0..N chunks (GitHub Pages has a
    // 100 MB file limit). A manifest next to the model lists them.
    const externalData = await this.loadExternalData_(onStatus);

    ort.env.wasm.wasmPaths = this.ortDistUrl;
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.max(1, Math.min(navigator.hardwareConcurrency || 1, 8))
      : 1;

    const providers = [];
    if (this.preferWebGPU && navigator.gpu) providers.push('webgpu');
    providers.push('wasm');
    let lastError = null;
    for (const ep of providers) {
      try {
        onStatus(`compiling on ${ep}…`);
        this.session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
          externalData: externalData.length ? externalData : undefined,
        });
        this.backend = ep;
        break;
      } catch (error) {
        console.warn(`[RelationDetector] ${ep} failed`, error);
        lastError = error;
      }
    }
    if (!this.session) throw lastError ?? new Error('no execution provider');
    this.inputNames = this.session.inputNames;
    this.outputNames = this.session.outputNames;
    this.loadMs = performance.now() - t0;
    onStatus(`relation model ready on ${this.backend}`);
    return this;
  }

  async loadExternalData_(onStatus) {
    const manifestUrl = this.modelUrl.replace(/[^/]+$/, 'manifest.json');
    try {
      const response = await fetch(manifestUrl);
      if (!response.ok) return [];
      const manifest = await response.json();
      const base = this.modelUrl.replace(/[^/]+$/, '');
      const name = this.modelUrl.split('/').pop();
      const entry = (manifest.models ?? manifest.relation ?? []).find?.((m) =>
        m.files?.some((f) => (f.name ?? f) === name)
      );
      const files = (entry?.files ?? [])
        .map((f) => f.name ?? f)
        .filter((f) => f.startsWith(name + '.data'));
      const out = [];
      for (const f of files) {
        onStatus(`downloading ${f}…`);
        const buf = await (await fetch(base + f)).arrayBuffer();
        this.modelBytes += buf.byteLength;
        out.push({path: f, data: new Uint8Array(buf)});
      }
      return out;
    } catch {
      return [];
    }
  }

  /** Activates a subset of the bank as the live predicate vocabulary. */
  setVocabulary(names) {
    const bank = this.bank;
    const index = new Map(bank.names.map((n, i) => [n, i]));
    const rows = [];
    for (const name of names) {
      const i = index.get(name);
      if (i === undefined) {
        console.warn(`[RelationDetector] '${name}' is not in the bank`);
        continue;
      }
      rows.push(i);
    }
    const dim = bank.dim;
    const W = new Float32Array(rows.length * dim);
    const alpha = new Float32Array(rows.length);
    const thr = new Float32Array(rows.length);
    const spatial = [];
    rows.forEach((i, v) => {
      W.set(bank.W[i], v * dim);
      alpha[v] = bank.alpha[i];
      const t = bank.thr?.[i];
      thr[v] = typeof t === 'number' && isFinite(t) ? t : this.threshold;
      spatial.push(!!bank.is_spatial?.[i]);
    });
    this.vocab = {
      names: rows.map((i) => bank.names[i]),
      W: new ort.Tensor('float32', W, [rows.length, dim]),
      alpha: new ort.Tensor('float32', alpha, [rows.length]),
      thr,
      spatial,
    };
    return this.vocab.names;
  }

  /**
   * Resizes the frame to the head's square input and packs it as CHW floats
   * in [0, 1]. A plain resize, not a letterbox: the head's geometry features
   * are normalized boxes, so the image and boxes distort identically.
   */
  preprocess_(image) {
    const S = this.imgSize;
    if (!this.canvas) {
      this.canvas =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(S, S)
          : Object.assign(document.createElement('canvas'), {
              width: S,
              height: S,
            });
    }
    const ctx = this.canvas.getContext('2d', {willReadFrequently: true});
    let source = image;
    if (image instanceof ImageData) {
      const tmp =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(image.width, image.height)
          : Object.assign(document.createElement('canvas'), {
              width: image.width,
              height: image.height,
            });
      tmp.getContext('2d').putImageData(image, 0, 0);
      source = tmp;
    }
    ctx.drawImage(source, 0, 0, S, S);
    const {data} = ctx.getImageData(0, 0, S, S);
    const chw = new Float32Array(3 * S * S);
    const plane = S * S;
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      chw[i] = data[p] / 255;
      chw[plane + i] = data[p + 1] / 255;
      chw[2 * plane + i] = data[p + 2] / 255;
    }
    return new ort.Tensor('float32', chw, [1, 3, S, S]);
  }

  /**
   * @param image - ImageData, canvas or bitmap of the frame the boxes refer to.
   * @param boxes - normalized [x0, y0, x1, y1] boxes (0..1, top-left origin).
   * @param options - `scores` per box (ranking only) and `labels` for output.
   * @returns scored triplets plus per-stage timings in ms.
   */
  async relate(image, boxes, {scores = null, labels = null} = {}) {
    if (!this.session) throw new Error('RelationDetector not loaded');
    const t0 = performance.now();
    const n = Math.min(boxes.length, this.maxBoxes);
    const boxTensor = new Float32Array(this.maxBoxes * 4);
    for (let i = 0; i < n; i++) {
      const [x0, y0, x1, y1] = boxes[i];
      boxTensor[i * 4] = (x0 + x1) / 2;
      boxTensor[i * 4 + 1] = (y0 + y1) / 2;
      boxTensor[i * 4 + 2] = x1 - x0;
      boxTensor[i * 4 + 3] = y1 - y0;
    }
    const feeds = {
      image: this.preprocess_(image),
      boxes: new ort.Tensor('float32', boxTensor, [1, this.maxBoxes, 4]),
      box_counts: new ort.Tensor('int64', BigInt64Array.from([BigInt(n)]), [1]),
    };
    if (this.inputNames.includes('W')) {
      feeds.W = this.vocab.W;
      feeds.alpha = this.vocab.alpha;
    }
    const t1 = performance.now();
    const out = await this.session.run(feeds);
    const t2 = performance.now();
    const triplets = this.decode_(out, n, scores, labels);
    const t3 = performance.now();
    return {
      triplets,
      boxCount: n,
      ms: {pre: t1 - t0, run: t2 - t1, post: t3 - t2, total: t3 - t0},
    };
  }

  decode_(out, nBoxes, boxScores, labels) {
    const pred = out.pred_logits;
    const pair = out.pair_logits;
    const sub = out.sub_idx.data;
    const obj = out.obj_idx.data;
    const valid = out.valid_mask.data;
    const [, K, V] = pred.dims;
    const {a, b} = this.calib;
    const w = this.pairWeight;
    const names = this.vocab.names;
    const thr = this.vocab.thr;
    const results = [];
    for (let k = 0; k < K; k++) {
      if (!valid[k]) continue;
      const s = Number(sub[k]);
      const o = Number(obj[k]);
      if (s === o || s >= nBoxes || o >= nBoxes) continue;
      const pl = Number(pair.data[k]);
      let best = -1;
      let bestScore = 0;
      for (let v = 0; v < V; v++) {
        const score = sigmoid(a * (Number(pred.data[k * V + v]) + w * pl) + b);
        const cut = this.usePerPredicateThresholds ? thr[v] : this.threshold;
        if (score >= cut && score > bestScore) {
          best = v;
          bestScore = score;
        }
      }
      if (best < 0) continue;
      let rank = bestScore;
      if (boxScores) rank *= (boxScores[s] ?? 1) * (boxScores[o] ?? 1);
      results.push({
        subject: s,
        object: o,
        predicate: names[best],
        score: bestScore,
        rank,
        spatial: this.vocab.spatial[best],
        subjectLabel: labels?.[s],
        objectLabel: labels?.[o],
      });
    }
    results.sort((x, y) => y.rank - x.rank);
    return results.slice(0, this.topk);
  }

  dispose() {
    this.session?.release?.();
    this.session = null;
  }
}
