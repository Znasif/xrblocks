/**
 * Voice and text search over scene-graph nodes with EmbeddingGemma 2.
 *
 * EmbeddingGemma 2 maps text, images and audio into one 768-d space, so a
 * spoken question is embedded straight from the microphone samples and
 * compared with an embedding of each object: its detector label interleaved
 * with its image crop. There is no speech-to-text step; the query never
 * becomes a string, and the image half of the object embedding still finds a
 * table the detector called "bed".
 */

export const DEFAULT_EMBED_MODEL = 'onnx-community/embeddinggemma-2-ONNX';

// Resolved by the page's import map at run time. Kept out of the static
// module graph so tests, which inject a fake library, do not need the package.
const TRANSFORMERS_MODULE = '@huggingface/transformers';

/**
 * Cosine similarity a spoken query must reach to count as a match. Measured
 * with the q4 weights on the simulator living room: queries for objects in
 * view scored 0.645 to 0.70, queries for absent objects ("the refrigerator",
 * "my car keys", "the weather") at most 0.607.
 */
export const DEFAULT_MATCH_SCORE = 0.625;

/** The audio encoder expects mono 16 kHz samples. */
export const EMBED_SAMPLE_RATE = 16000;

/** Converts 16-bit PCM (as recorded by `xb.core.sound`) to floats in [-1, 1). */
export function int16ToFloat32(pcm) {
  const ints = pcm instanceof Int16Array ? pcm : new Int16Array(pcm);
  const out = new Float32Array(ints.length);
  for (let i = 0; i < ints.length; i++) out[i] = ints[i] / 32768;
  return out;
}

/**
 * Band-limited resampling with a Hann-windowed sinc kernel. Downsampling
 * low-passes at the target Nyquist rate so 48 kHz microphone audio does not
 * alias into the 16 kHz band the audio encoder sees.
 */
export function resample(samples, fromRate, toRate = EMBED_SAMPLE_RATE) {
  if (fromRate === toRate || samples.length === 0) return samples;
  const ratio = toRate / fromRate;
  const length = Math.floor(samples.length * ratio);
  const out = new Float32Array(length);
  const cutoff = Math.min(1, ratio) * 0.95;
  const half = 16;
  for (let i = 0; i < length; i++) {
    const t = i / ratio;
    const centre = Math.floor(t);
    let acc = 0;
    let weight = 0;
    for (let k = centre - half + 1; k <= centre + half; k++) {
      if (k < 0 || k >= samples.length) continue;
      const d = t - k;
      const sinc =
        d === 0 ? cutoff : Math.sin(Math.PI * cutoff * d) / (Math.PI * d);
      const w = sinc * (0.5 + 0.5 * Math.cos((Math.PI * d) / half));
      acc += samples[k] * w;
      weight += w;
    }
    out[i] = weight ? acc / weight : 0;
  }
  return out;
}

/**
 * Drops leading and trailing silence, keeping `padSeconds` around the speech.
 * A frame is speech when its RMS clears both an absolute floor and a fraction
 * of the loudest frame. Returns an empty array when nothing clears the floor.
 */
export function trimSilence(
  samples,
  rate,
  {frameSeconds = 0.02, padSeconds = 0.15, floor = 0.01, relative = 0.1} = {}
) {
  const frame = Math.max(1, Math.round(rate * frameSeconds));
  const frames = Math.ceil(samples.length / frame);
  const rms = new Float32Array(frames);
  let peak = 0;
  for (let f = 0; f < frames; f++) {
    let sum = 0;
    const end = Math.min(samples.length, (f + 1) * frame);
    for (let i = f * frame; i < end; i++) sum += samples[i] * samples[i];
    rms[f] = Math.sqrt(sum / Math.max(1, end - f * frame));
    peak = Math.max(peak, rms[f]);
  }
  const threshold = Math.max(floor, peak * relative);
  let first = -1;
  let last = -1;
  for (let f = 0; f < frames; f++) {
    if (rms[f] < threshold) continue;
    if (first < 0) first = f;
    last = f;
  }
  if (first < 0) return new Float32Array(0);
  const pad = Math.round(rate * padSeconds);
  const start = Math.max(0, first * frame - pad);
  const end = Math.min(samples.length, (last + 1) * frame + pad);
  return samples.slice(start, end);
}

/**
 * Pixel rectangle for a normalized `[x0, y0, x1, y1]` box, grown by `pad` of
 * the box size on each side so the crop keeps a little context.
 */
export function cropRect(box, width, height, pad = 0.1) {
  const [x0, y0, x1, y1] = box;
  const px = (x1 - x0) * pad;
  const py = (y1 - y0) * pad;
  const left = Math.max(0, Math.floor((x0 - px) * width));
  const top = Math.max(0, Math.floor((y0 - py) * height));
  const right = Math.min(width, Math.ceil((x1 + px) * width));
  const bottom = Math.min(height, Math.ceil((y1 + py) * height));
  return {
    x: Math.min(left, width - 1),
    y: Math.min(top, height - 1),
    w: Math.max(1, right - left),
    h: Math.max(1, bottom - top),
  };
}

/** Copies an RGBA rectangle out of an ImageData-like `{data, width}`. */
export function cropPixels(image, rect) {
  const {x, y, w, h} = rect;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let row = 0; row < h; row++) {
    const from = ((y + row) * image.width + x) * 4;
    data.set(image.data.subarray(from, from + w * 4), row * w * 4);
  }
  return {data, width: w, height: h};
}

/** Normalizes a vector to unit length in place and returns it. */
export function normalize(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
  const norm = Math.sqrt(sum) || 1;
  for (let i = 0; i < vector.length; i++) vector[i] /= norm;
  return vector;
}

/** Ranks indexed nodes by cosine similarity to a unit query vector. */
export function rankNodes(query, vectors) {
  const ranked = [];
  for (const [id, entry] of vectors) {
    let score = 0;
    for (let i = 0; i < query.length; i++) score += query[i] * entry.vector[i];
    ranked.push({id, score});
  }
  return ranked.sort((a, b) => b.score - a.score);
}

/** The best-ranked node if it clears `minScore`, else null. */
export function pickMatch(ranked, minScore = DEFAULT_MATCH_SCORE) {
  const top = ranked[0];
  if (!top || top.score < minScore) return null;
  return {...top, margin: top.score - (ranked[1]?.score ?? 0)};
}

/** Document text for an object: its label, with the crop at `<|image|>`. */
export function objectDocument(label) {
  return `title: ${label} | text: <|image|>`;
}

/** Retrieval-query prefix for typed questions. Audio takes no prefix. */
export function textQuery(text) {
  return `task: search result | query: ${text}`;
}

/**
 * Keeps one embedding per scene-graph node and answers voice or text
 * queries against them. Every model call goes through one queue: ONNX
 * Runtime sessions do not run concurrently.
 */
export class ObjectFinder {
  constructor({
    modelId = DEFAULT_EMBED_MODEL,
    dtype = 'q4',
    preferWebGPU = true,
    softTokens = 70,
    minScore = DEFAULT_MATCH_SCORE,
    maxViews = 4,
    loadLibrary = () => import(/* @vite-ignore */ TRANSFORMERS_MODULE),
  } = {}) {
    this.modelId = modelId;
    this.dtype = dtype;
    this.preferWebGPU = preferWebGPU;
    this.softTokens = softTokens;
    this.minScore = minScore;
    this.maxViews = maxViews;
    this.loadLibrary = loadLibrary;
    this.lib = null;
    this.processor = null;
    this.model = null;
    this.backend = null;
    this.modelBytes = 0;
    this.loadMs = 0;
    /** @type {Map<string, {vector: Float32Array, label: string, views: number}>} */
    this.vectors = new Map();
    this.queue = Promise.resolve();
  }

  get ready() {
    return !!this.model;
  }

  async load(onStatus = () => {}) {
    const t0 = performance.now();
    onStatus('loading EmbeddingGemma 2…');
    this.lib = await this.loadLibrary();
    const {AutoModel, AutoProcessor} = this.lib;
    this.processor = await AutoProcessor.from_pretrained(this.modelId);
    // Object crops are small; 70 soft tokens per image keeps each embedding
    // cheap and well under the WebGPU per-batch token limit.
    if (this.processor.image_processor) {
      this.processor.image_processor.max_soft_tokens = this.softTokens;
    }
    const files = new Map();
    const progress_callback = (event) => {
      if (event.status !== 'progress' || !event.total) return;
      files.set(event.file, {loaded: event.loaded, total: event.total});
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      this.modelBytes = total;
      onStatus(
        `downloading EmbeddingGemma 2… ${Math.round((100 * loaded) / total)}%`
      );
    };
    const devices = [];
    if (this.preferWebGPU && globalThis.navigator?.gpu) devices.push('webgpu');
    devices.push('wasm');
    let lastError = null;
    for (const device of devices) {
      try {
        onStatus(`compiling EmbeddingGemma 2 on ${device}…`);
        this.model = await AutoModel.from_pretrained(this.modelId, {
          device,
          dtype: this.dtype,
          progress_callback,
        });
        this.backend = device;
        break;
      } catch (error) {
        console.warn(`[ObjectFinder] ${device} failed`, error);
        lastError = error;
      }
    }
    if (!this.model) throw lastError ?? new Error('no execution provider');
    this.loadMs = performance.now() - t0;
    onStatus(`EmbeddingGemma 2 ready on ${this.backend}`);
    return this;
  }

  /** Runs one model call after every call queued before it. */
  run_(fn) {
    const next = this.queue.then(fn, fn);
    this.queue = next.catch(() => {});
    return next;
  }

  async embed_(text, images, audio) {
    const inputs = await this.processor(text, images, audio);
    const {sentence_embedding} = await this.model(inputs);
    return normalize(Float32Array.from(sentence_embedding.data));
  }

  /**
   * Adds one observation of a node: the detector label interleaved with the
   * crop of `box` from `image`. Repeat views are averaged, capped at
   * `maxViews` so a node keeps following what it looks like now.
   */
  addObservation(nodeId, label, image, box) {
    return this.run_(async () => {
      const crop = cropPixels(image, cropRect(box, image.width, image.height));
      const raw = new this.lib.RawImage(
        crop.data,
        crop.width,
        crop.height,
        4
      ).rgb();
      const vector = await this.embed_([objectDocument(label)], [[raw]]);
      const entry = this.vectors.get(nodeId);
      if (entry) {
        const weight = Math.min(entry.views, this.maxViews - 1);
        for (let i = 0; i < vector.length; i++) {
          entry.vector[i] = entry.vector[i] * weight + vector[i];
        }
        normalize(entry.vector);
        entry.views += 1;
        entry.label = label;
      } else {
        this.vectors.set(nodeId, {vector, label, views: 1});
      }
      return vector;
    });
  }

  /**
   * Embeds a spoken query. Accepts samples at any rate; they are resampled
   * to 16 kHz and trimmed to the speech. Returns null for silence.
   */
  embedSpeech(samples, sampleRate) {
    const speech = trimSilence(
      resample(samples, sampleRate, EMBED_SAMPLE_RATE),
      EMBED_SAMPLE_RATE
    );
    if (speech.length < EMBED_SAMPLE_RATE * 0.2) {
      return Promise.resolve(null);
    }
    return this.run_(async () => ({
      vector: await this.embed_(null, null, [speech]),
      seconds: speech.length / EMBED_SAMPLE_RATE,
    }));
  }

  /** Embeds a typed query with the retrieval prefix. */
  embedText(text) {
    return this.run_(async () => ({
      vector: await this.embed_([textQuery(text)]),
      seconds: 0,
    }));
  }

  /** Ranks every indexed node against a query vector. */
  search(vector) {
    return rankNodes(vector, this.vectors);
  }

  /** The best match for a query vector, or null below `minScore`. */
  match(vector) {
    return pickMatch(this.search(vector), this.minScore);
  }

  dispose() {
    this.model?.dispose?.();
    this.model = null;
    this.vectors.clear();
  }
}
