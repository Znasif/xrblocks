import {describe, expect, it, vi} from 'vitest';

import {
  DEFAULT_MATCH_SCORE,
  ObjectFinder,
  cropPixels,
  cropRect,
  int16ToFloat32,
  normalize,
  objectDocument,
  pickMatch,
  rankNodes,
  resample,
  textQuery,
  trimSilence,
} from './ObjectFinder.js';

function tone(freq: number, rate: number, seconds: number, amp = 0.5) {
  const out = new Float32Array(Math.round(rate * seconds));
  for (let i = 0; i < out.length; i++) {
    out[i] = amp * Math.sin((2 * Math.PI * freq * i) / rate);
  }
  return out;
}

function rms(x: Float32Array, from = 0, to = x.length) {
  let sum = 0;
  for (let i = from; i < to; i++) sum += x[i] * x[i];
  return Math.sqrt(sum / (to - from));
}

describe('audio helpers', () => {
  it('converts 16-bit PCM to floats', () => {
    const pcm = Int16Array.from([0, 16384, -32768, 32767]);
    const out = int16ToFloat32(pcm.buffer);
    expect(Array.from(out)).toEqual([0, 0.5, -1, 32767 / 32768]);
  });

  it('resamples 48 kHz to 16 kHz keeping speech-band tones', () => {
    const out = resample(tone(440, 48000, 0.5), 48000, 16000);
    expect(out.length).toBe(8000);
    // A 0.5-amplitude sine has RMS 0.354; skip the kernel's edges.
    expect(rms(out, 100, 7900)).toBeCloseTo(0.354, 2);
  });

  it('low-passes content above the target Nyquist rate', () => {
    const out = resample(tone(12000, 48000, 0.5), 48000, 16000);
    expect(rms(out, 100, 7900)).toBeLessThan(0.02);
  });

  it('returns the input when the rate already matches', () => {
    const x = tone(440, 16000, 0.1);
    expect(resample(x, 16000, 16000)).toBe(x);
  });

  it('trims leading and trailing silence around speech', () => {
    const rate = 16000;
    const x = new Float32Array(rate * 3);
    x.set(tone(300, rate, 1), rate); // speech from 1 s to 2 s
    const out = trimSilence(x, rate, {padSeconds: 0.1});
    expect(out.length / rate).toBeGreaterThan(1.15);
    expect(out.length / rate).toBeLessThan(1.25);
  });

  it('returns nothing for silence', () => {
    const quiet = tone(300, 16000, 1, 0.001);
    expect(trimSilence(quiet, 16000).length).toBe(0);
  });
});

describe('crops', () => {
  it('pads a normalized box and clamps it to the image', () => {
    expect(cropRect([0.25, 0.25, 0.75, 0.75], 100, 100, 0.1)).toEqual({
      x: 20,
      y: 20,
      w: 60,
      h: 60,
    });
    expect(cropRect([0, 0.5, 0.25, 1], 100, 100, 0.5)).toEqual({
      x: 0,
      y: 25,
      w: 38,
      h: 75,
    });
  });

  it('copies the RGBA rows of a rectangle', () => {
    const width = 4;
    const data = new Uint8ClampedArray(width * 3 * 4);
    for (let i = 0; i < data.length / 4; i++) data[i * 4] = i;
    const crop = cropPixels({data, width}, {x: 1, y: 1, w: 2, h: 2});
    expect(crop.width).toBe(2);
    expect(crop.height).toBe(2);
    expect([crop.data[0], crop.data[4], crop.data[8], crop.data[12]]).toEqual([
      5, 6, 9, 10,
    ]);
  });
});

describe('ranking', () => {
  const vectors = new Map([
    ['cup_1', {vector: normalize(Float32Array.from([1, 0, 0])), label: 'cup'}],
    ['tv_2', {vector: normalize(Float32Array.from([0, 1, 0])), label: 'tv'}],
    [
      'couch_3',
      {vector: normalize(Float32Array.from([1, 1, 0])), label: 'couch'},
    ],
  ]);

  it('ranks nodes by cosine similarity', () => {
    const ranked = rankNodes(Float32Array.from([1, 0, 0]), vectors);
    expect(ranked.map((r) => r.id)).toEqual(['cup_1', 'couch_3', 'tv_2']);
    expect(ranked[0].score).toBeCloseTo(1);
    expect(ranked[1].score).toBeCloseTo(Math.SQRT1_2);
  });

  it('matches only above the threshold and reports the margin', () => {
    const ranked = [
      {id: 'cup_1', score: 0.68},
      {id: 'bed_2', score: 0.62},
    ];
    const match = pickMatch(ranked, DEFAULT_MATCH_SCORE);
    expect(match?.id).toBe('cup_1');
    expect(match?.margin).toBeCloseTo(0.06);
    expect(pickMatch([{id: 'couch_3', score: 0.6}])).toBeNull();
    expect(pickMatch([])).toBeNull();
  });

  it('formats documents and queries with the model prefixes', () => {
    expect(objectDocument('cup')).toBe('title: cup | text: <|image|>');
    expect(textQuery('where is the cup')).toBe(
      'task: search result | query: where is the cup'
    );
  });
});

/** A stand-in for @huggingface/transformers that records every call. */
function fakeLibrary(embedFor: (inputs: unknown[]) => number[]) {
  const calls: unknown[][] = [];
  let active = 0;
  let overlapped = false;
  const processor = Object.assign(
    vi.fn(async (...inputs: unknown[]) => {
      calls.push(inputs);
      return {inputs};
    }),
    {image_processor: {max_soft_tokens: 280}}
  );
  const model = async ({inputs}: {inputs: unknown[]}) => {
    active += 1;
    if (active > 1) overlapped = true;
    await new Promise((resolve) => setTimeout(resolve, 1));
    active -= 1;
    return {sentence_embedding: {data: Float32Array.from(embedFor(inputs))}};
  };
  class RawImage {
    constructor(
      public data: Uint8ClampedArray,
      public width: number,
      public height: number,
      public channels: number
    ) {}
    rgb() {
      return this;
    }
  }
  const from_pretrained = vi.fn(async () => model);
  return {
    calls,
    processor,
    get overlapped() {
      return overlapped;
    },
    from_pretrained,
    library: {
      AutoProcessor: {from_pretrained: async () => processor},
      AutoModel: {from_pretrained},
      RawImage,
    },
  };
}

const image = {data: new Uint8ClampedArray(8 * 8 * 4), width: 8, height: 8};

describe('ObjectFinder', () => {
  it('loads on wasm without WebGPU and lowers the image token budget', async () => {
    const fake = fakeLibrary(() => [1, 0]);
    const finder = new ObjectFinder({loadLibrary: async () => fake.library});
    await finder.load();
    expect(finder.ready).toBe(true);
    expect(finder.backend).toBe('wasm');
    expect(fake.processor.image_processor.max_soft_tokens).toBe(70);
    expect(fake.from_pretrained).toHaveBeenCalledWith(
      'onnx-community/embeddinggemma-2-ONNX',
      expect.objectContaining({device: 'wasm', dtype: 'q4'})
    );
  });

  it('embeds objects as label plus crop and averages repeat views', async () => {
    let next = [1, 0];
    const fake = fakeLibrary(() => next);
    const finder = new ObjectFinder({loadLibrary: async () => fake.library});
    await finder.load();
    await finder.addObservation('cup_1', 'cup', image, [0, 0, 0.5, 0.5]);
    const [text, images] = fake.calls[0] as [string[], unknown[][]];
    expect(text).toEqual(['title: cup | text: <|image|>']);
    expect(images[0][0]).toMatchObject({width: 5, height: 5, channels: 4});

    next = [0, 1];
    await finder.addObservation('cup_1', 'cup', image, [0, 0, 0.5, 0.5]);
    const entry = finder.vectors.get('cup_1')!;
    expect(entry.views).toBe(2);
    expect(entry.vector[0]).toBeCloseTo(Math.SQRT1_2);
    expect(entry.vector[1]).toBeCloseTo(Math.SQRT1_2);
  });

  it('embeds speech at 16 kHz with no text and matches the closest node', async () => {
    const fake = fakeLibrary((inputs) => (inputs[2] ? [0.8, 0.6] : [1, 0]));
    const finder = new ObjectFinder({loadLibrary: async () => fake.library});
    await finder.load();
    await finder.addObservation('cup_1', 'cup', image, [0, 0, 1, 1]);
    const speech = new Float32Array(48000 * 2);
    speech.set(tone(300, 48000, 1), 24000);
    const query = await finder.embedSpeech(speech, 48000);
    const [text, images, audio] = fake.calls[1] as [null, null, Float32Array[]];
    expect(text).toBeNull();
    expect(images).toBeNull();
    expect(audio[0].length / 16000).toBeCloseTo(1.3, 1);
    expect(query?.seconds).toBeCloseTo(1.3, 1);
    expect(finder.match(query!.vector)).toMatchObject({id: 'cup_1'});
  });

  it('skips the model for silent recordings', async () => {
    const fake = fakeLibrary(() => [1, 0]);
    const finder = new ObjectFinder({loadLibrary: async () => fake.library});
    await finder.load();
    expect(await finder.embedSpeech(new Float32Array(48000), 48000)).toBeNull();
    expect(fake.calls).toHaveLength(0);
  });

  it('never runs two model calls at once', async () => {
    const fake = fakeLibrary(() => [1, 0]);
    const finder = new ObjectFinder({loadLibrary: async () => fake.library});
    await finder.load();
    await Promise.all([
      finder.addObservation('a', 'cup', image, [0, 0, 1, 1]),
      finder.addObservation('b', 'tv', image, [0, 0, 1, 1]),
      finder.embedText('where is the cup'),
    ]);
    expect(fake.overlapped).toBe(false);
    expect(fake.calls).toHaveLength(3);
  });
});
