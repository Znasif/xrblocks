import {describe, expect, it, vi} from 'vitest';

vi.mock('onnxruntime-web', () => ({
  Tensor: class {
    constructor(
      public type: string,
      public data: ArrayLike<number> | BigInt64Array,
      public dims: number[]
    ) {}
  },
  InferenceSession: {create: vi.fn()},
  env: {wasm: {}},
}));

import {RelationDetector, externalDataFiles} from './RelationDetector.js';

const logit = (p: number) => Math.log(p / (1 - p));

function bank(names: string[]) {
  return {
    names,
    dim: 4,
    W: names.map((_, i) => [i, 1, 0, 0]),
    alpha: names.map(() => 0.5),
    thr: names.map(() => 0.95),
    is_spatial: names.map((n) => (n.includes('left') ? 1 : 0)),
    calibration: {a: 1, b: 0},
    img_size: 448,
    max_boxes: 32,
  };
}

function detectorWith(names: string[], opts = {}) {
  const detector = new RelationDetector({predicates: names, ...opts});
  detector.bank = bank(names);
  detector.setVocabulary(names);
  return detector;
}

/** Builds graph outputs for K pairs; `cells[k][v]` are predicate probabilities. */
function outputs(pairs: [number, number][], cells: number[][], pairProb = 0.5) {
  const K = pairs.length;
  const V = cells[0].length;
  return {
    pred_logits: {
      dims: [1, K, V],
      data: Float32Array.from(cells.flat().map(logit)),
    },
    pair_logits: {
      dims: [1, K],
      data: Float32Array.from(pairs.map(() => logit(pairProb))),
    },
    sub_idx: {data: BigInt64Array.from(pairs.map(([s]) => BigInt(s)))},
    obj_idx: {data: BigInt64Array.from(pairs.map(([, o]) => BigInt(o)))},
    valid_mask: {data: Uint8Array.from(pairs.map(() => 1))},
  };
}

describe('RelationDetector.setVocabulary', () => {
  it('activates a subset of the bank in the requested order', () => {
    const d = detectorWith(['on', 'holding', 'to the left of']);
    const names = d.setVocabulary(['holding', 'on']);
    expect(names).toEqual(['holding', 'on']);
    expect(d.vocab!.W.dims).toEqual([2, 4]);
    expect(Array.from(d.vocab!.W.data as Float32Array)).toEqual([
      1, 1, 0, 0, 0, 1, 0, 0,
    ]);
    expect(d.vocab!.spatial).toEqual([false, false]);
  });

  it('skips predicates the bank does not know', () => {
    const d = detectorWith(['on']);
    expect(d.setVocabulary(['on', 'juggling'])).toEqual(['on']);
  });
});

describe('RelationDetector.decode_', () => {
  it('keeps the best predicate per pair above threshold and ranks by score', () => {
    const d = detectorWith(['on', 'holding'], {threshold: 0.4});
    // pair weight 1 with pair prob 0.5 adds logit(0.5) = 0, so scores equal the cell probabilities.
    const out = outputs(
      [
        [0, 1],
        [1, 0],
        [0, 2],
      ],
      [
        [0.9, 0.2],
        [0.3, 0.3],
        [0.1, 0.7],
      ]
    );
    const triplets = d.decode_(out, 3, null, ['cup', 'table', 'hand']);
    expect(
      triplets.map((t) => `${t.subjectLabel} ${t.predicate} ${t.objectLabel}`)
    ).toEqual(['cup on table', 'cup holding hand']);
    expect(triplets[0].score).toBeCloseTo(0.9, 5);
  });

  it('drops self pairs, padded pairs and invalid slots', () => {
    const d = detectorWith(['on'], {threshold: 0.1});
    const out = outputs(
      [
        [0, 0],
        [0, 5],
        [0, 1],
      ],
      [[0.9], [0.9], [0.9]]
    );
    out.valid_mask.data[2] = 0;
    expect(d.decode_(out, 2, null, null)).toEqual([]);
  });

  it('applies calibration and the pair-existence logit', () => {
    const d = detectorWith(['on'], {threshold: 0});
    d.calib = {a: 0.5651, b: -1.9623};
    const out = outputs([[0, 1]], [[0.99]], 0.99);
    const [t] = d.decode_(out, 2, null, null);
    const expected =
      1 / (1 + Math.exp(-(0.5651 * (logit(0.99) + logit(0.99)) - 1.9623)));
    expect(t.score).toBeCloseTo(expected, 5);
  });

  it('uses the bank thresholds when asked', () => {
    const d = detectorWith(['on'], {
      threshold: 0.1,
      usePerPredicateThresholds: true,
    });
    const out = outputs([[0, 1]], [[0.9]]);
    expect(d.decode_(out, 2, null, null)).toEqual([]);
  });

  it('ranks by detector confidence without changing the reported score', () => {
    const d = detectorWith(['on'], {threshold: 0.1});
    const out = outputs(
      [
        [0, 1],
        [2, 1],
      ],
      [[0.8], [0.7]]
    );
    const triplets = d.decode_(out, 3, [0.1, 1, 1], null);
    expect(triplets[0].subject).toBe(2);
    expect(triplets[1].score).toBeCloseTo(0.8, 5);
  });
});

describe('externalDataFiles', () => {
  const model = 'relateanything_vits16plus_w16.onnx';

  it("reads the authors' v2 manifest: relation.files[].path, <stem>.dataN", () => {
    const manifest = {
      version: 2,
      relation: {
        files: [
          {path: 'models/relateanything_vits16plus_w16.onnx', size: 1},
          {path: 'models/relateanything_vits16plus_w16.data1', size: 3},
          {path: 'models/relateanything_vits16plus_w16.data0', size: 2},
        ],
      },
      detectors: [{files: [{path: 'models/yolo26n.onnx'}]}],
    };
    expect(externalDataFiles(manifest, model)).toEqual([
      'relateanything_vits16plus_w16.data0',
      'relateanything_vits16plus_w16.data1',
    ]);
  });

  it('accepts a models array with name fields or plain strings', () => {
    const manifest = {
      models: [
        {files: [{name: 'other.onnx'}, {name: 'other.onnx.data0'}]},
        {files: [model, `${model}.data0`]},
      ],
    };
    expect(externalDataFiles(manifest, model)).toEqual([`${model}.data0`]);
  });

  it('returns nothing for a manifest without chunks', () => {
    expect(externalDataFiles({}, model)).toEqual([]);
    expect(
      externalDataFiles({relation: {files: [{path: model}]}}, model)
    ).toEqual([]);
  });
});
