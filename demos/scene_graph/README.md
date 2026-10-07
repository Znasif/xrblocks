# Scene graph

Builds a real-world scene graph on device: an object detector finds objects
in the camera frame, the [RelateAnything](https://github.com/Maelic/RelateAnything)
relation head predicts scored `(subject, predicate, object)` triplets between
them, the depth mesh grounds every node to a 3D point, and scans are merged as
you move so the graph persists across viewpoints. Nodes and edges are drawn in
the room and listed in the panel; the graph is the structure a voice tool
router queries ("what is on the table?", "the cup next to the laptop").

Run `npm run dev` from the repository root and open
`http://127.0.0.1:8080/demos/scene_graph/` in Chrome. Press **Enter
Simulator** and walk the living room, or run it on a headset with camera
access. **scan** runs one pass; **auto** re-scans whenever you move or turn.

## How it works

1. `xb.core.deviceCamera.captureSnapshot()` freezes one frame, on the regular
   video path and on WebXR raw camera access alike.
2. `xb.core.world.objects.runDetection({snapshot})` runs the configured
   detector on that frame. The default is the on-device MediaPipe
   EfficientDet-Lite2 backend (80 COCO classes); `?detector=gemini` uses the
   Gemini backend instead.
3. `RelationDetector` resizes the frame to 448 px, packs the normalized boxes,
   and runs the relation ONNX graph through ONNX Runtime Web on WebGPU, falling
   back to WASM. The graph emits raw logits; the calibrated score
   `sigmoid(a·(pred + w·pair) + b)` and thresholds are applied here, so the
   threshold (`?thr=0.4`) and the predicate vocabulary can change per frame.
4. Each box centre is raycast against the depth mesh for a world point, and
   `SceneGraph` merges nodes by label and distance and decays edges that stop
   being observed.

Object labels are never an input to the relation model, so any box source
works: a different detector, WebXR planes, or class-free segments.

## Ask about an object

Hold **hold to ask** (or pinch and hold anywhere while in XR), say what you are
looking for, and release: "where is the cup?", "find my coffee mug", "where
can I sit down?". The closest object in the graph gets a pulsing ring, a
beacon and a highlighted label, its relations stay bright while the rest fade,
and the panel answers with the node's relations ("cup: resting on the table,
to the left of the book"). You can also type a question.

This runs on device with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2),
which maps text, images and audio into one 768-d space:

1. Every detection is embedded under its graph node as its label interleaved
   with its image crop (`title: cup | text: <|image|>`). Repeat views of a
   node are averaged.
2. The spoken question is embedded directly by the audio encoder. There is no
   speech-to-text step: the microphone audio is resampled to 16 kHz, trimmed
   to the speech, and compared with the node embeddings by cosine similarity.
3. The best node is highlighted when it scores at least `0.625` (`?match=`).

The image half matters: on the simulator living room the detector labels the
coffee table "bed", and "where is the table?" still finds it.

`ObjectFinder` defaults to the `q4` weights (about 470 MB, cached by the
browser after the first visit) and needs WebGPU. Every quantized variant of the
text and vision graphs uses the `GatherBlockQuantized` operator, which ONNX
Runtime Web's WASM backend does not implement.

Measured on the simulator living room, q4, with synthetic speech:

| spoken query               | highlighted   | score |
| -------------------------- | ------------- | ----- |
| where is the cup?          | cup           | 0.68  |
| find my coffee mug         | cup           | 0.65  |
| where is the television?   | tv            | 0.68  |
| where can I sit down?      | chair         | 0.67  |
| where is the table?        | table ("bed") | 0.66  |
| where is the refrigerator? | none          | 0.61  |
| find my car keys           | none          | 0.59  |

## Query parameters

| param      | default                 | meaning                                                                        |
| ---------- | ----------------------- | ------------------------------------------------------------------------------ |
| `model`    | authors' web export     | URL of the relation ONNX graph (`.data` chunks via `manifest.json` next to it) |
| `bank`     | authors' predicate bank | URL of `predicate_bank.json` (embeddings, gates, thresholds, spatial flags)    |
| `ort`      | jsDelivr                | ONNX Runtime Web `dist/` URL for the WASM files                                |
| `backend`  | `webgpu`                | `wasm` forces the WASM execution provider                                      |
| `thr`      | `0.4`                   | global relation-score threshold                                                |
| `bankthr`  | off                     | use the bank's calibrated per-predicate thresholds instead                     |
| `det`      | `0.3`                   | MediaPipe detector score threshold                                             |
| `auto`     | `1`                     | `0` disables movement-triggered rescans                                        |
| `voice`    | `1`                     | `0` turns off voice search and does not load EmbeddingGemma 2                  |
| `egdtype`  | `q4`                    | EmbeddingGemma 2 weights: `q4`, `q4f16`, `q8`, `fp16` or `fp32`                |
| `match`    | `0.625`                 | minimum cosine similarity for a voice or text query to highlight a node        |
| `headless` | off                     | automation mode: autostart the simulator, hide its UI                          |

Multi-threaded WASM needs the page to be cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`); without those headers ONNX
Runtime runs single-threaded.

## Credits and licenses

- RelateAnything by Maëlic Neau, code Apache-2.0,
  https://github.com/Maelic/RelateAnything. The released weights are
  derivatives of Meta's DINOv3 and carry the
  [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/);
  the app must show "Built with DINOv3". Nothing from the model is bundled or
  checked into this repository; it is downloaded at runtime.
- EmbeddingGemma 2 by Google DeepMind, Apache-2.0,
  https://huggingface.co/google/embeddinggemma-2; use is subject to the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
  The ONNX export is `onnx-community/embeddinggemma-2-ONNX`, run with
  [Transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0)
  and downloaded at runtime.
- ONNX Runtime Web is MIT, https://github.com/microsoft/onnxruntime.
- MediaPipe EfficientDet-Lite2 is Apache-2.0.
