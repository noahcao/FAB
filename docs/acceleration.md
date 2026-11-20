# TensorRT engines and speed

`fast=True` (the default of the `fab` command) compiles the body and replays it from CUDA graphs. With TensorRT
engines built on your GPU, the body and the person detector also run through TensorRT; the paper's timings use them.

```bash
pip install -e ".[engine]"       # the TensorRT / ONNX versions behind the released timings
pip uninstall -y opencv-python && pip install --force-reinstall --no-deps "opencv-python-headless>=4.9"
fab-build-trt --images assets/demo/                                # FAB-L: body at batch 1-4 + detector
fab-build-trt --images assets/demo/ --model fab-s --no_detector
```

The second line restores the headless OpenCV that the `engine` extra replaces ([troubleshooting.md](troubleshooting.md)).
A build takes about 10 min per body engine (about 45 min for FAB-L's four engines and the detector); finished steps
are kept if it stops.

| option | default | effect |
|---|---|---|
| `--images DIR` (or `$FAB_ENGINE_IMAGES`) | required | a few photos with people; the export checks run on them |
| `--model` | `fab-l` | `fab-l` or `fab-s` (FAB-H runs compiled PyTorch only) |
| `--batches` | `"2 3 4"` | extra engines for frames with several people |
| `--mix` | 1 | the 2.5D lift mix of the released setting |
| `--no_detector` | off | skip the detector engine |

## What it builds and checks

Engines go to `$FAB_HOME/engines/<gpu>/`: `<model>/body-m1.engine` (one person), `body-m1-b2` ... `-b4` (2-4 people)
and `detector.engine` (FP16), with `build.log` and the export checks. The build refuses an export that disagrees with
PyTorch on real crops, and keeps the detector engine only if its boxes match the PyTorch detector's on your images.
`FABPredictor` (`body_engine="auto"`, `body_batch="auto"`) uses an engine only if it was built for the weights file in
use, and says which backend it runs (`FABPredictor: body backend ...`). Rebuild after changing the GPU, TensorRT or
the weights.

## Speed

End-to-end fps on an RTX 4070 Ti SUPER (person detection to fitted SMPL, `FABPredictor(<model>, fast=True)`), by the
number of people in the frame:

| model | 1 person | 2 | 3 | 4 |
|---|---|---|---|---|
| FAB-S (TensorRT) | 63.8 | 48.6 | 41.3 | 33.8 |
| FAB-L (TensorRT) | 39.2 | 25.7 | 19.6 | 15.4 |
| FAB-H, MHR mesh → SMPL (compiled PyTorch; the default) | 24.9 | 14.8 | 10.2 | 7.8 |
| FAB-H, dense points → SMPL (compiled PyTorch) | 21.8 | 13.1 | 9.4 | 7.4 |

Each value is the median frame time over 40 frames (20 3DPW validation, 20 COCO val2017), each timed 10 times after a
warm-up on an otherwise idle GPU. One FAB-L predictor with its batch engines fits a 16 GB card. To time your GPU:

```python
import time, cv2, numpy as np, torch
from fab.predictor import FABPredictor

fab = FABPredictor("fab-l", fast=True)
frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in paths]   # frames with the same number of people
for im in frames[:3]:
    fab.predict(im)                                                          # warm-up (compiles the first time)
ts = []
for im in frames:
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fab.predict(im)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
print(f"{1 / np.median(ts):.1f} fps")
```
