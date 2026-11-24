# CrossPascal TVM

This repository supplies the Pascal TVM interface, the Qwen tokenizer and model helpers, CPU and CUDA vision adapters, and a pinned Windows x64 runtime installer. Delphi and Lazarus projects are provided for [Qwen CUDA chat](https://github.com/pisarev/crosspascal/tree/main/examples/qwen-chat) and [vision models](https://github.com/pisarev/crosspascal/tree/main/examples/vision-models).

The retained Qwen2.5-0.5B-Instruct legacy profile uses all 24 layers, FP32, batch size 1, context 64 and prefill chunk 64. Current conversion and chat applications also support separate profiles with larger converted contexts. The converter records the CUDA architecture, and QwenChat checks the running device against it. Generation settings include sampling, Auto answer budgets, a manual token limit and Stop. Loaded models can be reused between requests.

## Contents

| Location | Purpose |
| --- | --- |
| `pascal/TVM.pas`, `pascal/DLPack.pas` | TVM FFI, tensor and model interface |
| `pascal/BPETokenizer.pas`, `pascal/QwenTokenizer.pas` | UTF-8 byte BPE tokenizer with Unicode input |
| `pascal/QwenModel.pas` | Parameter loading, embedding, prefill, decode and KV-cache operations |
| `pascal/QwenCuda.pas` | Validation and execution of the supplied CUDA profile |
| `pascal/Qwen35Native.pas`, `pascal/Qwen35Weights.pas`, `pascal/Qwen35Session.pas` | Native dense Qwen 3.5/3.8 text decoder with layer streaming |
| `pascal/Qwen35PackedCuda.pas`, `pascal/Qwen35CudaBuffers.pas` | Original GGUF tensor decoding, bounded source cache and optional decoded CUDA buffers |
| `pascal/CrossVision.ML.pas`, `pascal/CrossVision.ML.CpuVision.pas` | Common result types and the explicit CPU vision consumer |
| `pascal/TVM.Yolo11.pas`, `pascal/TVM.RtDetr.pas`, `pascal/CrossVision.Float16.pas` | Existing model decoders and their small float16 dependency |
| `pascal/TVM.BiRefNet.pas` | Full HR800 SwinL CPU/CUDA loading, image normalization, mask resizing, and transparent cutouts |
| `pascal/TVM.OmDet.pas` | OmDet image preprocessing and class-aware detection postprocessing |
| `pascal/TVM.OmDetModel.pas` | Native CLIP and Swin-Tiny execution, learned class projection, and text-feature caching |
| `pascal/ClipTokenizer.pas` | Unicode CLIP byte-BPE tokenization for editable class and task text |
| `pascal/TVM.WindowsPaths.pas` | Windows Unicode runtime paths and temporary model-directory aliases |
| `pascal/TVM.Threads.pas` | Explicit shutdown of a short-lived inference thread's native pool |
| `native/windows-runtime/` | Pinned source archives, Windows pool-lifecycle patch and core build recipe |
| `runtime/windows-x64-cuda/` | Runtime destination, five tokenizer files, model source pins and licences |
| `recipes/TVMBuildOptions.txt` | Historical CPU build-options template; it is separate from the supplied CUDA kit |
| `docs/PASCAL-API-NOTES.md` | Ownership, tensor layout and runtime-lifetime requirements |

Use the four native runtime DLLs supplied with the Windows x64 SDK. Keep them together. The runtime requires the Microsoft Visual C++ x64 runtime. CUDA also requires a compatible NVIDIA driver.

The SDK core includes the [Windows host-thread shutdown correction](native/windows-runtime/README.md). An unmodified upstream core does not provide that entry and is rejected before inference. The runtime installer verifies upstream companion DLLs and uses the matching SDK core, selected with `--runtime-core` when installing from a source checkout. It copies the runtime licenses and notices into the destination. Python compiler packages retain their upstream binaries.

The retained FFI package version and its C ABI version are different identifiers: the pinned DLL reports C ABI `0.1.14`. The consumers check both the exact DLL bytes and that ABI tuple. CPU vision loads the first three runtime DLLs and does not initialize CUDA.

## Build the Pascal example manually

Extract `crosspascal`, `crosspascal-tvm` and `crosspascal-opencv` as sibling directories. In `crosspascal/examples/qwen-chat`, open `QwenChat.dproj` in Delphi or `QwenChat.lpi` in Lazarus and build for Windows x64. Follow the [example README](https://github.com/pisarev/crosspascal/tree/main/examples/qwen-chat) to select a converted model bundle and run a prompt.

`TVM.pas` currently exposes image helpers through `OpenCVTypes`, `OpenCVConsts` and `CrossVision.Mat`. Their complete Pascal sources are supplied by the sibling OpenCV repository. The text-only Qwen example does not initialize OpenCV or call its image operations. Leptonica is not a dependency of this text route.

For image classification and detection, open `crosspascal/examples/vision-models/VisionModels.dproj` in Delphi or `VisionModels.lpi` in Lazarus and build for Windows x64. That project also uses the main repository's geometry types and initializes the locally built OpenCV runtime. Its [README](https://github.com/pisarev/crosspascal/tree/main/examples/vision-models) describes the five profiles, external model-bundle requirements, preprocessing and source-provenance limits.

For background removal, open `crosspascal/examples/birefnet/BiRefNet.dproj` or `BiRefNet.lpi`. The [BiRefNet example](https://github.com/pisarev/crosspascal/tree/main/examples/birefnet) uses OpenCV to decode the image and save the original-size mask or transparent PNG. `TBiRefNetModel` checks the pinned source, tensor contract, model DLL hash, and selected CPU/CUDA device. Its 800-pixel profile differs from the upstream 2048-pixel workflow.

After the last inference, call `TBiRefNetModel.Close` before reporting success, then free the model in a `finally` block. If `BiRefNetRequiresProcessRestart` is true, restart the application before loading or running another BiRefNet model. `BiRefNetCleanupDiagnostic` records secondary cleanup errors while the original operation exception is preserved. The destructor is a fallback; unsafe CUDA resources can remain held until process termination.

For text-guided object detection, open `crosspascal/examples/omdet-turbo/OmDetTurbo.dproj` or `OmDetTurbo.lpi`. The [OmDet-Turbo example](https://github.com/pisarev/crosspascal/tree/main/examples/omdet-turbo) uses `TOmDetModel` to run the CLIP text encoder and Swin-Tiny detector on CPU or CUDA. It accepts up to 16 class names and a task instruction, then saves boxes, scores, and an annotated image. Bundle loading checks all five artifacts and their hashes.

For YOLO, open `crosspascal/examples/yolo-models/YoloModels.dproj` or `YoloModels.lpi`. The [YoloModels example](https://github.com/pisarev/crosspascal/tree/main/examples/yolo-models) runs 23 profiles across seven tasks on CPU or CUDA. `LoadYoloBundle` in `CrossVision.ML.YoloBundle` takes the expected profile and manifest pin; the application then opens the matching typed interface. `TYoloRunOptions` in `TVM.Yolo.RunOptions` sets box thresholds and a stop callback. See the [bundle and API notes](https://github.com/pisarev/crosspascal-convert/blob/main/docs/YOLO-NATIVE.md).

## Model and runtime checks

`QwenCuda.pas` validates the declared profile and immutable model revision, all 170 parameter entries, parameter length and SHA-256, model DLL SHA-256, the exact tokenizer files and the four runtime DLLs. It holds input files open against writes during generation. It checks GPU 0 and compute capability before model loading, then checks CUDA tensor shapes and finite logits. The memory guard checks a minimum reserve; allocation can still fail if other applications consume memory.

The [CrossPascal release notes](https://github.com/pisarev/crosspascal/releases) list the coordinated SDK, model, and runtime downloads. Generated model results record the actual device, input hashes, and token IDs.

## Qwen decoded CUDA buffers

A Qwen GGUF CUDA manifest can set `cuda_decoded_buffer_pool` to the JSON
boolean `true` to reuse decoded parameter storage between completed layers
and output-head tiles. The field is optional and disabled when absent or
false. The converter does not enable it by default. CPU and BF16 models
cannot enable this path.

Each simultaneously live parameter receives a distinct slot. CUDA completes
the current group before its storage is reused. Slots are sized from the
actual manifest, allocated lazily and retained until model unload. The packed
source cache reserves space for those buffers before choosing its budget.
Source bytes, tensor-decoding kernels and model graphs are unchanged.

Generation reports include `decoded_cuda_buffers`: whether the pool is
enabled, planned and allocated bytes, storage allocations, rentals, and groups.
Counters accumulate over the model lifetime.

## Licences

Original Pascal code and repository documentation use the root [MIT licence](LICENSE). Upstream TVM licences and notices are in `licenses/`. The [TVM FFI 0.1.14 licence](runtime/windows-x64-cuda/LICENSE-tvm-ffi-0.1.14.txt) and [notice](runtime/windows-x64-cuda/NOTICE-tvm-ffi-0.1.14.txt) are in the runtime kit. The Qwen model and tokenizer retain their official Apache 2.0 licence in `runtime/windows-x64-cuda/LICENSE-Qwen2.5-0.5B-Instruct.txt`; `MODEL-SOURCE.json` identifies the immutable upstream source and all seven source-file pins. See [runtime provenance](runtime/RUNTIME-PROVENANCE-windows-x64-cuda.json). The provenance report describes the files in the runtime kit; it is stored beside that directory and is not an input to the model-bundle copier.

### Windows TVM FFI

The Pascal examples and the converter use the same official TVM FFI 0.1.14 DLL. The Windows C ABI is 0.1.14. Apache TVM FFI includes the [Windows DbgHelp fix](https://github.com/apache/tvm-ffi/pull/754). The installer verifies the upstream wheel checksum and the DLL checksum before installation. Keep the upstream Apache licence and notice with the runtime.

The installer also preserves all licence and notice members supplied by the
two pinned upstream wheels. Their exact origins and hashes are recorded in
[`licenses/upstream-windows-0.26.0/SOURCE.json`](licenses/upstream-windows-0.26.0/SOURCE.json).
Each native installation includes that licence directory and
`TVM-PROVENANCE.json`, which lists the installed native libraries and metadata.
The native core is built from modified TVM source and provides explicit
host-thread pool shutdown. The companion DLLs retain their upstream bytes.
The installed `sources/windows-runtime` directory contains the source pins,
patch and build recipe. No byte-for-byte reproducible source build is claimed.

The default native installation contains the four consumer DLLs. Add
`--native-compiler` when preparing a native conversion toolchain to include
`tvm_compiler.dll` and `tvm_ffi.lib` as well. This option does not install Python
packages. Supply the matching SDK core with `--runtime-core`; the default path
is `runtime/windows-x64-cuda/tvm_runtime.dll` in this repository. For example,
with the pinned wheels already in `wheel-cache`:

```powershell
python scripts/install_tvm_windows.py --offline --cache wheel-cache --destination native-tvm --runtime-core C:\CrossPascal\tvm_runtime.dll --native-compiler
```

The installer accepts an existing destination only when its corresponding
files are identical. It rejects a different DLL or licence instead of replacing
it. Model-specific libraries, such as the SentencePiece wrapper, retain their
own source and licence records and must be added separately when needed.
When `--site-packages` is supplied, the Python compiler packages retain their
upstream binaries; the patched native core is installed only in the selected
native destination.

`pascal/DLPack.pas` is a Pascal adaptation of the DLPack 1.3 interface. The original C declarations are copyright DLPack contributors (2017 onward); see the [DLPack Apache 2.0 license](licenses/dlpack-1.3-LICENSE.txt).
