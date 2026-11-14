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
| `pascal/CrossVision.ML.pas`, `pascal/CrossVision.ML.CpuVision.pas` | Common result types and the explicit CPU vision consumer |
| `pascal/TVM.Yolo11.pas`, `pascal/TVM.RtDetr.pas`, `pascal/CrossVision.Float16.pas` | Existing model decoders and their small float16 dependency |
| `pascal/TVM.BiRefNet.pas` | Full HR800 SwinL CPU/CUDA loading, image normalization, mask resizing, and transparent cutouts |
| `pascal/TVM.WindowsPaths.pas` | Windows Unicode runtime paths and temporary model-directory aliases |
| `runtime/windows-x64-cuda/` | Runtime destination, five tokenizer files, model source pins and licences |
| `recipes/TVMBuildOptions.txt` | Historical CPU build-options template; it is separate from the supplied CUDA kit |
| `docs/PASCAL-API-NOTES.md` | Ownership, tensor layout and runtime-lifetime requirements |

Use the four native runtime DLLs supplied with the Windows x64 SDK. Keep them together. The runtime requires the Microsoft Visual C++ x64 runtime. CUDA also requires a compatible NVIDIA driver.

The retained FFI package version and its C ABI version are different identifiers: the pinned DLL reports C ABI `0.1.14`. The consumers check both the exact DLL bytes and that ABI tuple. CPU vision loads the first three runtime DLLs and does not initialize CUDA.

## Build the Pascal example manually

Extract `crosspascal`, `crosspascal-tvm` and `crosspascal-opencv` as sibling directories. In `crosspascal/examples/qwen-chat`, open `QwenChat.dproj` in Delphi or `QwenChat.lpi` in Lazarus and build for Windows x64. Follow the [example README](https://github.com/pisarev/crosspascal/tree/main/examples/qwen-chat) to select a converted model bundle and run a prompt.

`TVM.pas` currently exposes image helpers through `OpenCVTypes`, `OpenCVConsts` and `CrossVision.Mat`. Their complete Pascal sources are supplied by the sibling OpenCV repository. The text-only Qwen example does not initialize OpenCV or call its image operations. Leptonica is not a dependency of this text route.

For image classification and detection, open `crosspascal/examples/vision-models/VisionModels.dproj` in Delphi or `VisionModels.lpi` in Lazarus and build for Windows x64. That project also uses the main repository's geometry types and initializes the locally built OpenCV runtime. Its [README](https://github.com/pisarev/crosspascal/tree/main/examples/vision-models) describes the five profiles, external model-bundle requirements, preprocessing and source-provenance limits.

For background removal, open `crosspascal/examples/birefnet/BiRefNet.dproj` or `BiRefNet.lpi`. The [BiRefNet example](https://github.com/pisarev/crosspascal/tree/main/examples/birefnet) uses OpenCV to decode the image and save the original-size mask or transparent PNG. `TBiRefNetModel` checks the pinned source, tensor contract, model DLL hash, and selected CPU/CUDA device. Its 800-pixel profile differs from the upstream 2048-pixel workflow.

## Model and runtime checks

`QwenCuda.pas` validates the declared profile and immutable model revision, all 170 parameter entries, parameter length and SHA-256, model DLL SHA-256, the exact tokenizer files and the four runtime DLLs. It holds input files open against writes during generation. It checks GPU 0 and compute capability before model loading, then checks CUDA tensor shapes and finite logits. The memory guard checks a minimum reserve; allocation can still fail if other applications consume memory.

The runtime binaries are selected, pinned artifacts. Their presence does not claim that the TVM compiler has been rebuilt reproducibly or that another GPU, precision, context length or model has been tested. Hardware execution and delivery evidence belong to the corresponding [CrossPascal release notes](https://github.com/pisarev/crosspascal/releases). A generated result records the actual device, input hashes and token IDs; it does not declare a release accepted.

## Licences

Original Pascal code and repository documentation use the root [MIT licence](LICENSE). Upstream TVM licences and notices are in `licenses/`. The [TVM FFI 0.1.14 licence](runtime/windows-x64-cuda/LICENSE-tvm-ffi-0.1.14.txt) and [notice](runtime/windows-x64-cuda/NOTICE-tvm-ffi-0.1.14.txt) are in the runtime kit. The Qwen model and tokenizer retain their official Apache 2.0 licence in `runtime/windows-x64-cuda/LICENSE-Qwen2.5-0.5B-Instruct.txt`; `MODEL-SOURCE.json` identifies the immutable upstream source and all seven source-file pins. See [runtime provenance](runtime/RUNTIME-PROVENANCE-windows-x64-cuda.json). The provenance report describes the files in the runtime kit; it is stored beside that directory and is not an input to the model-bundle copier.

### Windows TVM FFI

The Pascal examples and the converter use the same official TVM FFI 0.1.14 DLL. The Windows C ABI is 0.1.14. Apache TVM FFI includes the [Windows DbgHelp fix](https://github.com/apache/tvm-ffi/pull/754). The installer verifies the upstream wheel checksum and the DLL checksum before installation. Keep the upstream Apache licence and notice with the runtime.

`pascal/DLPack.pas` is a Pascal adaptation of the DLPack 1.3 interface. The original C declarations are copyright DLPack contributors (2017 onward); see the [DLPack Apache 2.0 license](licenses/dlpack-1.3-LICENSE.txt).
