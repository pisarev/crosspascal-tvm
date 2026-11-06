# Pascal TVM API notes

The QwenChat and VisionModels examples were built and run on Windows x64 with Delphi 13.2 and Lazarus 4.8 (FPC 3.2.2).

## Tensor and object ownership

DLPack device enums occupy four bytes; `TDLDataType` keeps its byte-sized code and bit fields. Preserve those declarations when consuming the ABI.

A TVM tensor handle owns a runtime object. Release acquired references with `TTVMRuntime.ObjectRelease`, which clears the passed handle. FFI function returns that contain objects transfer a reference to the caller. The Qwen helper transfers a direct tensor result or retains the selected tuple element, then releases the tuple result.

CUDA tensor `data` points to device memory. Use `CopyBytesToTensor` and `CopyBytesFromTensor` for host transfers. Do not dereference CUDA memory as a Pascal array.

## Loading and shutdown

The Qwen project defines `TVM_LAZY_INIT` for every compiled unit. `QwenCuda` validates and loads the selected runtime directory explicitly; it does not use ambient `TVM_RUNTIME_LIB` or PATH to choose the four TVM DLLs.

Release tensors, the KV cache, VM, module and parameter references after a run. The runtime DLLs remain mapped until ordinary process shutdown. The retained runtime may have background threads and thread-local cleanup; calling `TVM.Finalize` to unload its DLLs while those are alive is unsafe. The example returns normally and does not bypass cleanup with `Halt`.

The Qwen consumer permits one run at a time. After a partially failed runtime initialization, restart the application. Restart it before changing to a different runtime directory.

## Parameter and tokenizer input

The Qwen profile uses 170 float32 tensors, in the exact order recorded in `model.json` and the two parameter indexes. It validates the full layout and checks the SHA-256 recorded during conversion. A minimum free-memory guard does not reserve memory against other applications.

Binary input readers use read-only streams. This allows the consumer to hold read locks on the inspected files for the duration of generation.

The Qwen tokenizer accepts Unicode strings, performs its retained NFC handling, and decodes concatenated byte pieces as UTF-8. Keep integer overflow checking disabled: the BPE hash deliberately uses wrapping unsigned multiplication. The tokenizer files are pinned exports for the supplied source revision.

## Qwen CUDA wrapper

`QwenCuda.pas` adds a fixed-profile consumer, input checks, device reporting and a short greedy generation loop around the retained Qwen implementation. It does not change the compiled model's graph, attention arithmetic, parameter values or KV-cache ABI.

The binary readers open input streams read-only. When the model returns a tuple, the helper retains the selected tensor before releasing the tuple reference.

