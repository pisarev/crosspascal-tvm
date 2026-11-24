# Windows TVM core

The supplied `tvm_runtime.dll` adds explicit shutdown of the current host thread's native pool. This prevents a Windows loader-lock deadlock when a Pascal worker exits after CPU inference. It also corrects parallel-task error publication, the caller's task ID, and the serial callback return code. The patch modifies only `src/runtime/thread_pool.cc` from the revision in [SOURCES.json](SOURCES.json). These CrossPascal modifications were made on 14 November 2026; the modified TVM source retains the upstream Apache 2.0 license.

The supplied core is 2,345,984 bytes, SHA-256 `e7580dbaeb87a480fcaac3ffe30bca1b294c0c370c7c02b24228b1e56fdd4584`. Keep the exact FFI0.1.14, extra-runtime and CUDA-runtime DLLs specified in [runtime provenance](../../runtime/RUNTIME-PROVENANCE-windows-x64-cuda.json). EA024 supplies the separately built FFI with the Windows UTF-8 filename fix; its pinned source and patch are in [FFI-SOURCES.json](FFI-SOURCES.json), with the build recipe in `build_ffi.py`. The core-only `build.py` recipe below produces the core and its import library; it does not build that separately patched FFI delivery.

## Build

Use an x64 Native Tools command prompt with Visual Studio 2022, Python 3.11 or later, Git, CMake, Ninja and the CUDA Toolkit. The supplied core was built with MSVC 19.44.35211, CMake 3.30.2, Ninja 1.12.1 and CUDA Toolkit 12.8. LLVM is disabled, and only the runtime target is built.

From this directory, run:

```text
python build.py --work C:\build\crosspascal-tvm-runtime
```

The work directory must be new. Downloads are checked against the three archive hashes before extraction. Hidden repository metadata is omitted. The script checks the original source, applies the pinned patch, normalizes that file to LF, checks the resulting source, configures Release with CUDA enabled, and builds `tvm_runtime` with four jobs. `--archive-cache` can point to a folder containing the three named archives. `--prepare-only` verifies and patches sources without compiling.

The result is `build/lib/tvm_runtime.dll` under the work directory. Logs and a build record are kept there. This is a source rebuild recipe, without a claim of byte-for-byte reproducibility: a different toolchain or build path can produce a different DLL hash. The supplied Pascal consumers require the exact SDK core. Using a different rebuild requires matching provenance and consumer pins, followed by validation of the complete runtime kit.

## Thread ownership

Call `FinishTVMThread` from `TVM.Threads` on the same short-lived host thread that performed inference, after its last native call and before it exits. Keep the runtime mapped while doing so. The supplied GUI workers do this in `finally` on success, Stop and error paths. A repeated shutdown is harmless, and shutdown before using the pool does not create workers.

Do not call shutdown from a TVM pool worker, during an active parallel launch, or from unit/DLL finalization under the Windows loader lock. A failure must be reported with the operation's original error preserved. Shutdown releases that host thread's pool; it does not free a retained model, VM or KV cache, and a later request can create another pool.
