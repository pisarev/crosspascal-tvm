# CrossPascal TVM

This private testing draft of `crosspascal-tvm` contains a CPU build-options template and the original TVM license and NOTICE. The planned project will distribute reproducible recipes and separately accepted native TVM artifacts.

## Current contents

`recipes/TVMBuildOptions.txt` is derived from a recorded CPU configuration. Configure `USE_LLVM` for your own installation:

```text
USE_LLVM /path/to/llvm/bin/llvm-config.exe --link-static
```

The other options preserve the selected historical configuration. This template has not been used for a new TVM build. TVM source, a compiler, runtime DLLs, and compiled models are absent.

## Verify the files

Use Python 3.12 or later from this repository's root:

```powershell
python -B scripts/verify_source.py
```

The inventory permits explicitly declared root Git metadata and checks the remaining files by size and SHA-256.

## License

Original documentation, publication utilities, and configuration choices use the root [MIT license](LICENSE). The unchanged upstream [TVM license](licenses/TVM-LICENSE) and [NOTICE](licenses/TVM-NOTICE) are retained for the TVM material and its original notices. No TVM binary is redistributed here.

The native TVM build and the Pascal runtime/image interface are separate parts of CrossPascal. This repository does not currently provide the Pascal interface.
