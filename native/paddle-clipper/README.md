# PaddleOCR polygon offset

A small C ABI wraps Clipper 6.4.2 for PP-OCRv5 DB postprocessing. Build `build.cmd` from an x64 Visual Studio Developer Command Prompt. The adapter uses static C/C++ runtime linkage and catches C++ exceptions before they cross the ABI.

Clipper sources retain their original Boost license and copyright notices. The adapter is MIT licensed. The distributed Pascal loader pins the released DLL by SHA-256; a custom rebuild requires updating its explicit pin.
