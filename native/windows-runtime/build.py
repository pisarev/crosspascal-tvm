# Copyright © 2026 Yuriy Pisarev. MIT license; see ../../LICENSE.
"""Build the patched TVM core from checksum-verified upstream archives."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from urllib.request import urlopen


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        roots = {PurePosixPath(member.name).parts[0] for member in bundle.getmembers()}
        if len(roots) != 1:
            raise ValueError("Expected one source archive root")
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or any(":" in part for part in name.parts)):
                raise ValueError("Unsafe archive member")
            parts = name.parts[1:]
            if not parts or parts[0].startswith("."):
                continue
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError("Unexpected source link or special file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="New, empty build directory")
    parser.add_argument("--archive-cache", type=Path, help="Optional directory containing the pinned archives")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    recipe = Path(__file__).resolve().parent
    spec = json.loads((recipe / "SOURCES.json").read_text(encoding="utf-8"))
    patch = recipe / spec["patch"]["file"]
    if digest(patch) != spec["patch"]["sha256"]:
        raise ValueError("Runtime patch checksum differs")
    args.work.mkdir(parents=True, exist_ok=False)
    archives = args.work / "archives"
    archives.mkdir()
    source_root = args.work / "tvm"
    for entry in spec["archives"]:
        archive = archives / entry["file"]
        cached = args.archive_cache / entry["file"] if args.archive_cache else None
        if cached and cached.is_file():
            if digest(cached) != entry["sha256"]:
                raise ValueError("Cached archive checksum differs: " + entry["file"])
            shutil.copyfile(cached, archive)
        else:
            with urlopen(entry["url"], timeout=60) as source, archive.open("xb") as output:
                shutil.copyfileobj(source, output)
        if digest(archive) != entry["sha256"]:
            raise ValueError("Archive checksum differs: " + entry["file"])
        extract(archive, source_root / entry["destination"])
    changed = source_root / "src/runtime/thread_pool.cc"
    if digest(changed) != spec["unmodified_thread_pool_sha256"]:
        raise ValueError("Unexpected upstream thread-pool source")
    subprocess.run(["git", "-C", str(source_root), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(source_root), "apply", str(patch)], check=True)
    changed.write_bytes(changed.read_bytes().replace(b"\r\n", b"\n"))
    if digest(changed) != spec["modified_thread_pool_sha256"]:
        raise ValueError("Patched source checksum differs")
    record = {"source_manifest": spec, "prepared_source": str(source_root), "compiled": False}
    if not args.prepare_only:
        environment = dict(os.environ, VSLANG="1033")
        build = args.work / "build"
        configure = [args.cmake, "-S", str(source_root), "-B", str(build), "-G", "Ninja",
                     "-DCMAKE_MAKE_PROGRAM=" + args.ninja, "-DCMAKE_BUILD_TYPE=Release",
                     "-DUSE_CUDA=ON", "-DUSE_LLVM=OFF", "-DUSE_GTEST=OFF",
                     "-DTVM_FFI_USE_LIBBACKTRACE=OFF"]
        compile_command = [args.cmake, "--build", str(build), "--target", "tvm_runtime",
                           "--parallel", str(args.jobs)]
        for command, name in [(configure, "configure.log"), (compile_command, "build.log")]:
            with (args.work / name).open("xb") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        core = build / "lib/tvm_runtime.dll"
        record.update(compiled=True, core_bytes=core.stat().st_size, core_sha256=digest(core),
                      configure=configure, compile_command=compile_command,
                      supplied_core_byte_match=digest(core) == spec["supplied_core_sha256"])
    with (args.work / "BUILD.json").open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2)
    print(json.dumps({key: value for key, value in record.items() if key != "source_manifest"}, indent=2))


if __name__ == "__main__":
    main()
