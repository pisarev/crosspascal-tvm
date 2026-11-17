# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Install the pinned CrossPascal native runtime and upstream TVM Python packages."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile


WHEELS = (
    (
        "apache_tvm-0.26.0-py3-none-win_amd64.whl",
        "a2/4f/ec2c06deac38d2a68a2be1474c13494d6fcbe5c48a09dcd8439300318ecb",
        "3993a1122b922ed2186e946689d6940e4beb8b97ee5012bb118d3cfa472ee681",
    ),
    (
        "apache_tvm_ffi-0.1.14-cp312-cp312-win_amd64.whl",
        "f7/b3/da93f7a4bc35570a4eddec3a48603e995fba2f6b9c825c518c3b0a1d4545",
        "a6af0a29916494a33d40dc469951d9a631b2f5938003a1bfaabc2a7a33e6e3a2",
    ),
)
RUNTIME = {
    "tvm/lib/tvm_runtime.dll": "45e3a9b5d76ea6950cde4442e5bd3208c66a8211d02936e2c0e727f52d2c4ac1",
    "tvm/lib/tvm_runtime_extra.dll": "4fe689aa56b1a46349def33dd75f5066f70e1a0ba5d156e2e74e8667d3a7beee",
    "tvm/lib/tvm_runtime_cuda.dll": "4c67e81596267e94b153e855290fda1320ac0f4852faa36f8ac7184be585e87b",
    "tvm_ffi/lib/tvm_ffi.dll": "2ed31dce9e331d23d6b4a2fa37e4166aa7843c334a3c17381f92dd3bcf122913",
}
NATIVE_COMPILER = {
    "tvm/lib/tvm_compiler.dll": "26d10e38175889caaa1b503781b46a4afe6fe1bfcb04443f438983d95357a6d1",
    "tvm_ffi/lib/tvm_ffi.lib": "5bb27ac9be41e0191701f53b26a8c8c5c16a8fb0eefb5d526e52717f0cbc139d",
}
UPSTREAM_SOURCE = Path(__file__).resolve().parents[1] / "licenses/upstream-windows-0.26.0/SOURCE.json"
REPOSITORY = Path(__file__).resolve().parents[1]
NATIVE_CORE_BYTES = 2345984
NATIVE_CORE_SHA256 = "e7580dbaeb87a480fcaac3ffe30bca1b294c0c370c7c02b24228b1e56fdd4584"
NATIVE_LICENSE_FILES = {
    "LICENSE-tvm-0.26.0.txt": "licenses/TVM-LICENSE",
    "NOTICE-tvm-0.26.0.txt": "licenses/TVM-NOTICE",
    "LICENSE-tvm-ffi-0.1.14.txt": "runtime/windows-x64-cuda/LICENSE-tvm-ffi-0.1.14.txt",
    "NOTICE-tvm-ffi-0.1.14.txt": "runtime/windows-x64-cuda/NOTICE-tvm-ffi-0.1.14.txt",
    "NOTICE-CrossPascal-runtime.txt": "runtime/windows-x64-cuda/NOTICE-CrossPascal-runtime.txt",
}
CORE_SOURCE_FILES = ("SOURCES.json", "threadpool-lifecycle.patch", "build.py")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_wheel(cache, filename, subpath, expected, offline):
    path = cache / filename
    if not path.exists() and not offline:
        url = f"https://files.pythonhosted.org/packages/{subpath}/{filename}"
        with tempfile.TemporaryDirectory(dir=cache) as work:
            download = Path(work) / filename
            try:
                with urlopen(url, timeout=60) as response, download.open("xb") as stream:
                    shutil.copyfileobj(response, stream)
            except URLError as exc:
                raise ValueError(f"Cannot download {filename}: {exc.reason}") from exc
            if sha256(download) != expected:
                raise ValueError(f"Wheel checksum differs for {filename}")
            download.replace(path)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Expected the pinned wheel at {path}")
    return path


def read_dependencies(lock):
    entries = json.loads(lock.read_text(encoding="utf-8"))["wheels"]
    wheels = []
    for entry in entries:
        filename, directory, digest = (entry[key] for key in ("filename", "directory", "sha256"))
        if (not re.fullmatch(r"[A-Za-z0-9._-]+\.whl", filename)
                or not re.fullmatch(r"[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}", directory)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid pinned wheel entry")
        wheels.append((filename, directory, digest))
    if len({entry[0].casefold() for entry in wheels}) != len(wheels):
        raise ValueError("Duplicate pinned wheel")
    return wheels


def validate_native_core(runtime_core):
    if (not runtime_core.is_file() or runtime_core.stat().st_size != NATIVE_CORE_BYTES
            or sha256(runtime_core) != NATIVE_CORE_SHA256):
        raise ValueError("Select the matching CrossPascal SDK tvm_runtime.dll with --runtime-core. "
                         "The upstream wheel core does not provide safe host-thread shutdown.")
    provenance = json.loads((REPOSITORY / "runtime/RUNTIME-PROVENANCE-windows-x64-cuda.json").read_text("utf-8-sig"))
    core = [entry for entry in provenance["files"] if entry["name"] == "tvm_runtime.dll"]
    if core != [{"name": "tvm_runtime.dll", "bytes": NATIVE_CORE_BYTES, "sha256": NATIVE_CORE_SHA256}]:
        raise ValueError("Native core provenance differs from the pinned runtime")
    if (provenance.get("tvm_modified") is not True
            or sha256(REPOSITORY / "native/windows-runtime/threadpool-lifecycle.patch")
            != provenance["core_build"]["patch_sha256"]):
        raise ValueError("Native core patch provenance differs from the supplied source")
    source = json.loads((REPOSITORY / "native/windows-runtime/SOURCES.json").read_text("utf-8-sig"))
    if (source["supplied_core_sha256"] != NATIVE_CORE_SHA256
            or source["patch"]["sha256"] != provenance["core_build"]["patch_sha256"]):
        raise ValueError("Native core source manifest differs from the pinned core and patch")
    return provenance


def stage_wheels(wheels, stage, include_python, include_native_compiler=False, runtime_core=None):
    runtime_core = runtime_core or REPOSITORY / "runtime/windows-x64-cuda/tvm_runtime.dll"
    core_provenance = validate_native_core(runtime_core)
    native = stage / "native"
    native.mkdir()
    python = stage / "python"
    source_bytes = UPSTREAM_SOURCE.read_bytes()
    provenance = json.loads(source_bytes)
    packages = {entry["wheel"]: entry for entry in provenance["packages"]}
    pinned_wheels = {entry[0]: entry[2] for entry in WHEELS}
    if {name: entry["sha256"] for name, entry in packages.items()} != pinned_wheels:
        raise ValueError("Upstream provenance differs from the pinned wheels")
    declared = {entry["member"]: entry["sha256"] for package in packages.values()
                for entry in package["binaries"]}
    if declared != RUNTIME | NATIVE_COMPILER:
        raise ValueError("Upstream provenance differs from the pinned native libraries")
    libraries = RUNTIME | (NATIVE_COMPILER if include_native_compiler else {})
    license_root = native / "licenses/upstream-windows-0.26.0"
    license_root.mkdir(parents=True)
    (license_root / "SOURCE.json").write_bytes(source_bytes)
    supplied_licenses = set()
    expected_licenses = {(package["wheel"], entry["member"])
                         for package in packages.values() for entry in package["licenses"]}
    for wheel in wheels:
        package = packages.get(wheel.name)
        licenses = {entry["member"]: entry for entry in package["licenses"]} if package else {}
        with ZipFile(wheel) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or any(":" in part for part in name.parts)):
                    raise ValueError(f"Unsafe wheel member {member.filename}")
                if member.is_dir():
                    continue
                if member.filename in libraries:
                    with archive.open(member) as source, (native / name.name).open("xb") as target:
                        shutil.copyfileobj(source, target)
                if member.filename in licenses:
                    entry = licenses[member.filename]
                    relative = PurePosixPath(entry["file"])
                    if (relative.is_absolute() or ".." in relative.parts or "\\" in entry["file"]
                            or any(":" in part for part in relative.parts)):
                        raise ValueError("Unsafe upstream license destination")
                    data = archive.read(member)
                    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                        raise ValueError(f"Upstream license checksum differs for {member.filename}")
                    path = license_root.joinpath(*relative.parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and path.read_bytes() != data:
                        raise ValueError(f"Conflicting upstream license at {path}")
                    path.write_bytes(data)
                    supplied_licenses.add((wheel.name, member.filename))
                if include_python:
                    if name.parts[0].endswith(".data"):
                        raise ValueError(f"Unsupported wheel installation layout: {member.filename}")
                    path = python.joinpath(*name.parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, path.open("xb") as target:
                        shutil.copyfileobj(source, target)
    if supplied_licenses != expected_licenses:
        raise ValueError("Pinned upstream license inventory is incomplete")
    for member, expected in libraries.items():
        path = native / PurePosixPath(member).name
        if sha256(path) != expected:
            raise ValueError(f"Native library checksum differs for {path.name}")
    shutil.copyfile(runtime_core, native / "tvm_runtime.dll")
    if sha256(native / "tvm_runtime.dll") != NATIVE_CORE_SHA256:
        raise ValueError("Patched runtime core changed during staging")
    for name, relative in NATIVE_LICENSE_FILES.items():
        shutil.copyfile(REPOSITORY / relative, native / name)
    core_sources = native / "sources/windows-runtime"
    core_sources.mkdir(parents=True)
    for name in CORE_SOURCE_FILES:
        shutil.copyfile(REPOSITORY / "native/windows-runtime" / name, core_sources / name)
    shutil.copyfile(REPOSITORY / "LICENSE", native / "LICENSE")
    for name in ("TVM-LICENSE", "TVM-NOTICE"):
        shutil.copyfile(REPOSITORY / "licenses" / name, native / "licenses" / name)
    installed_libraries = {PurePosixPath(member).name: expected for member, expected in libraries.items()}
    installed_libraries["tvm_runtime.dll"] = NATIVE_CORE_SHA256
    manifest = {
        "schema": "crosspascal.installed-native-tvm-windows.v1",
        "native_compiler_included": include_native_compiler,
        "source": "licenses/upstream-windows-0.26.0/SOURCE.json",
        "binary_modifications": True,
        "python_binaries_modified": False,
        "native_core": {
            "source": "sources/windows-runtime/SOURCES.json",
            "patch": "sources/windows-runtime/threadpool-lifecycle.patch",
            "build_recipe": "sources/windows-runtime/build.py",
            "upstream_core_sha256": RUNTIME["tvm/lib/tvm_runtime.dll"],
            "core_build": core_provenance["core_build"],
            "thread_shutdown": core_provenance["thread_shutdown"],
        },
        "reproducible_rebuild_claim": False,
        "files": [{"file": name, "bytes": (native / name).stat().st_size,
                   "sha256": expected} for name, expected in installed_libraries.items()],
        "metadata_files": [{"file": path.relative_to(native).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha256(path)}
                           for path in sorted(native.rglob("*"))
                           if path.is_file() and path.relative_to(native).as_posix() not in installed_libraries],
    }
    (native / "TVM-PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def install(stage, destination, site_packages):
    if site_packages and list(site_packages.glob("apache_tvm_ffi-*.dist-info")):
        names = {path.name for path in site_packages.glob("apache_tvm_ffi-*.dist-info")}
        if names != {"apache_tvm_ffi-0.1.14.dist-info"}:
            raise ValueError("Use a clean Python environment without a different TVM FFI version")
    roots = [(stage / "native", destination)]
    if site_packages:
        roots.append((stage / "python", site_packages))
    missing = []
    for root, target in roots:
        for source in root.rglob("*"):
            if source.is_file():
                path = target / source.relative_to(root)
                if path.exists():
                    if not path.is_file() or sha256(path) != sha256(source):
                        raise ValueError(f"Use a clean installation directory; existing file differs at {path}")
                else:
                    missing.append((source, path))
    for source, path in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            staged = Path(stream.name)
            try:
                with source.open("rb") as reader:
                    shutil.copyfileobj(reader, stream)
                stream.close()
                try:
                    staged.rename(path)
                except FileExistsError:
                    if not path.is_file() or sha256(path) != sha256(source):
                        raise ValueError(f"Existing file differs at {path}") from None
            finally:
                stream.close()
                staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Directory for verified wheel downloads")
    parser.add_argument("--destination", type=Path, help="TVM DLL directory in the SDK")
    parser.add_argument("--runtime-core", type=Path,
                        default=REPOSITORY / "runtime/windows-x64-cuda/tvm_runtime.dll",
                        help="Patched tvm_runtime.dll from the matching CrossPascal SDK")
    parser.add_argument("--site-packages", type=Path, help="Install the compiler's Python packages too")
    parser.add_argument("--offline", action="store_true", help="Use only wheels already in the cache")
    parser.add_argument("--native-compiler", action="store_true",
                        help="Include the native TVM compiler DLL and FFI import library")
    parser.add_argument("--dependencies", type=Path, help="Additional Python wheel lock for the SDK")
    parser.add_argument("--download-only", action="store_true", help="Prepare a verified cache without installing")
    args = parser.parse_args()
    if not args.download_only and args.destination is None:
        parser.error("--destination is required when installing")
    if args.dependencies and args.site_packages is None and not args.download_only:
        parser.error("--dependencies requires --site-packages")
    if not args.download_only:
        validate_native_core(args.runtime_core)
    packages = list(WHEELS)
    if args.dependencies:
        packages.extend(read_dependencies(args.dependencies))
    args.cache.mkdir(parents=True, exist_ok=True)
    wheels = [get_wheel(args.cache, *entry, args.offline) for entry in packages]
    if args.download_only:
        print(f"Verified {len(wheels)} pinned wheels in {args.cache}.")
        return
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        stage_wheels(wheels, stage, args.site_packages is not None, args.native_compiler, args.runtime_core)
        install(stage, args.destination, args.site_packages)
    print("Verified and installed the CrossPascal TVM 0.26.0 runtime and official TVM FFI 0.1.14.")
    if args.site_packages:
        print("The Python compiler packages retain their upstream binaries.")
    if args.dependencies:
        print(f"Verified and installed {len(packages) - len(WHEELS)} Python dependencies.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
