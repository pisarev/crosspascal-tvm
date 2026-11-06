# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Install pinned upstream Windows x64 TVM and Python packages."""

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


def stage_wheels(wheels, stage, include_python):
    native = stage / "native"
    native.mkdir()
    python = stage / "python"
    for wheel in wheels:
        with ZipFile(wheel) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or any(":" in part for part in name.parts)):
                    raise ValueError(f"Unsafe wheel member {member.filename}")
                if member.is_dir():
                    continue
                if member.filename in RUNTIME:
                    with archive.open(member) as source, (native / name.name).open("xb") as target:
                        shutil.copyfileobj(source, target)
                if include_python:
                    if name.parts[0].endswith(".data"):
                        raise ValueError(f"Unsupported wheel installation layout: {member.filename}")
                    path = python.joinpath(*name.parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, path.open("xb") as target:
                        shutil.copyfileobj(source, target)
    for member, expected in RUNTIME.items():
        path = native / PurePosixPath(member).name
        if sha256(path) != expected:
            raise ValueError(f"Native library checksum differs for {path.name}")


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
    parser.add_argument("--site-packages", type=Path, help="Install the compiler's Python packages too")
    parser.add_argument("--offline", action="store_true", help="Use only wheels already in the cache")
    parser.add_argument("--dependencies", type=Path, help="Additional Python wheel lock for the SDK")
    parser.add_argument("--download-only", action="store_true", help="Prepare a verified cache without installing")
    args = parser.parse_args()
    if not args.download_only and args.destination is None:
        parser.error("--destination is required when installing")
    if args.dependencies and args.site_packages is None and not args.download_only:
        parser.error("--dependencies requires --site-packages")
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
        stage_wheels(wheels, stage, args.site_packages is not None)
        install(stage, args.destination, args.site_packages)
    print("Verified and installed TVM 0.26.0 and TVM FFI 0.1.14 for Windows x64.")
    if args.dependencies:
        print(f"Verified and installed {len(packages) - len(WHEELS)} Python dependencies.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
