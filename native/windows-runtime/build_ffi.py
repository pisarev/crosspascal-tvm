# Copyright © 2026 Yuriy Pisarev. MIT license; see ../../LICENSE.
"""Build the pinned TVM FFI with correct UTF-8 Windows library paths.

Archives must already exist in --archive-cache. This recipe performs no downloads.
"""
import argparse
import json
import subprocess
from pathlib import Path
from build import digest,extract

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--archive-cache',type=Path,required=True)
    p.add_argument('--cmake',default='cmake')
    p.add_argument('--git',default='git')
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    if args.jobs<1:p.error('--jobs must be positive')
    recipe=Path(__file__).resolve().parent
    spec=json.loads((recipe/'FFI-SOURCES.json').read_text(encoding='utf-8'))
    patch=recipe/spec['patch']['file']
    if digest(patch)!=spec['patch']['sha256']:raise ValueError('FFI patch checksum differs')
    for row in spec['archives']:
        cached=args.archive_cache/row['file']
        if not cached.is_file() or digest(cached)!=row['sha256']:
            raise ValueError('Missing or changed pinned archive: '+str(cached))
    args.work.mkdir(parents=True,exist_ok=False)
    source=args.work/'tvm-ffi'
    for row in spec['archives']:extract(args.archive_cache/row['file'],source/row['destination'])
    changed=source/'src/ffi/extra/library_module_dynamic_lib.cc'
    if digest(changed)!=spec['unmodified_source_sha256']:raise ValueError('Unexpected upstream FFI source')
    for mode in ['--check','']:
        command=[args.git,'-C',str(source),'apply']+([mode] if mode else [])+[str(patch.resolve())]
        subprocess.run(command,check=True)
    changed.write_bytes(changed.read_bytes().replace(b'\r\n',b'\n'))
    if digest(changed)!=spec['modified_source_sha256']:raise ValueError('Unexpected patched FFI source')
    build=args.work/'build'
    commands=[([args.cmake,'-S',str(source),'-B',str(build),'-G','Visual Studio 17 2022','-A','x64',
       '-DTVM_FFI_USE_LIBBACKTRACE=OFF','-DTVM_FFI_BUILD_TESTS=OFF','-DTVM_FFI_BUILD_PYTHON_MODULE=OFF',
       '-DTVM_FFI_USE_EXTRA_CXX_API=ON'],'configure.log'),
       ([args.cmake,'--build',str(build),'--config','Release','--target','tvm_ffi_shared','--parallel',str(args.jobs)],'build.log')]
    for command,name in commands:
        with (args.work/name).open('xb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    outputs=list(build.rglob('tvm_ffi.dll'))
    if len(outputs)!=1:raise ValueError('Expected exactly one built FFI DLL')
    dll=outputs[0]
    result=dict(status='BUILT_PINNED_FFI_UTF8_WINDOWS_LOADER',source_manifest=spec,
        dll=str(dll.resolve()),bytes=dll.stat().st_size,sha256=digest(dll),numerical_acceptance=False)
    (args.work/'BUILD-RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
