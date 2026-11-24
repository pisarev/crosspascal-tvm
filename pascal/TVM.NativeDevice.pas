{ ************************************************************************** }
{                                                                            }
{ TVM.NativeDevice                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.NativeDevice;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ASMMODE INTEL}{$ENDIF}

interface

function NativeCpuSupportsAVX2FMA: Boolean;
function NativeCudaArchitecture(const DeviceID: Integer = 0): string;

implementation

uses
  {$IFDEF FPC}SysUtils, Dynlibs{$IFDEF MSWINDOWS}, Windows{$ENDIF}
  {$ELSE}System.SysUtils, Winapi.Windows{$ENDIF}, TVM.NativeFiles;

{ CPUID is guaranteed on x86_64. XGETBV is executed only after OSXSAVE is
  advertised; XMM and YMM state must both be enabled by the OS. RBX is the
  only nonvolatile register changed. Delphi x64 uses its unwind-aware
  .PUSHNV directive: docwiki.embarcadero.com/RADStudio/Florence/en/DCC64. }
function NativeCpuSupportsAVX2FMA: Boolean;
{$IF DEFINED(CPUX86_64) OR DEFINED(CPUX64)}
{$IFDEF FPC}assembler; nostackframe;{$ENDIF}
asm
  {$IFDEF FPC}push rbx{$ELSE}.PUSHNV RBX{$ENDIF}
  xor eax, eax
  cpuid
  cmp eax, 7
  jb @@unsupported
  mov eax, 1
  cpuid
  and ecx, $18001000
  cmp ecx, $18001000
  jne @@unsupported
  xor ecx, ecx
  db $0F, $01, $D0
  and eax, 6
  cmp eax, 6
  jne @@unsupported
  mov eax, 7
  xor ecx, ecx
  cpuid
  and ebx, $20
  setnz al
  movzx eax, al
  jmp @@finished
@@unsupported:
  xor eax, eax
@@finished:
  {$IFDEF FPC}pop rbx{$ENDIF}
end;
{$ELSE}
begin
  Result := False;
end;
{$ENDIF}

function NativeCudaArchitecture(const DeviceID: Integer): string;
type
  TInitialize = function(Flags: Cardinal): Integer; cdecl;
  TCount = function(out Count: Integer): Integer; cdecl;
  TDevice = function(out Device: Integer; Ordinal: Integer): Integer; cdecl;
  TAttribute = function(out Value: Integer; Attribute, Device: Integer): Integer; cdecl;
var
  Handle: {$IFDEF FPC}TLibHandle{$ELSE}HMODULE{$ENDIF};
  Initialize: TInitialize;
  Count: TCount;
  Device: TDevice;
  Attribute: TAttribute;
  Devices, Selected, Major, Minor: Integer;
begin
  NativeRequire(DeviceID >= 0, 'Invalid CUDA device index');
  {$IFDEF MSWINDOWS}
  Handle := {$IFDEF FPC}Windows.{$ENDIF}LoadLibraryExW('nvcuda.dll', 0, $800);
  {$ELSE}
  Handle := LoadLibrary('libcuda.so.1');
  {$ENDIF}
  NativeRequire(Handle <> 0, 'The NVIDIA CUDA driver is unavailable');
  try
    Initialize := TInitialize(GetProcAddress(Handle, 'cuInit'));
    Count := TCount(GetProcAddress(Handle, 'cuDeviceGetCount'));
    Device := TDevice(GetProcAddress(Handle, 'cuDeviceGet'));
    Attribute := TAttribute(GetProcAddress(Handle, 'cuDeviceGetAttribute'));
    NativeRequire(Assigned(Initialize) and Assigned(Count) and Assigned(Device) and Assigned(Attribute),
      'The CUDA driver does not export its device query API');
    NativeRequire(Initialize(0) = 0, 'Cannot initialize the CUDA device query');
    NativeRequire((Count(Devices) = 0) and (DeviceID < Devices), 'The selected CUDA device does not exist');
    NativeRequire(Device(Selected, DeviceID) = 0, 'Cannot query the selected CUDA device');
    { CUDA driver cuda.h: CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR. }
    NativeRequire((Attribute(Major, 75, Selected) = 0) and (Attribute(Minor, 76, Selected) = 0) and
      (Major > 0) and (Minor >= 0) and (Minor <= 9), 'Cannot query CUDA compute capability');
    Result := 'sm_' + IntToStr(Major) + IntToStr(Minor);
  finally
    {$IFDEF FPC}UnloadLibrary(Handle){$ELSE}FreeLibrary(Handle){$ENDIF};
  end;
end;

end.
