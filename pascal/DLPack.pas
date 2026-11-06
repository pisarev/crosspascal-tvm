{ ************************************************************************** }
{                                                                            }
{ DLPack                                                                     }
{                                                                            }
{ Copyright © 2025, 2026 Yuriy Pisarev (ypisareff@outlook.com)               }
{                                                                            }
{ ************************************************************************** }

unit DLPack;

{$B-}
{$IFDEF FPC}
  {$MODE DELPHI}
  {$M+}
{$ENDIF}

{$IFDEF FPC}
  {$PACKENUM 4}
{$ELSE}
  {$MINENUMSIZE 4}
{$ENDIF}

interface

uses
  SysUtils;

type
  TDLDeviceType = (
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16
  );

  TDLDevice = record
    device_type: TDLDeviceType;
    device_id: Int32;
  end;

  TDLDataTypeCode = (
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
    kDLFloat8_e3m4 = 7,
    kDLFloat8_e4m3 = 8,
    kDLFloat8_e4m3b11fnuz = 9,
    kDLFloat8_e4m3fn = 10,
    kDLFloat8_e4m3fnuz = 11,
    kDLFloat8_e5m2 = 12,
    kDLFloat8_e5m2fnuz = 13,
    kDLFloat8_e8m0fnu = 14,
    kDLFloat6_e2m3fn = 15,
    kDLFloat6_e3m2fn = 16,
    kDLFloat4_e2m1fn = 17
  );

  TDLDataType = record
    code: UInt8;
    bits: UInt8;
    lanes: UInt16;
  end;

  PDLTensor = ^TDLTensor;
  TDLTensor = record
    data: Pointer;
    device: TDLDevice;
    ndim: Int32;
    dtype: TDLDataType;
    shape: PInt64;
    strides: PInt64;
    byte_offset: UInt64;
  end;

  PDLManagedTensor = ^TDLManagedTensor;
  TDLManagedTensor = record
    dl_tensor: TDLTensor;
    manager_ctx: Pointer;
    deleter: procedure(Tensor: PDLManagedTensor); cdecl;
  end;

  TDLPackVersion = record
    major: UInt32;
    minor: UInt32;
  end;

  PDLManagedTensorVersioned = ^TDLManagedTensorVersioned;
  TDLManagedTensorVersioned = record
    version: TDLPackVersion;
    manager_ctx: Pointer;
    deleter: procedure(Tensor: PDLManagedTensorVersioned); cdecl;
    flags: UInt64;
    dl_tensor: TDLTensor;
  end;

const
  DLPACK_MAJOR_VERSION = 1;
  DLPACK_MINOR_VERSION = 3;
  DLPACK_FLAG_BITMASK_READ_ONLY = UInt64(1) shl 0;
  DLPACK_FLAG_BITMASK_IS_COPIED = UInt64(1) shl 1;
  DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED = UInt64(1) shl 2;

implementation

end.
