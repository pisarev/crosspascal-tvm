{ ************************************************************************** }
{                                                                            }
{ TVM                                                                        }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit TVM;

{$B-}
{$IFDEF FPC}
  {$MODE DELPHI}
{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}
  {$IFDEF MSWINDOWS}Windows,{$ENDIF}{$IFDEF FPC}dynlibs,{$ENDIF}Classes, Math,
  SysUtils, Types, DLPack, OpenCVTypes, OpenCVConsts, CrossVision.Mat;
  {$ELSE}
  WinApi.Windows, System.SysUtils, System.Classes, System.Math, DLPack, OpenCVTypes,
  OpenCVConsts, CrossVision.Mat;
  {$ENDIF}

const
  kTVMFFINone = 0;
  kTVMFFIInt = 1;
  kTVMFFIBool = 2;
  kTVMFFIFloat = 3;
  kTVMFFIOpaquePtr = 4;
  kTVMFFIDataType = 5;
  kTVMFFIDevice = 6;
  kTVMFFIDLTensorPtr = 7;
  kTVMFFIRawStr = 8;
  kTVMFFIByteArrayPtr = 9;
  kTVMFFIObjectRValueRef = 10;
  kTVMFFISmallStr = 11;
  kTVMFFIStaticObjectBegin = 64;
  kTVMFFIObject = 64;
  kTVMFFIStr = 65;
  kTVMFFIBytes = 66;
  kTVMFFIError = 67;
  kTVMFFIFunction = 68;
  kTVMFFIShape = 69;
  kTVMFFITensor = 70;
  kTVMFFIModule = 73;
  TVMFFIObjectPayloadOffset = 24;
  TVMFFIObjectTypeIndexOffset = 8;
  kTVMVMAllocNaive = 1;
  kTVMVMAllocPooled = 2;

type
  PTVMFFIAny = ^TTVMFFIAny;
  TTVMFFIAny = record
    TypeIndex: Int32;
    Padding: UInt32;
    case Integer of
      0: (VInt64: Int64);
      1: (VFloat64: Double);
      2: (VPtr: Pointer);
      3: (VCStr: PAnsiChar);
      4: (VObj: Pointer);
      5: (VDataType: TDLDataType);
      6: (VDevice: TDLDevice);
      7: (VBytes: array[0..7] of AnsiChar);
      8: (VUInt64: UInt64);
  end;

  PTVMFFIByteArray = ^TTVMFFIByteArray;
  TTVMFFIByteArray = record
    Data: PAnsiChar;
    Size: NativeUInt;
  end;

  PTVMFFIShapeCell = ^TTVMFFIShapeCell;
  TTVMFFIShapeCell = record
    Data: PInt64;
    Size: NativeUInt;
  end;

  TTVMFFIVersion = record
    Major: UInt32;
    Minor: UInt32;
    Patch: UInt32;
  end;

  PTVMFFIErrorCell = ^TTVMFFIErrorCell;
  TTVMFFIErrorCell = record
    Kind: TTVMFFIByteArray;
    Message: TTVMFFIByteArray;
    Backtrace: TTVMFFIByteArray;
    UpdateBacktrace: Pointer;
    CauseChain: Pointer;
    ExtraContext: Pointer;
  end;

  TTVMFFIObjectHandle = Pointer;

  TTVMModuleHandle = TTVMFFIObjectHandle;
  TTVMFunctionHandle = TTVMFFIObjectHandle;
  TTVMObjectHandle = TTVMFFIObjectHandle;

  TTVMInferenceObserver = function(const Context: Pointer; const InputTensor: PDLTensor;
    const OutputTensors: TArray<PDLTensor>; const Examined: Int64; const Error: PString): Boolean;

  TModelDescriptor = record
    Device: TDLDevice;
    ExecutableHandle: TTVMFFIObjectHandle;
    VMHandle: TTVMFFIObjectHandle;
    InitFunc: TTVMFFIObjectHandle;
    SetInputFunc: TTVMFFIObjectHandle;
    RunFunc: TTVMFFIObjectHandle;
    GetOutputArityFunc: TTVMFFIObjectHandle;
    GetOutputFunc: TTVMFFIObjectHandle;
    OutputHandles: TArray<TTVMFFIObjectHandle>;
    InputSpec: PDLTensor;
    EntryName: string;
    InputNames: TArray<string>;
    InputShapes: TArray<TArray<Int64>>;
    InferenceObserver: TTVMInferenceObserver;
    InferenceObserverContext: Pointer;
  end;

  TTVMInputBinding = record
    Name: string;
    Tensor: PDLTensor;
    DTypeCode: TDLDataTypeCode;
    DTypeBits: Integer;
  end;

  TDLDataTypeCodeSet = set of TDLDataTypeCode;

  TImageTensor = record
  public
    var
      BIndex: Integer;
      WIndex: Integer;
      HIndex: Integer;
      CIndex: Integer;
  end;

  TTVMRuntime = record
  public
    type
      TModule = record
      public
        type
          TTVMFFIGetVersion = procedure(out OutVersion: TTVMFFIVersion); cdecl;
          TTVMFFIObjectIncRef = function(Obj: TTVMFFIObjectHandle): Integer; cdecl;
          TTVMFFIObjectDecRef = function(Obj: TTVMFFIObjectHandle): Integer; cdecl;
          TTVMFFIFunctionGetGlobal = function(const Name: PTVMFFIByteArray;
            out OutHandle: TTVMFFIObjectHandle): Integer; cdecl;
          TTVMFFIFunctionCall = function(Func: TTVMFFIObjectHandle; Args: PTVMFFIAny; NumArgs: Int32;
            Res: PTVMFFIAny): Integer; cdecl;
          TTVMFFIErrorMoveFromRaised = procedure(out OutHandle: TTVMFFIObjectHandle); cdecl;
          TTVMFFITensorFromDLPackVersioned = function(From: PDLManagedTensorVersioned;
            RequireAlignment, RequireContiguous: Int32; out OutHandle: TTVMFFIObjectHandle): Integer; cdecl;
          TTVMFFITensorToDLPackVersioned = function(From: TTVMFFIObjectHandle;
            out OutTensor: PDLManagedTensorVersioned): Integer; cdecl;
          TTVMFFISafeCall = function(Handle: Pointer; Args: PTVMFFIAny; NumArgs: Int32;
            Res: PTVMFFIAny): Integer; cdecl;
          TTVMFFIFunctionCreate = function(Self: Pointer; SafeCall: TTVMFFISafeCall; Deleter: Pointer;
            out OutFunc: TTVMFFIObjectHandle): Integer; cdecl;
          TTVMFFIFunctionSetGlobal = function(const Name: PTVMFFIByteArray; Func: TTVMFFIObjectHandle;
            AllowOverride: Int32): Integer; cdecl;
          TTVMFFIErrorSetRaisedFromCStr = procedure(Kind, ErrMessage: PAnsiChar); cdecl;
        const
          {$IFDEF UNIX}
          {$IFDEF NO_LOCAL_TVM}
          Path = '';
          FFIName = 'libtvm_ffi.so';
          RuntimeName = 'libtvm_runtime.so';
          {$ELSE}
          Path = 'bin/tvm/';
          FFIName = 'libtvm_ffi.so';
          RuntimeName = 'libtvm_runtime.so';
          {$ENDIF}
          {$ELSE}
          {$IFDEF NO_LOCAL_TVM}
          Path = '';
          FFIName = 'tvm_ffi.dll';
          RuntimeName = 'tvm_runtime.dll';
          {$ELSE}
          Path = 'tvm\';
          FFIName = 'tvm_ffi.dll';
          RuntimeName = 'tvm_runtime.dll';
          {$ENDIF}
          {$ENDIF}
          {$IFDEF UNIX}
          CudaRuntimeName = 'libtvm_runtime_cuda.so';
          ExtraRuntimeName = 'libtvm_runtime_extra.so';
          {$ELSE}
          CudaRuntimeName = 'tvm_runtime_cuda.dll';
          ExtraRuntimeName = 'tvm_runtime_extra.dll';
          {$ENDIF}
        class var
          Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          RuntimeHandle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          CudaRuntimeHandle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          ExtraRuntimeHandle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          LibraryPath: string;
          LibraryName: string;
          Error: string;
          TVMFFIGetVersion: TTVMFFIGetVersion;
          TVMFFIObjectIncRef: TTVMFFIObjectIncRef;
          TVMFFIObjectDecRef: TTVMFFIObjectDecRef;
          TVMFFIFunctionGetGlobal: TTVMFFIFunctionGetGlobal;
          TVMFFIFunctionCall: TTVMFFIFunctionCall;
          TVMFFIErrorMoveFromRaised: TTVMFFIErrorMoveFromRaised;
          TVMFFITensorFromDLPackVersioned: TTVMFFITensorFromDLPackVersioned;
          TVMFFITensorToDLPackVersioned: TTVMFFITensorToDLPackVersioned;
          TVMFFIFunctionCreate: TTVMFFIFunctionCreate;
          TVMFFIFunctionSetGlobal: TTVMFFIFunctionSetGlobal;
          TVMFFIErrorSetRaisedFromCStr: TTVMFFIErrorSetRaisedFromCStr;
          GExecutableLoadFromFile: TTVMFFIObjectHandle;
          GModuleLoadFromFile: TTVMFFIObjectHandle;
          GModuleGetFunction: TTVMFFIObjectHandle;
          GTensorCopyFromBytes: TTVMFFIObjectHandle;
          GTensorCopyToBytes: TTVMFFIObjectHandle;
          GTensorCopyFromTo: TTVMFFIObjectHandle;
        class function GetProcAddress(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          const ProcName: string; const RaiseError: Boolean): Pointer; static;
        class procedure Initialize(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          const RaiseError: Boolean); static;
        class function GetGlobalFunc(const Name: string): TTVMFFIObjectHandle; static;
        class procedure ResolveGlobals; static;
      end;
      TValue = record
      public
        procedure Reset;
        var
          case Byte of
            0: (Unsigned8: PByte);
            1: (Signed8: PShortint);
            2: (Unsigned16: PWord);
            3: (Signed16: PSmallint);
            4: (Unsigned32: PLongWord);
            5: (Signed32: PInteger);
            6: (Signed64: PInt64);
            7: (Unsigned64: PUInt64);
            8: (Float32: PSingle);
            9: (Float64: PDouble);
            10: (Float80: PExtended);
      end;
    class function CallGlobal(const FuncName: string; const Args: array of TTVMFFIAny; out Res: TTVMFFIAny;
      const Error: PString = nil): Boolean; static;
    class function CallFunc(const Func: TTVMFFIObjectHandle; const Args: array of TTVMFFIAny;
      out Res: TTVMFFIAny; const Error: PString = nil): Boolean; static;
    class function LastError: string; static;
    class function RegisterGlobalFunc(const Name: string; const SafeCall: TModule.TTVMFFISafeCall;
      const AllowOverride: Boolean; const Error: PString = nil): Boolean; static;
    class function AnyInt(const Value: Int64): TTVMFFIAny; static;
    class function AnyBool(const Value: Boolean): TTVMFFIAny; static;
    class function AnyFloat(const Value: Double): TTVMFFIAny; static;
    class function AnyPtr(const Value: Pointer): TTVMFFIAny; static;
    class function AnyCStr(const Value: PAnsiChar): TTVMFFIAny; static;
    class function AnyObject(const Value: TTVMFFIObjectHandle): TTVMFFIAny; static;
    class function AnyDLTensor(const Value: PDLTensor): TTVMFFIAny; static;
    class function AnyDevice(const Value: TDLDevice): TTVMFFIAny; static;
    class function AnyDataType(const Value: TDLDataType): TTVMFFIAny; static;
    class procedure ObjectRetain(const Obj: TTVMFFIObjectHandle); static;
    class procedure ObjectRelease(var Obj: TTVMFFIObjectHandle); static;
    class function TensorDLTensor(const TensorObj: TTVMFFIObjectHandle): PDLTensor; static;
    class function ModuleLoadFromFile(const FileName, ModuleFormat: string;
      out ModuleHandle: TTVMFFIObjectHandle; const Error: PString = nil): Boolean; static;
    class function ExecutableLoadFromFile(const FileName: string; out ExecutableHandle: TTVMFFIObjectHandle;
      const Error: PString = nil): Boolean; static;
    class function ModuleGetFunction(const ModuleHandle: TTVMFFIObjectHandle; const Name: string;
      const QueryImports: Boolean; out FuncHandle: TTVMFFIObjectHandle;
      const Error: PString = nil): Boolean; static;
    class function CopyToHost(const Tensor: PDLTensor; const Host: Pointer; const NBytes: NativeUInt;
      const Error: PString = nil): Boolean; static;
    class function CopyFromHost(const Tensor: PDLTensor; const Host: Pointer; const NBytes: NativeUInt;
      const Error: PString = nil): Boolean; static;
    class function ConvertImageToTensorFormat(const Source: IMat; out Target: IMat; const Tensor: PDLTensor;
      const ImageTensor: TImageTensor; const Scale: Single; const Interpolation: Integer = INTER_LINEAR;
      const Error: PString = nil): Boolean; static;
    class function ConvertImageToDLTensor(const Image: IMat; const Device: TDLDevice;
      const TargetLayout: TImageTensor; out OutputTensor: PDLTensor;
      const DTypeCode: TDLDataTypeCode = kDLUInt; const DTypeBits: Integer = 8; const DTypeLanes: Integer = 1;
      const BatchSize: Int64 = 1; const Error: PString = nil): Boolean; static;
    class function CreateImageArrayFromTensor(const Tensor: PDLTensor; out ImageArray: TArray<IMat>;
      const ImageTensor: TImageTensor; const Error: PString = nil): Boolean; static;
    class function ConvertImageTo8U(const Image: IMat): IMat; static;
    class function AllocateTensor(const Shape: PInt64;
      const NDim, DTypeCode, DTypeBits, DTypeLanes, DeviceType, DeviceID: Integer; out Tensor: PDLTensor;
      const Error: PString = nil): Boolean; static;
    class function AdoptCpuTensor(var Source: PDLTensor; out TensorHandle: TTVMFFIObjectHandle;
      const Error: PString = nil): Boolean; static;
    class function AllocDeviceTensor(const Shape: array of Int64;
      const DTypeCode, DTypeBits, DTypeLanes, DeviceType, DeviceID: Integer;
      out TensorHandle: TTVMFFIObjectHandle; const Error: PString = nil): Boolean; static;
    class function CopyBytesToTensor(const TensorHandle: TTVMFFIObjectHandle; const Source: Pointer;
      const ByteCount: Int64; const Error: PString = nil): Boolean; static;
    class function CopyBytesFromTensor(const TensorHandle: TTVMFFIObjectHandle;
      const Target: Pointer; const ByteCount: Int64; const Error: PString = nil): Boolean; static;
    class function GetOrCreateCpuTensor(const SrcTensor: PDLTensor; out CpuTensor: PDLTensor;
      const TensorName, ContextPrefix: string; const Error: PString = nil): PDLTensor; static;
    class procedure FreeTensors(var Tensors: TArray<PDLTensor>); static;
    class function LoadModel(const ModelFileName, ModelFormat: string; out Descriptor: TModelDescriptor;
      const DeviceType: TDLDeviceType; const DeviceID: Integer = 0; const EntryName: string = 'main';
      const Error: PString = nil): Boolean; static;
    class procedure SetInputSpec(var Descriptor: TModelDescriptor; const Shape: array of Int64;
      const DTypeCode: TDLDataTypeCode = kDLFloat; const DTypeBits: Integer = 32;
      const DTypeLanes: Integer = 1); static;
    class procedure FreeModel(var Descriptor: TModelDescriptor); static;
    class function GetInputTensor(const Descriptor: TModelDescriptor; const TensorIndex: Int64;
      out InputTensor: PDLTensor; const Error: PString = nil): Boolean; static;
    class function GetInputTensorIndex(const Descriptor: TModelDescriptor; const TensorName: string;
      out TensorIndex: Int64; const Error: PString = nil): Boolean; static;
    class function GetOutputTensor(const Descriptor: TModelDescriptor; const TensorIndex: Int64;
      out OutputTensor: PDLTensor; const Error: PString = nil): Boolean; static;
    class function GetTensorShape(const Tensor: PDLTensor): TArray<Int64>; static;
    class function InitializeOutputTensors(const Descriptor: TModelDescriptor;
      out OutputTensors: TArray<PDLTensor>; const DeviceType, DeviceID: Integer;
      const Error: PString = nil): Boolean; static;
    class function CheckTensor(const Tensor: PDLTensor; const DTypeCodeSet: TDLDataTypeCodeSet;
      const DTypeBits: Integer; const DTypeLanes: Integer = 1): Boolean; static;
    class function TensorsAreFinite(const Tensors: TArray<PDLTensor>; out Examined: Int64;
      const Error: PString = nil): Boolean; static;
    class function RunInference(var Descriptor: TModelDescriptor; const InputTensorName: string;
      const InputTensor: PDLTensor; var OutputTensors: TArray<PDLTensor>;
      const Error: PString = nil): Boolean; static;
    class function RunInferenceMulti(var Descriptor: TModelDescriptor;
      const Inputs: array of TTVMInputBinding; var OutputTensors: TArray<PDLTensor>;
      const Error: PString = nil): Boolean; static;
    class function IsImageShape(const Tensor: TDLTensor; const ImageTensor: TImageTensor): Boolean; static;
  end;

  TTVMBackend = record
  public
    type
      TModule = record
      public
        type
          TTVMBackendAllocWorkspace = function(DeviceType, DeviceID: Integer; NBytes: UInt64;
            DTypeCodeHint, DTypeBitsHint: Integer): Pointer; cdecl;
          TTVMBackendFreeWorkspace = function(DeviceType, DeviceID: Integer; Ptr: Pointer): Integer; cdecl;
          TTVMBackendParallelBarrier = function(TaskID: Integer; PEnv: Pointer): Integer; cdecl;
        class var
          Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          Error: string;
          TVMBackendAllocWorkspace: TTVMBackendAllocWorkspace;
          TVMBackendFreeWorkspace: TTVMBackendFreeWorkspace;
          TVMBackendParallelBarrier: TTVMBackendParallelBarrier;
        class function GetProcAddress(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          const ProcName: string; const RaiseError: Boolean): Pointer; static;
        class procedure Initialize(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
          const RaiseError: Boolean); static;
      end;
  end;

const
  NHWCTensor: TImageTensor = (BIndex: 0; WIndex: 2; HIndex: 1; CIndex: 3);
  BHWCTensor: TImageTensor = (BIndex: 0; WIndex: 2; HIndex: 1; CIndex: 3);
  NCHWTensor: TImageTensor = (BIndex: 0; WIndex: 3; HIndex: 2; CIndex: 1);
  BCHWTensor: TImageTensor = (BIndex: 0; WIndex: 3; HIndex: 2; CIndex: 1);
  TensorflowTensor: TImageTensor = (BIndex: 0; WIndex: 2; HIndex: 1; CIndex: 3);
  Yolo11Tensor: TImageTensor = (BIndex: 0; WIndex: 3; HIndex: 2; CIndex: 1);

procedure Initialize;
procedure Finalize;

function ApplySigmoidToImage(const Image32F: IMat): Boolean; inline;
function ShapeToString(const Shape: TArray<Int64>): string; inline;
function CreateShapeArray(const Shape: PInt64; const NDim: Integer): TArray<Int64>; inline;

function ConvertBgrToRgb(const Image: IMat): IMat;

implementation

uses
  {$IFDEF FPC}
  Generics.Collections, Generics.Defaults;
  {$ELSE}
  System.Types, System.Generics.Collections, System.Generics.Defaults;
  {$ENDIF}

function Float16ToSingle(const Bits: Word): Single;
type
  TFloatBits = record
    case Integer of
      0: (FloatValue: Single);
      1: (IntegerBits: LongWord);
  end;
var
  SignBit, Mantissa: LongWord;
  Exponent: Integer;
  R: TFloatBits;
begin
  SignBit := (Bits shr 15) and $1;
  Exponent := (Bits shr 10) and $1F;
  Mantissa := Bits and $3FF;
  if Exponent = 0 then
  begin
    if Mantissa = 0 then
      R.IntegerBits := SignBit shl 31
    else begin
      Exponent := 1;
      while (Mantissa and $400) = 0 do
      begin
        Mantissa := Mantissa shl 1;
        Dec(Exponent);
      end;
      Mantissa := Mantissa and $3FF;
      Exponent := Exponent + (127 - 15);
      R.IntegerBits := (SignBit shl 31) or (LongWord(Exponent) shl 23) or (Mantissa shl 13);
    end;
  end
  else if Exponent = 31 then
      R.IntegerBits := (SignBit shl 31) or ($FF shl 23) or (Mantissa shl 13)
    else
      R.IntegerBits := (SignBit shl 31) or (LongWord(Exponent + (127 - 15)) shl 23) or (Mantissa shl
        13);
  Result := R.FloatValue;
end;

function FFIByteArrayToString(const ByteArray: TTVMFFIByteArray): string;
var
  Ansi: AnsiString;
begin
  if (ByteArray.Data = nil) or (ByteArray.Size = 0) then Exit('');
  SetString(Ansi, ByteArray.Data, ByteArray.Size);
  Result := string(Ansi);
end;

function DLTensorDataSize(const Tensor: PDLTensor): NativeUInt;
var
  I: Integer;
  Count: Int64;
begin
  if not Assigned(Tensor) or (Tensor.ndim < 0) then Exit(0);
  Count := 1;
  for I := 0 to Tensor.ndim - 1 do Count := Count * Tensor.shape[I];
  Result := NativeUInt(Count) * NativeUInt((Tensor.dtype.bits * Tensor.dtype.lanes + 7) div 8);
end;

function NewCpuTensor(const Shape: array of Int64;
  const DTypeCode, DTypeBits, DTypeLanes: Integer): PDLTensor;
const
  DataAlign = 64;
var
  NDim, I: Integer;
  Count, Stride: Int64;
  HeaderBytes, ShapeBytes, MetaBytes, DataStart, DataSize, Total: NativeUInt;
  Block: PByte;
  ShapePtr, StridesPtr: PInt64;
begin
  NDim := Length(Shape);
  Count := 1;
  for I := 0 to NDim - 1 do Count := Count * Shape[I];
  DataSize := NativeUInt(Count) * NativeUInt((DTypeBits * DTypeLanes + 7) div 8);
  HeaderBytes := SizeOf(TDLTensor);
  ShapeBytes := NativeUInt(NDim) * SizeOf(Int64);
  MetaBytes := HeaderBytes + ShapeBytes + ShapeBytes;
  DataStart := (MetaBytes + (DataAlign - 1)) and not NativeUInt(DataAlign - 1);
  Total := DataStart + DataSize;
  GetMem(Block, Total);
  FillChar(Block^, Total, 0);
  Result := PDLTensor(Block);
  ShapePtr := PInt64(Block + HeaderBytes);
  StridesPtr := PInt64(Block + HeaderBytes + ShapeBytes);
  Result.data := Block + DataStart;
  Result.device.device_type := kDLCPU;
  Result.device.device_id := 0;
  Result.ndim := NDim;
  Result.dtype.code := DTypeCode;
  Result.dtype.bits := DTypeBits;
  Result.dtype.lanes := DTypeLanes;
  Result.shape := ShapePtr;
  Result.strides := StridesPtr;
  Result.byte_offset := 0;
  Stride := 1;
  for I := NDim - 1 downto 0 do
  begin
    ShapePtr[I] := Shape[I];
    StridesPtr[I] := Stride;
    Stride := Stride * Shape[I];
  end;
end;

function ConvertBgrToRgb(const Image: IMat): IMat;
begin
  if Assigned(Image) and (Image.Channels = 3) then
    Result := Image.CvtColor(COLOR_BGR2RGB)
  else
    Result := Image;
end;

function ConvertColorChannels(const Source: IMat; const Channels: Integer): IMat;
begin
  Result := Source;
  if not Assigned(Source) or (Source.Channels = Channels) then Exit;
  case Source.Channels of
    1:
      case Channels of
        3: Result := Source.CvtColor(COLOR_GRAY2BGR);
        4: Result := Source.CvtColor(COLOR_GRAY2BGRA);
      end;
    3:
      case Channels of
        1: Result := Source.CvtColor(COLOR_BGR2GRAY);
        4: Result := Source.CvtColor(COLOR_BGR2BGRA);
      end;
    4:
      case Channels of
        1: Result := Source.CvtColor(COLOR_BGRA2GRAY);
        3: Result := Source.CvtColor(COLOR_BGRA2BGR);
      end;
  end;
end;

class function TTVMRuntime.TModule.GetProcAddress(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$IFEND};
  const ProcName: string; const RaiseError: Boolean): Pointer;
begin
  Result := {$IF Defined(FPC) AND Defined(UNIX)}dynlibs{$ELSE}{$IFNDEF FPC}Winapi.{$ENDIF}Windows{$ENDIF}.GetProcAddress(Handle,
    PChar(ProcName));
  if not Assigned(Result) and RaiseError then
    raise Exception.CreateFmt('Error while loading TVM function "%s" from "%s"',
      [ProcName, TTVMRuntime.TModule.LibraryName]);
end;

class procedure TTVMRuntime.TModule.Initialize(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
  const RaiseError: Boolean);
begin
  TVMFFIGetVersion := GetProcAddress(Handle, 'TVMFFIGetVersion', RaiseError);
  TVMFFIObjectIncRef := GetProcAddress(Handle, 'TVMFFIObjectIncRef', RaiseError);
  TVMFFIObjectDecRef := GetProcAddress(Handle, 'TVMFFIObjectDecRef', RaiseError);
  TVMFFIFunctionGetGlobal := GetProcAddress(Handle, 'TVMFFIFunctionGetGlobal', RaiseError);
  TVMFFIFunctionCall := GetProcAddress(Handle, 'TVMFFIFunctionCall', RaiseError);
  TVMFFIErrorMoveFromRaised := GetProcAddress(Handle, 'TVMFFIErrorMoveFromRaised', RaiseError);
  TVMFFITensorFromDLPackVersioned := GetProcAddress(Handle, 'TVMFFITensorFromDLPackVersioned', RaiseError);
  TVMFFITensorToDLPackVersioned := GetProcAddress(Handle, 'TVMFFITensorToDLPackVersioned', RaiseError);
  TVMFFIFunctionCreate := GetProcAddress(Handle, 'TVMFFIFunctionCreate', False);
  TVMFFIFunctionSetGlobal := GetProcAddress(Handle, 'TVMFFIFunctionSetGlobal', False);
  TVMFFIErrorSetRaisedFromCStr := GetProcAddress(Handle, 'TVMFFIErrorSetRaisedFromCStr', False);
end;

class function TTVMRuntime.TModule.GetGlobalFunc(const Name: string): TTVMFFIObjectHandle;
var
  AnsiName: AnsiString;
  ByteArray: TTVMFFIByteArray;
begin
  Result := nil;
  if not Assigned(TVMFFIFunctionGetGlobal) then Exit;
  AnsiName := AnsiString(Name);
  ByteArray.Data := PAnsiChar(AnsiName);
  ByteArray.Size := Length(AnsiName);
  if TVMFFIFunctionGetGlobal(@ByteArray, Result) <> 0 then
    Result := nil;
end;

class procedure TTVMRuntime.TModule.ResolveGlobals;
begin
  GExecutableLoadFromFile := GetGlobalFunc('relax.ExecutableLoadFromFile');
  GModuleLoadFromFile := GetGlobalFunc('ffi.ModuleLoadFromFile');
  GModuleGetFunction := GetGlobalFunc('ffi.ModuleGetFunction');
  GTensorCopyFromBytes := GetGlobalFunc('runtime.TVMTensorCopyFromBytes');
  GTensorCopyToBytes := GetGlobalFunc('runtime.TVMTensorCopyToBytes');
  GTensorCopyFromTo := GetGlobalFunc('runtime.TVMTensorCopyFromTo');
end;

procedure TTVMRuntime.TValue.Reset;
begin
  FillChar(Self, SizeOf(TValue), 0);
end;

class function TTVMBackend.TModule.GetProcAddress(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$IFEND};
  const ProcName: string; const RaiseError: Boolean): Pointer;
begin
  Result := {$IF Defined(FPC) AND Defined(UNIX)}dynlibs{$ELSE}{$IFNDEF FPC}Winapi.{$ENDIF}Windows{$ENDIF}.GetProcAddress(Handle,
    PChar(ProcName));
  if not Assigned(Result) and RaiseError then
    raise Exception.CreateFmt('Error while loading TVM Backend function "%s" from "%s"',
      [ProcName, TTVMRuntime.TModule.LibraryName]);
end;

class procedure TTVMBackend.TModule.Initialize(const Handle: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
  const RaiseError: Boolean);
begin
  TVMBackendAllocWorkspace := GetProcAddress(Handle, 'TVMBackendAllocWorkspace', RaiseError);
  TVMBackendFreeWorkspace := GetProcAddress(Handle, 'TVMBackendFreeWorkspace', RaiseError);
  TVMBackendParallelBarrier := GetProcAddress(Handle, 'TVMBackendParallelBarrier', RaiseError);
end;

class function TTVMRuntime.LastError: string;
var
  ErrHandle: TTVMFFIObjectHandle;
  Cell: PTVMFFIErrorCell;
  Kind, Message: string;
begin
  Result := '';
  if not Assigned(TModule.TVMFFIErrorMoveFromRaised) then Exit;
  ErrHandle := nil;
  TModule.TVMFFIErrorMoveFromRaised(ErrHandle);
  if not Assigned(ErrHandle) then Exit;
  try
    Cell := PTVMFFIErrorCell(PByte(ErrHandle) + TVMFFIObjectPayloadOffset);
    Kind := FFIByteArrayToString(Cell.Kind);
    Message := FFIByteArrayToString(Cell.Message);
    if Kind <> '' then
      Result := Kind + ': ' + Message
    else
      Result := Message;
  finally
    ObjectRelease(ErrHandle);
  end;
end;

class function TTVMRuntime.AnyInt(const Value: Int64): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIInt;
  Result.VInt64 := Value;
end;

class function TTVMRuntime.AnyBool(const Value: Boolean): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIBool;
  Result.VInt64 := Ord(Value);
end;

class function TTVMRuntime.AnyFloat(const Value: Double): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIFloat;
  Result.VFloat64 := Value;
end;

class function TTVMRuntime.AnyPtr(const Value: Pointer): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIOpaquePtr;
  Result.VPtr := Value;
end;

class function TTVMRuntime.AnyCStr(const Value: PAnsiChar): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIRawStr;
  Result.VCStr := Value;
end;

class function TTVMRuntime.AnyObject(const Value: TTVMFFIObjectHandle): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  if Assigned(Value) then
    Result.TypeIndex := PInteger(PByte(Value) + TVMFFIObjectTypeIndexOffset)^
  else
    Result.TypeIndex := kTVMFFINone;
  Result.VObj := Value;
end;

class function TTVMRuntime.AnyDLTensor(const Value: PDLTensor): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIDLTensorPtr;
  Result.VPtr := Value;
end;

class function TTVMRuntime.AnyDevice(const Value: TDLDevice): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIDevice;
  Result.VDevice := Value;
end;

class function TTVMRuntime.AnyDataType(const Value: TDLDataType): TTVMFFIAny;
begin
  FillChar(Result, SizeOf(TTVMFFIAny), 0);
  Result.TypeIndex := kTVMFFIDataType;
  Result.VDataType := Value;
end;

class procedure TTVMRuntime.ObjectRetain(const Obj: TTVMFFIObjectHandle);
begin
  if Assigned(Obj) and Assigned(TModule.TVMFFIObjectIncRef) then
    TModule.TVMFFIObjectIncRef(Obj);
end;

class procedure TTVMRuntime.ObjectRelease(var Obj: TTVMFFIObjectHandle);
begin
  if Assigned(Obj) and Assigned(TModule.TVMFFIObjectDecRef) then
    TModule.TVMFFIObjectDecRef(Obj);
  Obj := nil;
end;

class function TTVMRuntime.TensorDLTensor(const TensorObj: TTVMFFIObjectHandle): PDLTensor;
begin
  if not Assigned(TensorObj) then Exit(nil);
  Result := PDLTensor(PByte(TensorObj) + TVMFFIObjectPayloadOffset);
end;

const
  MxcsrFlagsBits   = $0000003F;
  MxcsrAbiStandard = $00001F80;

{$IFDEF FPC}{$PUSH}{$ASMMODE INTEL}{$ENDIF}
procedure LoadMxcsrExact(const Value: LongWord);
var
  ControlWord: LongWord;
asm
  mov     eax, Value
  mov     ControlWord, eax
  ldmxcsr ControlWord
end;
{$IFDEF FPC}{$POP}{$ENDIF}

function EnterForeignFloatAbi: LongWord;
begin
  Result := GetMXCSR;
  LoadMxcsrExact((Result and MxcsrFlagsBits) or MxcsrAbiStandard);
end;

procedure LeaveForeignFloatAbi(const Saved: LongWord);
begin
  LoadMxcsrExact(Saved);
end;

class function TTVMRuntime.CallFunc(const Func: TTVMFFIObjectHandle; const Args: array of TTVMFFIAny;
  out Res: TTVMFFIAny; const Error: PString): Boolean;
var
  N: Integer;
  ArgsPtr: PTVMFFIAny;
  SavedFloatAbi: LongWord;
  Failed: Boolean;
begin
  FillChar(Res, SizeOf(TTVMFFIAny), 0);
  Res.TypeIndex := kTVMFFINone;
  if not Assigned(Func) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.CallFunc: function handle is nil';
    Exit(False);
  end;
  if not Assigned(TModule.TVMFFIFunctionCall) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.CallFunc: TVMFFIFunctionCall not loaded';
    Exit(False);
  end;
  N := Length(Args);
  if N > 0 then
    ArgsPtr := @Args[0]
  else
    ArgsPtr := nil;
  SavedFloatAbi := EnterForeignFloatAbi;
  try
    Failed := TModule.TVMFFIFunctionCall(Func, ArgsPtr, N, @Res) <> 0;
  finally
    LeaveForeignFloatAbi(SavedFloatAbi);
  end;
  if Failed then
  begin
    if Assigned(Error) then Error^ := LastError;
    Exit(False);
  end;
  Result := True;
end;

class function TTVMRuntime.CallGlobal(const FuncName: string; const Args: array of TTVMFFIAny;
  out Res: TTVMFFIAny; const Error: PString): Boolean;
var
  Func: TTVMFFIObjectHandle;
begin
  FillChar(Res, SizeOf(TTVMFFIAny), 0);
  Res.TypeIndex := kTVMFFINone;
  Func := TModule.GetGlobalFunc(FuncName);
  if not Assigned(Func) then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.CallGlobal: global function "%s" not found', [FuncName]);
    Exit(False);
  end;
  Result := CallFunc(Func, Args, Res, Error);
end;

class function TTVMRuntime.RegisterGlobalFunc(const Name: string; const SafeCall: TModule.TTVMFFISafeCall;
  const AllowOverride: Boolean; const Error: PString): Boolean;
var
  FuncHandle: TTVMFFIObjectHandle;
  AnsiName: AnsiString;
  ByteArray: TTVMFFIByteArray;
  Ov: Int32;
begin
  if not Assigned(TModule.TVMFFIFunctionCreate) or not Assigned(TModule.TVMFFIFunctionSetGlobal) then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.RegisterGlobalFunc: Create/SetGlobal unavailable for "%s" (an old runtime?)', [Name]);
    Exit(False);
  end;
  if not Assigned(SafeCall) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.RegisterGlobalFunc: SafeCall is nil';
    Exit(False);
  end;
  FuncHandle := nil;
  if TModule.TVMFFIFunctionCreate(nil, SafeCall, nil, FuncHandle) <> 0 then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.RegisterGlobalFunc: TVMFFIFunctionCreate failed for "%s": %s',
        [Name, LastError]);
    Exit(False);
  end;
  AnsiName := AnsiString(Name);
  ByteArray.Data := PAnsiChar(AnsiName);
  ByteArray.Size := Length(AnsiName);
  if AllowOverride then
    Ov := 1
  else
    Ov := 0;
  if TModule.TVMFFIFunctionSetGlobal(@ByteArray, FuncHandle, Ov) <> 0 then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.RegisterGlobalFunc: TVMFFIFunctionSetGlobal failed for "%s": %s',
        [Name, LastError]);
    ObjectRelease(FuncHandle);
    Exit(False);
  end;
  Result := True;
end;

class function TTVMRuntime.ModuleLoadFromFile(const FileName, ModuleFormat: string;
  out ModuleHandle: TTVMFFIObjectHandle; const Error: PString): Boolean;
var
  AnsiPath: AnsiString;
  Res: TTVMFFIAny;
begin
  ModuleHandle := nil;
  if not Assigned(TModule.GModuleLoadFromFile) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ModuleLoadFromFile: global ffi.ModuleLoadFromFile not resolved';
    Exit(False);
  end;
  AnsiPath := AnsiString(FileName);
  if not CallFunc(TModule.GModuleLoadFromFile,
       [AnyCStr(PAnsiChar(AnsiPath))], Res, Error) then Exit(False);
  if Res.TypeIndex <> kTVMFFIModule then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.ModuleLoadFromFile: expected module, got type_index %d', [Res.TypeIndex]);
    Exit(False);
  end;
  ModuleHandle := Res.VObj;
  Result := True;
end;

class function TTVMRuntime.ExecutableLoadFromFile(const FileName: string;
  out ExecutableHandle: TTVMFFIObjectHandle; const Error: PString): Boolean;
var
  AnsiPath: AnsiString;
  Res: TTVMFFIAny;
begin
  ExecutableHandle := nil;
  if not Assigned(TModule.GExecutableLoadFromFile) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ExecutableLoadFromFile: global relax.ExecutableLoadFromFile not resolved';
    Exit(False);
  end;
  AnsiPath := AnsiString(FileName);
  if not CallFunc(TModule.GExecutableLoadFromFile, [AnyCStr(PAnsiChar(AnsiPath))], Res, Error) then
    Exit(False);
  if Res.TypeIndex <> kTVMFFIModule then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.ExecutableLoadFromFile: expected module, got type_index %d',
        [Res.TypeIndex]);
    Exit(False);
  end;
  ExecutableHandle := Res.VObj;
  Result := True;
end;

class function TTVMRuntime.ModuleGetFunction(const ModuleHandle: TTVMFFIObjectHandle; const Name: string;
  const QueryImports: Boolean; out FuncHandle: TTVMFFIObjectHandle; const Error: PString): Boolean;
var
  AnsiName: AnsiString;
  Res: TTVMFFIAny;
begin
  FuncHandle := nil;
  if not Assigned(TModule.GModuleGetFunction) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ModuleGetFunction: global ffi.ModuleGetFunction not resolved';
    Exit(False);
  end;
  if not Assigned(ModuleHandle) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.ModuleGetFunction: module handle is nil';
    Exit(False);
  end;
  AnsiName := AnsiString(Name);
  if not CallFunc(TModule.GModuleGetFunction,
    [AnyObject(ModuleHandle), AnyCStr(PAnsiChar(AnsiName)), AnyBool(QueryImports)], Res, Error) then
      Exit(False);
  if Res.TypeIndex <> kTVMFFIFunction then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.ModuleGetFunction: function "%s" not found (type_index %d)',
        [Name, Res.TypeIndex]);
    Exit(False);
  end;
  FuncHandle := Res.VObj;
  Result := True;
end;

class function TTVMRuntime.CopyFromHost(const Tensor: PDLTensor; const Host: Pointer;
  const NBytes: NativeUInt; const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
begin
  if not Assigned(TModule.GTensorCopyFromBytes) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CopyFromHost: global runtime.TVMTensorCopyFromBytes not resolved';
    Exit(False);
  end;
  Result := CallFunc(TModule.GTensorCopyFromBytes,
    [AnyDLTensor(Tensor), AnyPtr(Host), AnyInt(Int64(NBytes))], Res, Error);
end;

class function TTVMRuntime.CopyToHost(const Tensor: PDLTensor; const Host: Pointer; const NBytes: NativeUInt;
  const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
begin
  if not Assigned(TModule.GTensorCopyToBytes) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CopyToHost: global runtime.TVMTensorCopyToBytes not resolved';
    Exit(False);
  end;
  Result := CallFunc(TModule.GTensorCopyToBytes,
    [AnyDLTensor(Tensor), AnyPtr(Host), AnyInt(Int64(NBytes))], Res, Error);
end;

class function TTVMRuntime.AllocateTensor(const Shape: PInt64;
  const NDim, DTypeCode, DTypeBits, DTypeLanes, DeviceType, DeviceID: Integer; out Tensor: PDLTensor;
  const Error: PString): Boolean;
var
  ShapeArr: array of Int64;
  I: Integer;
begin
  Tensor := nil;
  if DeviceType <> Ord(kDLCPU) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.AllocateTensor: only CPU tensors are supported (device tensors are managed by Relax VM)';
    Exit(False);
  end;
  if (NDim < 1) or not Assigned(Shape) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.AllocateTensor: invalid shape or ndim';
    Exit(False);
  end;
  SetLength(ShapeArr, NDim);
  for I := 0 to NDim - 1 do ShapeArr[I] := Shape[I];
  Tensor := NewCpuTensor(ShapeArr, DTypeCode, DTypeBits, DTypeLanes);
  if not Assigned(Tensor) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.AllocateTensor: out of memory';
    Exit(False);
  end;
  Result := True;
end;

procedure ReleaseAdoptedTensor(Managed: PDLManagedTensorVersioned); cdecl;
begin
  if not Assigned(Managed) then
    Exit;
  if Assigned(Managed.manager_ctx) then
    FreeMem(Managed.manager_ctx);
  FreeMem(Managed);
end;

class function TTVMRuntime.AdoptCpuTensor(var Source: PDLTensor; out TensorHandle: TTVMFFIObjectHandle;
  const Error: PString): Boolean;
var
  Managed: PDLManagedTensorVersioned;
  Status: Integer;
begin
  Result := False;
  TensorHandle := nil;
  if not Assigned(Source) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.AdoptCpuTensor: source tensor is nil';
    Exit;
  end;
  if not Assigned(TModule.TVMFFITensorFromDLPackVersioned) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.AdoptCpuTensor: TVMFFITensorFromDLPackVersioned is not resolved';
    Exit;
  end;
  GetMem(Managed, SizeOf(TDLManagedTensorVersioned));
  FillChar(Managed^, SizeOf(TDLManagedTensorVersioned), 0);
  Managed.version.major := DLPACK_MAJOR_VERSION;
  Managed.version.minor := DLPACK_MINOR_VERSION;
  Managed.manager_ctx := Source;
  Managed.deleter := ReleaseAdoptedTensor;
  Managed.flags := 0;
  Managed.dl_tensor := Source^;
  Status := TModule.TVMFFITensorFromDLPackVersioned(Managed, 0, 1, TensorHandle);
  if Status <> 0 then
  begin
    FreeMem(Managed);
    TensorHandle := nil;
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.AdoptCpuTensor: TVMFFITensorFromDLPackVersioned failed with status %d', [Status]);
    Exit;
  end;
  Source := nil;
  Result := True;
end;

class function TTVMRuntime.AllocDeviceTensor(const Shape: array of Int64;
  const DTypeCode, DTypeBits, DTypeLanes, DeviceType, DeviceID: Integer;
  out TensorHandle: TTVMFFIObjectHandle; const Error: PString): Boolean;
var
  ShapeArgs: array of TTVMFFIAny;
  ShapeObj: TTVMFFIObjectHandle;
  DType: TDLDataType;
  Device: TDLDevice;
  Res: TTVMFFIAny;
  I: Integer;
begin
  Result := False;
  TensorHandle := nil;
  ShapeObj := nil;
  SetLength(ShapeArgs, Length(Shape));
  for I := 0 to High(Shape) do
    ShapeArgs[I] := AnyInt(Shape[I]);
  if not CallGlobal('ffi.Shape', ShapeArgs, Res, Error) then
    Exit;
  if Res.TypeIndex <> kTVMFFIShape then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.AllocDeviceTensor: ffi.Shape gave type_index %d', [Res.TypeIndex]);
    Exit;
  end;
  ShapeObj := Res.VObj;
  try
    DType.code := DTypeCode;
    DType.bits := DTypeBits;
    DType.lanes := DTypeLanes;
    Device.device_type := TDLDeviceType(DeviceType);
    Device.device_id := DeviceID;
    if not CallGlobal('runtime.TVMTensorAllocWithScope',
         [AnyObject(ShapeObj), AnyDataType(DType), AnyDevice(Device),
          AnyObject(nil)], Res, Error) then
        Exit;
    if Res.TypeIndex <> kTVMFFITensor then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.AllocDeviceTensor: got type_index %d, expected a tensor',
          [Res.TypeIndex]);
      Exit;
    end;
    TensorHandle := Res.VObj;
    Result := True;
  finally
    ObjectRelease(ShapeObj);
  end;
end;

class function TTVMRuntime.CopyBytesToTensor(const TensorHandle: TTVMFFIObjectHandle;
  const Source: Pointer; const ByteCount: Int64; const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
  Raw: PDLTensor;
begin
  Result := False;
  if not Assigned(TensorHandle) or not Assigned(Source) or (ByteCount <= 0) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CopyBytesToTensor: nil tensor, nil source or empty size';
    Exit;
  end;
  Raw := TensorDLTensor(TensorHandle);
  Result := CallGlobal('runtime.TVMTensorCopyFromBytes',
    [AnyDLTensor(Raw), AnyPtr(Source), AnyInt(ByteCount)], Res, Error);
end;

class function TTVMRuntime.CopyBytesFromTensor(const TensorHandle: TTVMFFIObjectHandle;
  const Target: Pointer; const ByteCount: Int64; const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
  Raw: PDLTensor;
begin
  Result := False;
  if not Assigned(TensorHandle) or not Assigned(Target) or (ByteCount <= 0) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CopyBytesFromTensor: nil tensor, nil target or empty size';
    Exit;
  end;
  Raw := TensorDLTensor(TensorHandle);
  Result := CallGlobal('runtime.TVMTensorCopyToBytes',
    [AnyDLTensor(Raw), AnyPtr(Target), AnyInt(ByteCount)], Res, Error);
end;

class function TTVMRuntime.CheckTensor(const Tensor: PDLTensor; const DTypeCodeSet: TDLDataTypeCodeSet;
  const DTypeBits, DTypeLanes: Integer): Boolean;
begin
  Result := Assigned(Tensor) and (Tensor.dtype.bits = DTypeBits) and
    (Tensor.dtype.lanes = DTypeLanes) and (TDLDataTypeCode(Tensor.dtype.code) in DTypeCodeSet);
end;

class function TTVMRuntime.ConvertImageTo8U(const Image: IMat): IMat;
var
  Scale, Shift: Double;
begin
  repeat
    if (Image.Depth = CV_32F) or (Image.Depth = CV_64F) then
    begin
      Scale := 255;
      Shift := 0;
      Break;
    end;
    if (Image.Depth = CV_8S) or (Image.Depth = CV_16S) or (Image.Depth = CV_16U) or
      (Image.Depth = CV_32S) then
      begin
        Scale := 1;
        Shift := 0;
        Break;
      end;
    Scale := 1;
    Shift := 0;
  until True;
  Result := Image.ConvertTo(CV_MAKETYPE(CV_8U, Image.Channels), Scale, Shift);
end;

class function TTVMRuntime.ConvertImageToDLTensor(const Image: IMat; const Device: TDLDevice;
  const TargetLayout: TImageTensor; out OutputTensor: PDLTensor; const DTypeCode: TDLDataTypeCode;
  const DTypeBits, DTypeLanes: Integer; const BatchSize: Int64; const Error: PString): Boolean;
var
  TypeSize: NativeInt;
  Shape: array of Int64;
  W, H, C, I, J, K: Integer;
  RowStride: NativeInt;
  SourcePtr, TargetPtr: PByte;
  RowBytes: NativeInt;
begin
  OutputTensor := nil;
  if not Assigned(Image) or Image.IsEmpty then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ConvertImageToDLTensor: Input image (IMat) is nil or empty';
    Exit(False);
  end;
  if (Image.Channels <= 0) or (Image.Width <= 0) or (Image.Height <= 0) then
  begin
    if Assigned(Error) then
      Error^ := Format(
        'TTVMRuntime.ConvertImageToDLTensor: Invalid input image properties. Channels: %d, Width: %d, Height: %d. All must be positive',
        [
          Image.Channels,
          Image.Width,
          Image.Height
        ]
      );
    Exit(False);
  end;
  case DTypeCode of
    kDLFloat, kDLInt, kDLUInt:
      case DTypeBits of
        8: TypeSize := 1;
        16: TypeSize := 2;
        32: TypeSize := 4;
        64: TypeSize := 8;
      else
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.ConvertImageToDLTensor: Unsupported DTypeBits (%d)', [DTypeBits]);
        Exit(False);
      end;
  else
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.ConvertImageToDLTensor: Unsupported DTypeCode (%d)', [Ord(DTypeCode)]);
    Exit(False);
  end;
  W := Image.Width;
  H := Image.Height;
  C := Image.Channels;
  SetLength(Shape, 4);
  Shape[TargetLayout.BIndex] := BatchSize;
  Shape[TargetLayout.WIndex] := W;
  Shape[TargetLayout.HIndex] := H;
  Shape[TargetLayout.CIndex] := C;
  OutputTensor := NewCpuTensor(Shape, Ord(DTypeCode), DTypeBits, DTypeLanes);
  if not Assigned(OutputTensor) then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.ConvertImageToDLTensor: failed to allocate CPU tensor. Shape: %s',
        [ShapeToString(GetTensorShape(OutputTensor))]);
    Exit(False);
  end;
  SourcePtr := PByte(Image.DataPtr);
  TargetPtr := OutputTensor.data;
  RowStride := Image.Step;
  if (TargetLayout.CIndex = 1) and (TargetLayout.HIndex = 2) and (TargetLayout.WIndex = 3) then
    for I := 0 to C - 1 do for J := 0 to H - 1 do for K := 0 to W - 1 do
      System.Move(
        Pointer(NativeUInt(SourcePtr) + NativeUInt(J * RowStride + K * C * TypeSize + I * TypeSize))^,
        Pointer(NativeUInt(TargetPtr) + NativeUInt((I * W * H + J * W + K) * TypeSize))^,
        TypeSize)
  else begin
    RowBytes := NativeInt(W) * C * TypeSize;
    for J := 0 to H - 1 do
      System.Move(
        Pointer(NativeUInt(SourcePtr) + NativeUInt(J * RowStride))^,
        Pointer(NativeUInt(TargetPtr) + NativeUInt(J * RowBytes))^,
        RowBytes
      );
  end;
  Result := True;
end;

class function TTVMRuntime.ConvertImageToTensorFormat(const Source: IMat; out Target: IMat;
  const Tensor: PDLTensor; const ImageTensor: TImageTensor; const Scale: Single; const Interpolation: Integer;
  const Error: PString): Boolean;

  function MakeImage(out Image: IMat): Boolean;
  var
    TargetW, TargetH: Int64;
  begin
    Image := nil;
    TargetW := Tensor.shape[ImageTensor.WIndex];
    TargetH := Tensor.shape[ImageTensor.HIndex];
    if TargetW < 1 then
      TargetW := Source.Width;
    if TargetH < 1 then
      TargetH := Source.Height;
    if (TargetW < 1) or (TargetH < 1) then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.ConvertImageToTensorFormat.MakeImage: Invalid target dimensions (W: %d, H: %d)',
          [TargetW, TargetH]);
      Exit(False);
    end;
    if (TargetW < Source.Width) or (TargetH < Source.Height) then
      Image := Source.Resize(TargetW, TargetH, INTER_AREA)
    else
      Image := Source.Resize(TargetW, TargetH, Interpolation);
    if not Assigned(Image) or Image.IsEmpty then
    begin
      if Assigned(Error) then
        Error^ := Format(
          'TTVMRuntime.ConvertImageToTensorFormat.MakeImage: Resize failed. Target Size: (%d x %d), Source Depth: %d, Source Channels: %d',
          [
            TargetW,
            TargetH,
            Source.Depth,
            Source.Channels
          ]
        );
      Exit(False);
    end;
    Image := ConvertColorChannels(Image, Tensor.shape[ImageTensor.CIndex]);
    if not Assigned(Image) or Image.IsEmpty then
    begin
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.ConvertImageToTensorFormat.MakeImage: channel conversion failed';
      Exit(False);
    end;
    Result := True;
  end;

  procedure ApplyDepth(const CVDepth: Integer);
  begin
    if Target.Depth <> CVDepth then
    begin
      Target := Target.ConvertTo(CV_MAKETYPE(CVDepth, Target.Channels), 1, 0);
      if Scale <> 1 then
        Target := Target.ConvertTo(-1, Scale, 0);
    end;
  end;

begin
  Target := nil;
  if not Assigned(Source) or Source.IsEmpty then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ConvertImageToTensorFormat: Input Source (IMat) is nil or empty';
    Exit(False);
  end;
  if not Assigned(Tensor) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ConvertImageToTensorFormat: Input Tensor (PDLTensor) is nil';
    Exit(False);
  end;
  if not Assigned(Tensor.shape) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.ConvertImageToTensorFormat: Input Tensor.shape is nil';
    Exit(False);
  end;
  if not TTVMRuntime.IsImageShape(Tensor^, ImageTensor) then
  begin
    if Assigned(Error) then
    begin
      Error^ := Format('TTVMRuntime.ConvertImageToTensorFormat: Tensor shape is not a valid image shape. Tensor ndim: %d, shape: %s',
        [Tensor.ndim, ShapeToString(GetTensorShape(Tensor))]);
    end;
    Exit(False);
  end;
  case Tensor.dtype.code of
    Ord(kDLFloat):
      case Tensor.dtype.bits of
        32:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_32F);
          end;
        64:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_64F);
          end;
      else
        if Assigned(Error) then
          Error^ := Format(
            'TTVMRuntime.ConvertImageToTensorFormat: Tensor has kDLFloat type but unsupported DTypeBits: %d (expected 32 or 64)',
            [Tensor.dtype.bits]
          );
        Exit(False);
      end;
    Ord(kDLInt):
      case Tensor.dtype.bits of
        8:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_8S);
          end;
        16:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_16S);
          end;
        32:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_32S);
          end
      else
        if Assigned(Error) then
          Error^ := Format(
            'TTVMRuntime.ConvertImageToTensorFormat: Tensor has kDLInt type but unsupported DTypeBits: %d (expected 8 or 16 or 32)',
            [Tensor.dtype.bits]
          );
        Exit(False);
      end;
    Ord(kDLUInt):
      case Tensor.dtype.bits of
        8:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_8U);
          end;
        16:
          begin
            if not MakeImage(Target) then Exit(False);
            ApplyDepth(CV_16U);
          end
      else
        if Assigned(Error) then
          Error^ := Format(
            'TTVMRuntime.ConvertImageToTensorFormat: Tensor has kDLUInt type but unsupported DTypeBits: %d (expected 8 or 16)',
            [Tensor.dtype.bits]
          );
        Exit(False);
      end;
  else
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.ConvertImageToTensorFormat: Unsupported Tensor DTypeCode: %d',
        [Tensor.dtype.code]);
    Exit(False);
  end;
  Result := Assigned(Target) and not Target.IsEmpty;
end;

class function TTVMRuntime.CreateImageArrayFromTensor(const Tensor: PDLTensor; out ImageArray: TArray<IMat>;
  const ImageTensor: TImageTensor; const Error: PString): Boolean;
type
  TSize3 = record
    W: Integer;
    H: Integer;
    C: Integer;
  end;
var
  Size3: TSize3;
  CHWLayout: Boolean;
  I, PixelsPerChannel: Integer;
  ElementTypeSize: NativeInt;
  A, B, C: TValue;
  DestPtr: PByte;
begin
  if not Assigned(Tensor) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CreateImageArrayFromTensor: Input Tensor (PDLTensor) is nil';
    Exit(False);
  end;
  if not Assigned(Tensor.data) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CreateImageArrayFromTensor: Input Tensor.data is nil';
    Exit(False);
  end;
  if not Assigned(Tensor.shape) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.CreateImageArrayFromTensor: Input Tensor.shape is nil';
    Exit(False);
  end;
  FillChar(Size3, SizeOf(TSize3), 0);
  if Tensor.ndim = 4 then
  begin
    if (ImageTensor.CIndex = 1) and (ImageTensor.HIndex = 2) and (ImageTensor.WIndex = 3) then
      CHWLayout := True
    else
      if (ImageTensor.HIndex = 1) and (ImageTensor.WIndex = 2) and (ImageTensor.CIndex = 3) then
        CHWLayout := False
      else
        CHWLayout := (Tensor.shape[1] < Tensor.shape[2]) and (Tensor.shape[1] < Tensor.shape[3]);
    Size3.W := Tensor.shape[ImageTensor.WIndex];
    Size3.H := Tensor.shape[ImageTensor.HIndex];
    Size3.C := Tensor.shape[ImageTensor.CIndex];
  end
  else if Tensor.ndim = 3 then
      if (Tensor.shape[0] < Tensor.shape[1]) and (Tensor.shape[0] < Tensor.shape[2]) then
      begin
        CHWLayout := True;
        Size3.C := Tensor.shape[0];
        Size3.H := Tensor.shape[1];
        Size3.W := Tensor.shape[2];
      end
      else if (Tensor.shape[2] < Tensor.shape[0]) and (Tensor.shape[2] < Tensor.shape[1]) then
        begin
          CHWLayout := False;
          Size3.H := Tensor.shape[0];
          Size3.W := Tensor.shape[1];
          Size3.C := Tensor.shape[2];
        end
        else begin
          if Assigned(Error) then
            Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Cannot determine CHW/HWC layout for 3D tensor. Shape: %s',
              [ShapeToString(GetTensorShape(Tensor))]);
          Exit(False);
        end
    else begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Unsupported tensor dimension count: %d. Expected 3 or 4. Shape: %s',
          [Tensor.ndim, ShapeToString(GetTensorShape(Tensor))]);
      Exit(False);
    end;
  if (Size3.W < 1) or (Size3.H < 1) or (Size3.C < 1) then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Invalid calculated dimensions (W: %d, H: %d, C: %d must be > 0)',
        [Size3.W, Size3.H, Size3.C]);
    Exit(False);
  end;
  PixelsPerChannel := Size3.W * Size3.H;
  Result := True;
  SetLength(ImageArray, Size3.C);
  try
    for I := Low(ImageArray) to High(ImageArray) do
    begin
      case Tensor.dtype.code of
        Ord(kDLFloat):
          case Tensor.dtype.bits of
            16: ElementTypeSize := SizeOf(Word);
            32: ElementTypeSize := SizeOf(Single);
            64: ElementTypeSize := SizeOf(Double);
          else
            if Assigned(Error) then
              Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Invalid DTypeBits (%d) for Float tensor for channel %d',
                [Tensor.dtype.bits, I]);
            Exit(False);
          end;
        Ord(kDLInt):
          case Tensor.dtype.bits of
            8: ElementTypeSize := SizeOf(Shortint);
            16: ElementTypeSize := SizeOf(Smallint);
            32: ElementTypeSize := SizeOf(Integer);
            64: ElementTypeSize := SizeOf(Int64);
          else
            if Assigned(Error) then
              Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Invalid DTypeBits (%d) for Int tensor for channel %d',
                [Tensor.dtype.bits, I]);
            Exit(False);
          end;
        Ord(kDLUInt):
          case Tensor.dtype.bits of
            8: ElementTypeSize := SizeOf(Byte);
            16: ElementTypeSize := SizeOf(Word);
            32: ElementTypeSize := SizeOf(LongWord);
            64: ElementTypeSize := SizeOf(UInt64);
          else
            if Assigned(Error) then
              Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Invalid DTypeBits (%d) for UInt tensor for channel %d',
                [Tensor.dtype.bits, I]);
            Exit(False);
          end;
      else
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.CreateImageArrayFromTensor: Unsupported Tensor DTypeCode: %d for channel %d',
            [Tensor.dtype.code, I]);
        Exit(False);
      end;
      case Tensor.dtype.code of
        Ord(kDLFloat):
          case Tensor.dtype.bits of
            16, 32: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_32F, 1));
            64: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_64F, 1));
          end;
        Ord(kDLInt):
          case Tensor.dtype.bits of
            8: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_8S, 1));
            16: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_16S, 1));
            32, 64: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_32S, 1));
          end;
        Ord(kDLUInt):
          case Tensor.dtype.bits of
            8: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_8U, 1));
            16: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_16U, 1));
            32, 64: ImageArray[I] := NewMat(Size3.W, Size3.H, CV_MAKETYPE(CV_32S, 1));
          end;
      end;
      if not Assigned(ImageArray[I]) or ImageArray[I].IsEmpty then
      begin
        if Assigned(Error) then
          Error^ := Format(
            'TTVMRuntime.CreateImageArrayFromTensor: NewMat failed for channel %d (W:%d, H:%d). Tensor type: code %d, bits %d',
            [
              I,
              Size3.W,
              Size3.H,
              Tensor.dtype.code,
              Tensor.dtype.bits
            ]
          );
        Exit(False);
      end;

      DestPtr := PByte(ImageArray[I].DataPtr);
      A.Reset;
      B.Reset;
      C.Reset;

      if CHWLayout then
        A.Signed8 := PShortint(NativeUInt(Tensor.data) + NativeUInt(I * PixelsPerChannel * ElementTypeSize))
      else
        A.Signed8 := PShortint(NativeUInt(Tensor.data) + NativeUInt(I * ElementTypeSize));

      case Tensor.dtype.code of
        Ord(kDLFloat):
          case Tensor.dtype.bits of
            16:
              begin
                B.Float32 := PSingle(DestPtr);
                C.Float32 := PSingle(DestPtr + PixelsPerChannel * SizeOf(Single));
                while B.Float32 < C.Float32 do
                begin
                  B.Float32^ := Float16ToSingle(A.Unsigned16^);
                  if CHWLayout then
                    Inc(A.Unsigned16)
                  else
                    Inc(A.Unsigned16, Size3.C);
                  Inc(B.Float32);
                end;
              end;
            32:
              begin
                B.Float32 := PSingle(DestPtr);
                C.Float32 := PSingle(DestPtr + PixelsPerChannel * SizeOf(Single));
                while B.Float32 < C.Float32 do
                begin
                  B.Float32^ := A.Float32^;
                  if CHWLayout then
                    Inc(A.Float32)
                  else
                    Inc(A.Float32, Size3.C);
                  Inc(B.Float32);
                end;
              end;
            64:
              begin
                B.Float64 := PDouble(DestPtr);
                C.Float64 := PDouble(DestPtr + PixelsPerChannel * ElementTypeSize);
                while B.Float64 < C.Float64 do
                begin
                  B.Float64^ := A.Float64^;
                  if CHWLayout then
                    Inc(A.Float64)
                  else
                    Inc(A.Float64, Size3.C);
                  Inc(B.Float64);
                end;
              end;
          end;
        Ord(kDLInt):
          case Tensor.dtype.bits of
            8:
              begin
                B.Signed8 := PShortint(DestPtr);
                C.Signed8 := PShortint(DestPtr + PixelsPerChannel * ElementTypeSize);
                while B.Signed8 < C.Signed8 do
                begin
                  B.Signed8^ := A.Signed8^;
                  if CHWLayout then
                    Inc(A.Signed8)
                  else
                    Inc(A.Signed8, Size3.C);
                  Inc(B.Signed8);
                end;
              end;
            16:
              begin
                B.Signed16 := PSmallint(DestPtr);
                C.Signed16 := PSmallint(DestPtr + PixelsPerChannel * ElementTypeSize);
                while B.Signed16 < C.Signed16 do
                begin
                  B.Signed16^ := A.Signed16^;
                  if CHWLayout then
                    Inc(A.Signed16)
                  else
                    Inc(A.Signed16, Size3.C);
                  Inc(B.Signed16);
                end;
              end;
            32:
              begin
                B.Signed32 := PInteger(DestPtr);
                C.Signed32 := PInteger(DestPtr + PixelsPerChannel * SizeOf(Integer));
                while B.Signed32 < C.Signed32 do
                begin
                  B.Signed32^ := A.Signed32^;
                  if CHWLayout then
                    Inc(A.Signed32)
                  else
                    Inc(A.Signed32, Size3.C);
                  Inc(B.Signed32);
                end;
              end;
            64:
              begin
                B.Signed32 := PInteger(DestPtr);
                C.Signed32 := PInteger(DestPtr + PixelsPerChannel * SizeOf(Integer));
                while B.Signed32 < C.Signed32 do
                begin
                  B.Signed32^ := A.Signed64^;
                  if CHWLayout then
                    Inc(A.Signed64)
                  else
                    Inc(A.Signed64, Size3.C);
                  Inc(B.Signed32);
                end;
              end;
          end;
        Ord(kDLUInt):
          case Tensor.dtype.bits of
            8:
              begin
                B.Unsigned8 := PByte(DestPtr);
                C.Unsigned8 := PByte(DestPtr + PixelsPerChannel * ElementTypeSize);
                while B.Unsigned8 < C.Unsigned8 do
                begin
                  B.Unsigned8^ := A.Unsigned8^;
                  if CHWLayout then
                    Inc(A.Unsigned8)
                  else
                    Inc(A.Unsigned8, Size3.C);
                  Inc(B.Unsigned8);
                end;
              end;
            16:
              begin
                B.Unsigned16 := PWord(DestPtr);
                C.Unsigned16 := PWord(DestPtr + PixelsPerChannel * ElementTypeSize);
                while B.Unsigned16 < C.Unsigned16 do
                begin
                  B.Unsigned16^ := A.Unsigned16^;
                  if CHWLayout then
                    Inc(A.Unsigned16)
                  else
                    Inc(A.Unsigned16, Size3.C);
                  Inc(B.Unsigned16);
                end;
              end;
            32:
              begin
                B.Signed32 := PInteger(DestPtr);
                C.Signed32 := PInteger(DestPtr + PixelsPerChannel * SizeOf(Integer));
                while B.Signed32 < C.Signed32 do
                begin
                  B.Signed32^ := Integer(A.Unsigned32^);
                  if CHWLayout then
                    Inc(A.Unsigned32)
                  else
                    Inc(A.Unsigned32, Size3.C);
                  Inc(B.Signed32);
                end;
              end;
            64:
              begin
                B.Signed32 := PInteger(DestPtr);
                C.Signed32 := PInteger(DestPtr + PixelsPerChannel * SizeOf(Integer));
                while B.Signed32 < C.Signed32 do
                begin
                  B.Signed32^ := Integer(A.Unsigned64^);
                  if CHWLayout then
                    Inc(A.Unsigned64)
                  else
                    Inc(A.Unsigned64, Size3.C);
                  Inc(B.Signed32);
                end;
              end;
          end;
      end;
    end;
  finally
    if not Result then
      ImageArray := nil;
  end;
end;

class procedure TTVMRuntime.FreeModel(var Descriptor: TModelDescriptor);
var
  I: Integer;
begin
  Descriptor.InferenceObserver := nil;
  Descriptor.InferenceObserverContext := nil;
  for I := 0 to High(Descriptor.OutputHandles) do
    ObjectRelease(Descriptor.OutputHandles[I]);
  Descriptor.OutputHandles := nil;
  ObjectRelease(Descriptor.GetOutputFunc);
  ObjectRelease(Descriptor.GetOutputArityFunc);
  ObjectRelease(Descriptor.RunFunc);
  ObjectRelease(Descriptor.SetInputFunc);
  ObjectRelease(Descriptor.InitFunc);
  ObjectRelease(Descriptor.VMHandle);
  ObjectRelease(Descriptor.ExecutableHandle);
  if Assigned(Descriptor.InputSpec) then
  begin
    FreeMem(Descriptor.InputSpec);
    Descriptor.InputSpec := nil;
  end;
  Descriptor.InputShapes := nil;
  Descriptor.InputNames := nil;
  Descriptor.EntryName := '';
  FillChar(Descriptor.Device, SizeOf(Descriptor.Device), 0);
end;

class procedure TTVMRuntime.FreeTensors(var Tensors: TArray<PDLTensor>);
var
  I: Integer;
begin
  for I := 0 to High(Tensors) do
    if Assigned(Tensors[I]) then FreeMem(Tensors[I]);
  Tensors := nil;
end;

class function TTVMRuntime.GetInputTensor(const Descriptor: TModelDescriptor; const TensorIndex: Int64;
  out InputTensor: PDLTensor; const Error: PString): Boolean;
begin
  InputTensor := Descriptor.InputSpec;
  if not Assigned(InputTensor) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.GetInputTensor: input spec is not set (call TTVMRuntime.SetInputSpec after LoadModel)';
    Exit(False);
  end;
  Result := True;
end;

class procedure TTVMRuntime.SetInputSpec(var Descriptor: TModelDescriptor; const Shape: array of Int64;
  const DTypeCode: TDLDataTypeCode; const DTypeBits, DTypeLanes: Integer);
begin
  if Assigned(Descriptor.InputSpec) then
  begin
    FreeMem(Descriptor.InputSpec);
    Descriptor.InputSpec := nil;
  end;
  Descriptor.InputSpec := NewCpuTensor(Shape, Ord(DTypeCode), DTypeBits, DTypeLanes);
end;

function ReadInputNamesBeside(const ModelFileName: string): TArray<string>;
var
  SidecarPath: string;
  Lines: TStringList;
  I, P1, P2: Integer;
  S, InputName: string;
begin
  Result := nil;
  SidecarPath := ModelFileName + '.inputs';
  if not FileExists(SidecarPath) then
    Exit;
  Lines := TStringList.Create;
  try
    try
      Lines.LoadFromFile(SidecarPath, TEncoding.UTF8);
    except
      Exit;
    end;
    for I := 0 to Lines.Count - 1 do
    begin
      S := Lines[I];
      P1 := Pos(#9, S);
      if P1 < 1 then
        Continue;
      P2 := Pos(#9, S, P1 + 1);
      if P2 < 1 then
        InputName := Copy(S, P1 + 1, MaxInt)
      else
        InputName := Copy(S, P1 + 1, P2 - P1 - 1);
      if InputName <> '' then
      begin
        SetLength(Result, Length(Result) + 1);
        Result[High(Result)] := InputName;
      end;
    end;
  finally
    Lines.Free;
  end;
end;

function ReadInputShapesBeside(const ModelFileName: string;
  const ExpectedNames: TArray<string>): TArray<TArray<Int64>>;
var
  Lines, Dimensions: TStringList;
  S, Name: string;
  I, J, P1, P2: Integer;
  Value: Int64;
begin
  Result := nil;
  if Length(ExpectedNames) < 2 then Exit;
  if not FileExists(ModelFileName + '.inputs') then Exit;
  Lines := TStringList.Create;
  Dimensions := TStringList.Create;
  try
    try
      Lines.LoadFromFile(ModelFileName + '.inputs', TEncoding.UTF8);
      if Lines.Count <> Length(ExpectedNames) then Exit;
      Dimensions.Delimiter := ',';
      Dimensions.StrictDelimiter := True;
      SetLength(Result, Lines.Count);
      for I := 0 to Lines.Count - 1 do
      begin
        S := Lines[I];
        P1 := Pos(#9, S);
        if P1 < 2 then Exit(nil);
        P2 := Pos(#9, S, P1 + 1);
        if P2 <= P1 + 1 then Exit(nil);
        if StrToIntDef(Copy(S, 1, P1 - 1), -1) <> I then Exit(nil);
        Name := Copy(S, P1 + 1, P2 - P1 - 1);
        if (Name = '') or (Name <> ExpectedNames[I]) then Exit(nil);
        Dimensions.DelimitedText := Copy(S, P2 + 1, MaxInt);
        if (Dimensions.Count < 1) or (Dimensions.Count > 8) then Exit(nil);
        SetLength(Result[I], Dimensions.Count);
        for J := 0 to Dimensions.Count - 1 do
        begin
          if not TryStrToInt64(Dimensions[J], Value) or (Value < 1) then
            Exit(nil);
          Result[I][J] := Value;
        end;
      end;
    except
      on E: Exception do Result := nil;
    end;
  finally
    Dimensions.Free;
    Lines.Free;
  end;
end;

class function TTVMRuntime.GetInputTensorIndex(const Descriptor: TModelDescriptor; const TensorName: string;
  out TensorIndex: Int64; const Error: PString): Boolean;
var
  I: Integer;
  KnownNames: string;
begin
  TensorIndex := 0;
  if Length(Descriptor.InputNames) = 0 then
    Exit(True);
  for I := 0 to High(Descriptor.InputNames) do
    if Descriptor.InputNames[I] = TensorName then
    begin
      TensorIndex := I;
      Exit(True);
    end;
  if Assigned(Error) then
  begin
    KnownNames := '';
    for I := 0 to High(Descriptor.InputNames) do
    begin
      if KnownNames <> '' then
        KnownNames := KnownNames + ', ';
      KnownNames := KnownNames + Descriptor.InputNames[I];
    end;
    Error^ := Format('TTVMRuntime.GetInputTensorIndex: model has no input named "%s"; it has: %s',
      [TensorName, KnownNames]);
  end;
  Result := False;
end;

class function TTVMRuntime.GetOrCreateCpuTensor(const SrcTensor: PDLTensor; out CpuTensor: PDLTensor;
  const TensorName, ContextPrefix: string; const Error: PString): PDLTensor;
var
  Shape: array of Int64;
  I: Integer;
  NBytes: NativeUInt;
begin
  CpuTensor := nil;
  if not Assigned(SrcTensor) then
  begin
    if Assigned(Error) then
      Error^ := Format('%s: source tensor "%s" is nil', [ContextPrefix, TensorName]);
    Exit(nil);
  end;
  if SrcTensor.device.device_type = kDLCPU then
    Exit(SrcTensor);
  SetLength(Shape, SrcTensor.ndim);
  for I := 0 to SrcTensor.ndim - 1 do Shape[I] := SrcTensor.shape[I];
  CpuTensor := NewCpuTensor(Shape, SrcTensor.dtype.code, SrcTensor.dtype.bits, SrcTensor.dtype.lanes);
  if not Assigned(CpuTensor) then
  begin
    if Assigned(Error) then
      Error^ := Format('%s: failed to allocate CPU copy for "%s"', [ContextPrefix, TensorName]);
    Exit(nil);
  end;
  NBytes := DLTensorDataSize(CpuTensor);
  if not CopyToHost(SrcTensor, CpuTensor.data, NBytes, Error) then
  begin
    FreeMem(CpuTensor);
    CpuTensor := nil;
    Exit(nil);
  end;
  Result := CpuTensor;
end;

class function TTVMRuntime.GetOutputTensor(const Descriptor: TModelDescriptor; const TensorIndex: Int64;
  out OutputTensor: PDLTensor; const Error: PString): Boolean;
begin
  OutputTensor := nil;
  if (TensorIndex < 0) or (TensorIndex > High(Descriptor.OutputHandles)) then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.GetOutputTensor: index %d out of range (have %d outputs)',
        [TensorIndex, Length(Descriptor.OutputHandles)]);
    Exit(False);
  end;
  OutputTensor := TensorDLTensor(Descriptor.OutputHandles[TensorIndex]);
  Result := Assigned(OutputTensor);
end;

class function TTVMRuntime.GetTensorShape(const Tensor: PDLTensor): TArray<Int64>;
begin
  if not Assigned(Tensor) then Exit(nil);
  Result := CreateShapeArray(Tensor.shape, Tensor.ndim);
end;

class function TTVMRuntime.InitializeOutputTensors(const Descriptor: TModelDescriptor;
  out OutputTensors: TArray<PDLTensor>; const DeviceType, DeviceID: Integer; const Error: PString): Boolean;
begin
  OutputTensors := nil;
  Result := True;
end;

class function TTVMRuntime.IsImageShape(const Tensor: TDLTensor; const ImageTensor: TImageTensor): Boolean;
begin
  Result := (Tensor.ndim = 4) and (Tensor.shape[ImageTensor.CIndex] in [1, 3, 4]);
end;

class function TTVMRuntime.LoadModel(const ModelFileName, ModelFormat: string;
  out Descriptor: TModelDescriptor; const DeviceType: TDLDeviceType; const DeviceID: Integer;
  const EntryName: string; const Error: PString): Boolean;
var
  VMLoadFunc: TTVMFFIObjectHandle;
  VMRes, InitRes: TTVMFFIAny;
begin
  Result := False;
  Descriptor.InferenceObserver := nil;
  Descriptor.InferenceObserverContext := nil;
  Descriptor.ExecutableHandle := nil;
  Descriptor.VMHandle := nil;
  Descriptor.InitFunc := nil;
  Descriptor.SetInputFunc := nil;
  Descriptor.RunFunc := nil;
  Descriptor.GetOutputArityFunc := nil;
  Descriptor.GetOutputFunc := nil;
  Descriptor.OutputHandles := nil;
  Descriptor.InputSpec := nil;
  Descriptor.InputShapes := nil;
  Descriptor.Device.device_type := DeviceType;
  Descriptor.Device.device_id := DeviceID;
  Descriptor.EntryName := EntryName;
  Descriptor.InputNames := ReadInputNamesBeside(ModelFileName);
  Descriptor.InputShapes := ReadInputShapesBeside(ModelFileName, Descriptor.InputNames);
  try
    if not ModuleLoadFromFile(ModelFileName, ModelFormat, Descriptor.ExecutableHandle, Error) then
      Exit(False);
    if not ModuleGetFunction(Descriptor.ExecutableHandle, 'vm_load_executable', True, VMLoadFunc, Error) then
      Exit(False);
    try
      if not CallFunc(VMLoadFunc, [], VMRes, Error) then Exit(False);
    finally
      ObjectRelease(VMLoadFunc);
    end;
    if VMRes.TypeIndex <> kTVMFFIModule then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.LoadModel: vm_load_executable returned type_index %d (expected module)',
          [VMRes.TypeIndex]);
      Exit(False);
    end;
    Descriptor.VMHandle := VMRes.VObj;
    if not ModuleGetFunction(Descriptor.VMHandle, 'vm_initialization', True, Descriptor.InitFunc, Error) then
      Exit(False);
    if not ModuleGetFunction(Descriptor.VMHandle, 'set_input', True, Descriptor.SetInputFunc, Error) then
      Exit(False);
    if not ModuleGetFunction(Descriptor.VMHandle, 'invoke_stateful', True, Descriptor.RunFunc, Error) then
      Exit(False);
    if not ModuleGetFunction(Descriptor.VMHandle, 'get_output_arity', True,
      Descriptor.GetOutputArityFunc, Error) then
        Exit(False);
    if not ModuleGetFunction(Descriptor.VMHandle, 'get_output', True, Descriptor.GetOutputFunc, Error) then
      Exit(False);
    if not CallFunc(Descriptor.InitFunc,
      [AnyInt(Ord(DeviceType)), AnyInt(DeviceID), AnyInt(kTVMVMAllocPooled)], InitRes, Error) then
        Exit(False);
    Result := True;
  finally
    if not Result then
      FreeModel(Descriptor);
  end;
end;

class function TTVMRuntime.TensorsAreFinite(const Tensors: TArray<PDLTensor>; out Examined: Int64;
  const Error: PString): Boolean;

  function Compact(const T: PDLTensor): Boolean;
  var
    D: Integer;
    Expected: Int64;
  begin
    if T.strides = nil then Exit(True);
    Expected := 1;
    for D := T.ndim - 1 downto 0 do
    begin
      if T.strides[D] <> Expected then Exit(False);
      Expected := Expected * T.shape[D];
    end;
    Result := True;
  end;

var
  I, D: Integer;
  K, Count: Int64;
  T: PDLTensor;
  P16: PWord;
  P32: PLongWord;
  P64: PUInt64;
  Bits64: UInt64;
  Bits32: LongWord;
  Bits16: Word;
  Bad: Boolean;
  Kind: string;
begin
  Examined := 0;
  for I := 0 to High(Tensors) do
  begin
    T := Tensors[I];
    if not Assigned(T) then Continue;
    if not (T.device.device_type in [kDLCPU, kDLCUDAHost]) then Continue;
    if T.dtype.lanes <> 1 then Continue;
    if not Assigned(T.data) then Continue;
    if not Compact(T) then Continue;
    Count := 1;
    for D := 0 to T.ndim - 1 do Count := Count * T.shape[D];
    if Count <= 0 then Continue;
    Bad := False;
    Kind := '';
    K := 0;
    if (T.dtype.code = Ord(kDLFloat)) and (T.dtype.bits = 32) then
    begin
      P32 := PLongWord(PByte(T.data) + T.byte_offset);
      while K < Count do
      begin
        Bits32 := P32[K];
        if (Bits32 and $7F800000) = $7F800000 then
        begin
          Bad := True;
          if (Bits32 and $007FFFFF) <> 0 then
            Kind := 'NaN'
          else
            Kind := 'infinity';
          Break;
        end;
        Inc(K);
      end;
    end
    else if (T.dtype.code = Ord(kDLFloat)) and (T.dtype.bits = 64) then
      begin
        P64 := PUInt64(PByte(T.data) + T.byte_offset);
        while K < Count do
        begin
          Bits64 := P64[K];
          if (Bits64 and UInt64($7FF0000000000000)) = UInt64($7FF0000000000000) then
          begin
            Bad := True;
            if (Bits64 and UInt64($000FFFFFFFFFFFFF)) <> 0 then
              Kind := 'NaN'
            else
              Kind := 'infinity';
            Break;
          end;
          Inc(K);
        end;
      end
      else if (T.dtype.code = Ord(kDLFloat)) and (T.dtype.bits = 16) then
        begin
          P16 := PWord(PByte(T.data) + T.byte_offset);
          while K < Count do
          begin
            Bits16 := P16[K];
            if (Bits16 and $7C00) = $7C00 then
            begin
              Bad := True;
              if (Bits16 and $03FF) <> 0 then
                Kind := 'NaN'
              else
                Kind := 'infinity';
              Break;
            end;
            Inc(K);
          end;
        end
        else if (T.dtype.code = Ord(kDLBfloat)) and (T.dtype.bits = 16) then
          begin
            P16 := PWord(PByte(T.data) + T.byte_offset);
            while K < Count do
            begin
              Bits16 := P16[K];
              if (Bits16 and $7F80) = $7F80 then
              begin
                Bad := True;
                if (Bits16 and $007F) <> 0 then
                  Kind := 'NaN'
                else
                  Kind := 'infinity';
                Break;
              end;
              Inc(K);
            end;
          end
          else
            Continue;
    Examined := Examined + K;
    if Bad then
    begin
      if Assigned(Error) then
        Error^ := Format(
          'TTVMRuntime.TensorsAreFinite: output %d holds %s at element %d of %d ' +
          '(the foreign computer returned a value that is not a number)',
          [
            I,
            Kind,
            K,
            Count
          ]
        );
      Exit(False);
    end;
  end;
  Result := True;
end;

class function TTVMRuntime.RunInference(var Descriptor: TModelDescriptor; const InputTensorName: string;
  const InputTensor: PDLTensor; var OutputTensors: TArray<PDLTensor>; const Error: PString): Boolean;
var
  AnsiEntry: AnsiString;
  EntryAny, Res: TTVMFFIAny;
  Arity, I: Integer;
  Examined: Int64;
begin
  OutputTensors := nil;
  if not Assigned(Descriptor.SetInputFunc) or not Assigned(Descriptor.RunFunc) or not Assigned(Descriptor.GetOutputArityFunc) or
    not Assigned(Descriptor.GetOutputFunc) then
    begin
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInference: model is not loaded (VM method handles are nil)';
      Exit(False);
    end;
  if not Assigned(InputTensor) then
  begin
    if Assigned(Error) then Error^ := 'TTVMRuntime.RunInference: InputTensor is nil';
    Exit(False);
  end;
  for I := 0 to High(Descriptor.OutputHandles) do
    ObjectRelease(Descriptor.OutputHandles[I]);
  Descriptor.OutputHandles := nil;
  AnsiEntry := AnsiString(Descriptor.EntryName);
  EntryAny := AnyCStr(PAnsiChar(AnsiEntry));
  if not CallFunc(Descriptor.SetInputFunc, [EntryAny, AnyDLTensor(InputTensor)], Res, Error) then
    Exit(False);
  if not CallFunc(Descriptor.RunFunc, [EntryAny], Res, Error) then Exit(False);
  if not CallFunc(Descriptor.GetOutputArityFunc, [EntryAny], Res, Error) then Exit(False);
  if Res.TypeIndex = kTVMFFIInt then
    Arity := Res.VInt64
  else
    Arity := -1;
  if Arity <= 0 then
  begin
    if not CallFunc(Descriptor.GetOutputFunc, [EntryAny], Res, Error) then Exit(False);
    if Res.TypeIndex <> kTVMFFITensor then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.RunInference: single output is not a tensor (type_index %d)',
          [Res.TypeIndex]);
      Exit(False);
    end;
    SetLength(Descriptor.OutputHandles, 1);
    Descriptor.OutputHandles[0] := Res.VObj;
    SetLength(OutputTensors, 1);
    OutputTensors[0] := TensorDLTensor(Res.VObj);
  end
  else begin
    SetLength(Descriptor.OutputHandles, Arity);
    SetLength(OutputTensors, Arity);
    for I := 0 to Arity - 1 do
    begin
      if not CallFunc(Descriptor.GetOutputFunc, [EntryAny, AnyInt(I)], Res, Error) then Exit(False);
      if Res.TypeIndex <> kTVMFFITensor then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInference: output %d is not a tensor (type_index %d)',
            [I, Res.TypeIndex]);
        Exit(False);
      end;
      Descriptor.OutputHandles[I] := Res.VObj;
      OutputTensors[I] := TensorDLTensor(Res.VObj);
    end;
  end;
  if not TensorsAreFinite(OutputTensors, Examined, Error) then Exit(False);
  if Assigned(Descriptor.InferenceObserver) and
    not Descriptor.InferenceObserver(Descriptor.InferenceObserverContext,
      InputTensor, OutputTensors, Examined, Error) then Exit(False);
  Result := True;
end;

class function TTVMRuntime.RunInferenceMulti(var Descriptor: TModelDescriptor;
  const Inputs: array of TTVMInputBinding; var OutputTensors: TArray<PDLTensor>;
  const Error: PString): Boolean;
const
  MaxMultiInputs = 16;
  MaxMultiOutputs = 64;
var
  Args: TArray<TTVMFFIAny>;
  RawInputs: TArray<PDLTensor>;
  AnsiEntry: AnsiString;
  EntryAny, Res: TTVMFFIAny;
  Tensor: PDLTensor;
  I, J, K, Arity: Integer;
  Stride, Elements: Int64;
  ElementBytes: Integer;
  Examined: Int64;
  HasFloat, SingleOutput: Boolean;
  WrongHandle: TTVMFFIObjectHandle;
begin
  Result := False;
  OutputTensors := nil;
  if Length(Descriptor.OutputHandles) > 0 then
  begin
    if not Assigned(TModule.TVMFFIObjectDecRef) then
    begin
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInferenceMulti: cannot release previous outputs (FFI decref unavailable)';
      Exit;
    end;
    for I := 0 to High(Descriptor.OutputHandles) do
      ObjectRelease(Descriptor.OutputHandles[I]);
    Descriptor.OutputHandles := nil;
  end;
  if not Assigned(Descriptor.SetInputFunc) or not Assigned(Descriptor.RunFunc) or not Assigned(Descriptor.GetOutputArityFunc) or
    not Assigned(Descriptor.GetOutputFunc) or not Assigned(TModule.TVMFFIFunctionCall) or
    not Assigned(TModule.TVMFFIObjectDecRef) then
    begin
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInferenceMulti: VM/FFI functions are unavailable';
      Exit;
    end;
  if (Length(Inputs) < 2) or (Length(Inputs) > MaxMultiInputs) then
  begin
    if Assigned(Error) then
      Error^ := Format('TTVMRuntime.RunInferenceMulti: expected 2..%d inputs, got %d',
        [MaxMultiInputs, Length(Inputs)]);
    Exit;
  end;
  if (Length(Descriptor.InputNames) <> Length(Inputs)) or
    (Length(Descriptor.InputShapes) <> Length(Inputs)) then
    begin
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInferenceMulti: missing or inconsistent .inputs names/shapes sidecar';
      Exit;
    end;
  if Descriptor.EntryName = '' then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.RunInferenceMulti: empty VM entry name';
    Exit;
  end;
  SetLength(Args, Length(Inputs) + 1);
  SetLength(RawInputs, Length(Inputs));
  AnsiEntry := AnsiString(Descriptor.EntryName);
  EntryAny := AnyCStr(PAnsiChar(AnsiEntry));
  Args[0] := EntryAny;
  HasFloat := False;
  for I := 0 to High(Inputs) do
  begin
    if (Inputs[I].Name = '') or (Inputs[I].Name <> Descriptor.InputNames[I]) then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d name %s does not match .inputs (%s)',
          [I, Inputs[I].Name, Descriptor.InputNames[I]]);
      Exit;
    end;
    for K := 0 to I - 1 do
      if Inputs[K].Name = Inputs[I].Name then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: duplicate input name %s',
            [Inputs[I].Name]);
        Exit;
      end;
    Tensor := Inputs[I].Tensor;
    if not Assigned(Tensor) or not Assigned(Tensor.data) or not Assigned(Tensor.shape) then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d has missing DLTensor/data/shape', [I]);
      Exit;
    end;
    if (Tensor.device.device_type <> kDLCPU) or
      (Tensor.device.device_id <> 0) or (Tensor.byte_offset <> 0) then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d must be a CPU tensor with zero byte_offset', [I]);
        Exit;
      end;
    if not (((Inputs[I].DTypeCode = kDLFloat) and
      (Inputs[I].DTypeBits = 32)) or
      ((Inputs[I].DTypeCode = kDLInt) and
      (Inputs[I].DTypeBits = 64))) then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d contract dtype must be float32 or int64', [I]);
        Exit;
      end;
    if (Tensor.dtype.code <> Ord(Inputs[I].DTypeCode)) or
      (Tensor.dtype.bits <> Inputs[I].DTypeBits) or
      (Tensor.dtype.lanes <> 1) then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d dtype differs from contract (code %d, bits %d, lanes %d)',
            [I, Tensor.dtype.code, Tensor.dtype.bits, Tensor.dtype.lanes]);
        Exit;
      end;
    if (Tensor.ndim < 1) or (Tensor.ndim > 8) or
      (Length(Descriptor.InputShapes[I]) <> Tensor.ndim) then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d rank differs from .inputs', [I]);
        Exit;
      end;
    Elements := 1;
    Stride := 1;
    for J := Tensor.ndim - 1 downto 0 do
    begin
      if (Tensor.shape[J] < 1) or
        (Tensor.shape[J] <> Descriptor.InputShapes[I][J]) or
        (Stride > High(Int64) div Tensor.shape[J]) then
        begin
          if Assigned(Error) then
            Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d shape differs from .inputs or overflows', [I]);
          Exit;
        end;
      if Assigned(Tensor.strides) and (Tensor.strides[J] <> Stride) then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d is not contiguous row-major', [I]);
        Exit;
      end;
      Stride := Stride * Tensor.shape[J];
      Elements := Stride;
    end;
    ElementBytes := Inputs[I].DTypeBits div 8;
    if Elements > High(NativeInt) div ElementBytes then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.RunInferenceMulti: input %d byte size overflows', [I]);
      Exit;
    end;
    if Inputs[I].DTypeCode = kDLFloat then HasFloat := True;
    RawInputs[I] := Tensor;
    Args[I + 1] := AnyDLTensor(Tensor);
  end;
  if not TensorsAreFinite(RawInputs, Examined, Error) then Exit;
  if HasFloat and (Examined = 0) then
  begin
    if Assigned(Error) then
      Error^ := 'TTVMRuntime.RunInferenceMulti: no float input elements were examined';
    Exit;
  end;
  try
    if not CallFunc(Descriptor.SetInputFunc, Args, Res, Error) then Exit;
    if Res.TypeIndex <> kTVMFFINone then
    begin
      if Res.TypeIndex >= kTVMFFIStaticObjectBegin then
      begin
        WrongHandle := Res.VObj;
        ObjectRelease(WrongHandle);
      end;
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInferenceMulti: set_input must return None';
      Exit;
    end;
    if not CallFunc(Descriptor.RunFunc, [EntryAny], Res, Error) then Exit;
    if Res.TypeIndex <> kTVMFFINone then
    begin
      if Res.TypeIndex >= kTVMFFIStaticObjectBegin then
      begin
        WrongHandle := Res.VObj;
        ObjectRelease(WrongHandle);
      end;
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInferenceMulti: invoke_stateful must return None';
      Exit;
    end;
    if not CallFunc(Descriptor.GetOutputArityFunc, [EntryAny], Res, Error) then Exit;
    if Res.TypeIndex <> kTVMFFIInt then
    begin
      if Res.TypeIndex >= kTVMFFIStaticObjectBegin then
      begin
        WrongHandle := Res.VObj;
        ObjectRelease(WrongHandle);
      end;
      if Assigned(Error) then
        Error^ := 'TTVMRuntime.RunInferenceMulti: VM output arity is not an integer';
      Exit;
    end;
    if (Res.VInt64 <> -1) and ((Res.VInt64 < 1) or (Res.VInt64 > MaxMultiOutputs)) then
    begin
      if Assigned(Error) then
        Error^ := Format('TTVMRuntime.RunInferenceMulti: invalid output count %d', [Res.VInt64]);
      Exit;
    end;
    SingleOutput := Res.VInt64 = -1;
    Arity := Integer(Res.VInt64);
    if SingleOutput then Arity := 1;
    SetLength(Descriptor.OutputHandles, Arity);
    SetLength(OutputTensors, Arity);
    for I := 0 to Arity - 1 do
    begin
      if SingleOutput then
      begin
        if not CallFunc(Descriptor.GetOutputFunc, [EntryAny], Res, Error) then Exit;
      end
      else if not CallFunc(Descriptor.GetOutputFunc, [EntryAny, AnyInt(I)], Res, Error) then
          Exit;
      if (Res.TypeIndex <> kTVMFFITensor) or not Assigned(Res.VObj) then
      begin
        if (Res.TypeIndex >= kTVMFFIStaticObjectBegin) and Assigned(Res.VObj) then
        begin
          WrongHandle := Res.VObj;
          ObjectRelease(WrongHandle);
        end;
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: output %d is not a tensor object', [I]);
        Exit;
      end;
      Descriptor.OutputHandles[I] := Res.VObj;
      OutputTensors[I] := TensorDLTensor(Res.VObj);
      if not Assigned(OutputTensors[I]) then
      begin
        if Assigned(Error) then
          Error^ := Format('TTVMRuntime.RunInferenceMulti: output %d has no DLTensor', [I]);
        Exit;
      end;
    end;
    if not TensorsAreFinite(OutputTensors, Examined, Error) then Exit;
    Result := True;
  finally
    if not Result then
    begin
      for I := 0 to High(Descriptor.OutputHandles) do
        ObjectRelease(Descriptor.OutputHandles[I]);
      Descriptor.OutputHandles := nil;
      OutputTensors := nil;
    end;
  end;
end;

procedure Initialize;
var
  {$IFDEF MSWINDOWS}Path,{$ENDIF}Candidate, S: string;
  Candidates: array[0..1] of string;
  I: Integer;

  function TryLoadFrom(const Dir: string): Boolean;
  begin
    Result := False;
    if not FileExists(Dir + TTVMRuntime.TModule.RuntimeName) then Exit;
    {$IFDEF MSWINDOWS}
    SetCurrentDir(Dir);
    {$ENDIF}
    TTVMRuntime.TModule.RuntimeHandle := LoadLibrary({$IFDEF FPC}Dir + TTVMRuntime.TModule.RuntimeName{$ELSE}PChar(Dir +
      TTVMRuntime.TModule.RuntimeName){$ENDIF});
    if TTVMRuntime.TModule.RuntimeHandle = 0 then Exit;
    TTVMRuntime.TModule.Handle := LoadLibrary({$IFDEF FPC}Dir + TTVMRuntime.TModule.FFIName{$ELSE}PChar(Dir +
      TTVMRuntime.TModule.FFIName){$ENDIF});
    if TTVMRuntime.TModule.Handle = 0 then
      TTVMRuntime.TModule.Handle := TTVMRuntime.TModule.RuntimeHandle;
    TTVMRuntime.TModule.LibraryPath := Dir;
    TTVMRuntime.TModule.LibraryName := Dir + TTVMRuntime.TModule.RuntimeName;
    TTVMBackend.TModule.Handle := TTVMRuntime.TModule.RuntimeHandle;
    Result := True;
  end;

begin
  {$IFDEF MSWINDOWS}
  Path := GetCurrentDir;
  {$ENDIF}
  FillChar(Candidates, SizeOf(Candidates), 0);
  Candidates[0] := ExtractFilePath(ParamStr(0));
  Candidates[1] := IncludeTrailingPathDelimiter(GetCurrentDir);
  for I := Low(Candidates) to High(Candidates) do
  begin
    if TryLoadFrom(Candidates[I] + TTVMRuntime.TModule.Path) then Break;
    if TryLoadFrom(Candidates[I]) then Break;
  end;
  if TTVMRuntime.TModule.RuntimeHandle = 0 then
  begin
    Candidate := GetEnvironmentVariable('TVM_RUNTIME_LIB');
    if Candidate <> '' then
    begin
      Candidate := IncludeTrailingPathDelimiter(ExtractFilePath(Candidate));
      TryLoadFrom(Candidate);
    end;
  end;
  if TTVMRuntime.TModule.RuntimeHandle = 0 then
  begin
    TTVMRuntime.TModule.RuntimeHandle := LoadLibrary(TTVMRuntime.TModule.RuntimeName);
    if TTVMRuntime.TModule.RuntimeHandle <> 0 then
    begin
      TTVMRuntime.TModule.Handle := LoadLibrary(TTVMRuntime.TModule.FFIName);
      if TTVMRuntime.TModule.Handle = 0 then
        TTVMRuntime.TModule.Handle := TTVMRuntime.TModule.RuntimeHandle;
      TTVMRuntime.TModule.LibraryPath := '<System Path>';
      TTVMRuntime.TModule.LibraryName := TTVMRuntime.TModule.RuntimeName;
      TTVMBackend.TModule.Handle := TTVMRuntime.TModule.RuntimeHandle;
    end;
  end;
  if TTVMRuntime.TModule.RuntimeHandle <> 0 then
  begin
    Candidate := TTVMRuntime.TModule.LibraryPath;
    if (Candidate <> '') and (Candidate[1] <> '<') then
    begin
      if FileExists(Candidate + TTVMRuntime.TModule.ExtraRuntimeName) then
        TTVMRuntime.TModule.ExtraRuntimeHandle := LoadLibrary({$IFDEF FPC}Candidate + TTVMRuntime.TModule.ExtraRuntimeName{$ELSE}PChar(Candidate +
          TTVMRuntime.TModule.ExtraRuntimeName){$ENDIF});
      if FileExists(Candidate + TTVMRuntime.TModule.CudaRuntimeName) then
        TTVMRuntime.TModule.CudaRuntimeHandle := LoadLibrary({$IFDEF FPC}Candidate + TTVMRuntime.TModule.CudaRuntimeName{$ELSE}PChar(Candidate +
          TTVMRuntime.TModule.CudaRuntimeName){$ENDIF});
    end
    else begin
      TTVMRuntime.TModule.ExtraRuntimeHandle := LoadLibrary({$IFDEF FPC}TTVMRuntime.TModule.ExtraRuntimeName{$ELSE}PChar(TTVMRuntime.TModule.ExtraRuntimeName){$ENDIF});
      TTVMRuntime.TModule.CudaRuntimeHandle := LoadLibrary({$IFDEF FPC}TTVMRuntime.TModule.CudaRuntimeName{$ELSE}PChar(TTVMRuntime.TModule.CudaRuntimeName){$ENDIF});
    end;
  end;
  if TTVMRuntime.TModule.Handle = 0 then
  begin
    S := {$IFDEF FPC}GetLoadErrorStr{$ELSE}SysErrorMessage(GetLastError){$ENDIF};
    TTVMRuntime.TModule.Error := Format('Failed to load %s / %s: %s',
      [TTVMRuntime.TModule.RuntimeName, TTVMRuntime.TModule.FFIName, S]);
    {$IFDEF MSWINDOWS}
    SetCurrentDir(Path);
    {$ENDIF}
    Exit;
  end;
  TTVMRuntime.TModule.Initialize(TTVMRuntime.TModule.Handle, False);
  TTVMBackend.TModule.Initialize(TTVMBackend.TModule.Handle, False);
  TTVMRuntime.TModule.ResolveGlobals;
  {$IFDEF MSWINDOWS}
  SetCurrentDir(Path);
  {$ENDIF}
end;

procedure Finalize;

var
  Runtime: {$IFDEF FPC}TLibHandle{$ELSE}THandle{$ENDIF};
begin
  Runtime := TTVMRuntime.TModule.RuntimeHandle;
  if TTVMRuntime.TModule.CudaRuntimeHandle <> 0 then
    FreeLibrary(TTVMRuntime.TModule.CudaRuntimeHandle);
  if TTVMRuntime.TModule.ExtraRuntimeHandle <> 0 then
    FreeLibrary(TTVMRuntime.TModule.ExtraRuntimeHandle);
  if TTVMRuntime.TModule.Handle <> 0 then
    FreeLibrary(TTVMRuntime.TModule.Handle);
  if (Runtime <> 0) and (Runtime <> TTVMRuntime.TModule.Handle) then
    FreeLibrary(Runtime);
  TTVMRuntime.TModule.CudaRuntimeHandle := 0;
  TTVMRuntime.TModule.ExtraRuntimeHandle := 0;
  TTVMRuntime.TModule.Handle := 0;
  TTVMRuntime.TModule.RuntimeHandle := 0;
  TTVMBackend.TModule.Handle := 0;
  TTVMRuntime.TModule.LibraryPath := '';
  TTVMRuntime.TModule.LibraryName := '';
  TTVMRuntime.TModule.TVMFFIGetVersion := nil;
  TTVMRuntime.TModule.TVMFFIObjectIncRef := nil;
  TTVMRuntime.TModule.TVMFFIObjectDecRef := nil;
  TTVMRuntime.TModule.TVMFFIFunctionGetGlobal := nil;
  TTVMRuntime.TModule.TVMFFIFunctionCall := nil;
  TTVMRuntime.TModule.TVMFFIErrorMoveFromRaised := nil;
  TTVMRuntime.TModule.TVMFFITensorFromDLPackVersioned := nil;
  TTVMRuntime.TModule.TVMFFITensorToDLPackVersioned := nil;
  TTVMRuntime.TModule.TVMFFIFunctionCreate := nil;
  TTVMRuntime.TModule.TVMFFIFunctionSetGlobal := nil;
  TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr := nil;
  TTVMRuntime.TModule.GExecutableLoadFromFile := nil;
  TTVMRuntime.TModule.GModuleLoadFromFile := nil;
  TTVMRuntime.TModule.GModuleGetFunction := nil;
  TTVMRuntime.TModule.GTensorCopyFromBytes := nil;
  TTVMRuntime.TModule.GTensorCopyToBytes := nil;
  TTVMRuntime.TModule.GTensorCopyFromTo := nil;
end;

function ApplySigmoidToImage(const Image32F: IMat): Boolean;
var
  Y: Integer;
  Base: PByte;
  Stride: NativeInt;
  L, R: PSingle;
begin
  if not Assigned(Image32F) or Image32F.IsEmpty or (Image32F.Depth <> CV_32F) or
    (Image32F.Channels <> 1) then
      Exit(False);
  Base := PByte(Image32F.DataPtr);
  Stride := Image32F.Step;
  for Y := 0 to Image32F.Height - 1 do
  begin
    L := PSingle(NativeUInt(Base) + NativeUInt(Y * Stride));
    R := PSingle(NativeUInt(L) + LongWord(Image32F.Width) * SizeOf(Single));
    while L < R do
    begin
      L^ := 1 / (1 + Exp(-L^));
      Inc(L);
    end;
  end;
  Result := True;
end;

function ShapeToString(const Shape: TArray<Int64>): string;
var
  I: Integer;
begin
  if Length(Shape) = 0 then Exit('[]');
  Result := '[';
  for I := Low(Shape) to High(Shape) do
  begin
    if I > Low(Shape) then
      Result := Result + ',';
    Result := Result + IntToStr(Shape[I]);
  end;
  Result := Result + ']';
end;

function CreateShapeArray(const Shape: PInt64; const NDim: Integer): TArray<Int64>;
begin
  if (NDim < 1) or not Assigned(Shape) then Exit(nil);
  SetLength(Result, NDim);
  System.Move(Shape^, Pointer(Result)^, NDim * SizeOf(Int64));
end;

initialization
  {$IFNDEF TVM_LAZY_INIT}
  Initialize;
  {$ENDIF}

end.
