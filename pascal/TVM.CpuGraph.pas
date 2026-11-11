{ ************************************************************************** }
{                                                                            }
{ TVM.CpuGraph                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit TVM.CpuGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  TVM, DLPack, CrossPascal.Json;

type
  ECpuGraph = class(Exception);
  ECpuGraphStopped = class(ECpuGraph);
  TTVMCallError = {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
  PTVMCallError = ^TTVMCallError;
  TTensorHandles = TArray<TTVMFFIObjectHandle>;
  TCpuGraphStop = function: Boolean of object;

  TMappedModelFile = class
  private
    FAddress: Pointer;
    FSize: Int64;
  public
    constructor Create(const FileName: string);
    destructor Destroy; override;
    function AtOffset(const Offset, ByteCount: Int64): Pointer;
    property Size: Int64 read FSize;
  end;

  TCpuGraph = class
  private
    FModule: TTVMFFIObjectHandle;
    FVM: TTVMFFIObjectHandle;
    FMain: TTVMFFIObjectHandle;
    FShouldStop: TCpuGraphStop;
    FStopped: Boolean;
  public
    constructor Create(const LibraryName: string; const DeviceType: TDLDeviceType = kDLCPU);
    destructor Destroy; override;
    function Run(const Inputs: array of TTVMFFIObjectHandle;
      const ShouldStop: TCpuGraphStop = nil): TTVMFFIAny;
  end;

procedure RequireCpu(const Condition: Boolean; const MessageText: string);
procedure RequireTVM(const Condition: Boolean; const Error: PTVMCallError);
function BundleMember(const Folder, Name: string): string;
function ReadModelJSON(const FileName: string): TJSONValue;
function ModelObject(const Value: TJSONValue; const Name: string): TJSONObject;
function ModelArray(const Value: TJSONValue; const Name: string): TJSONArray;
function ModelText(const Value: TJSONValue): string;
function ModelInt(const Value: TJSONValue; const Name: string): Int64;
function ModelFloat(const Value: TJSONValue; const Name: string): Double;
function ModelShape(const Value: TJSONValue): TArray<Int64>;
function TensorElements(const Shape: array of Int64): Int64;
procedure VerifyArtifacts(const Folder: string; const Hashes: TJSONObject);
function MappedFloatTensor(const Address: Pointer; const Shape: array of Int64): TTVMFFIObjectHandle;
function CpuTensor(const Shape: array of Int64; const Code: TDLDataTypeCode; const Bits: Integer;
  const Data: Pointer): TTVMFFIObjectHandle;
function DeviceTensor(const Shape: array of Int64; const Code: TDLDataTypeCode; const Bits: Integer;
  const Data: Pointer; const DeviceType: TDLDeviceType): TTVMFFIObjectHandle;
function TensorItem(const Items: TTVMFFIObjectHandle; const Index: Integer): TTVMFFIObjectHandle;
function ReadFloatTensor(const Handle: TTVMFFIObjectHandle): TArray<Single>;
procedure ReleaseTensors(var Tensors: TTensorHandles);

implementation

uses
  {$IFDEF FPC}Windows, Math{$ELSE}Winapi.Windows, System.Math{$ENDIF},
  CrossPascal.SHA256;

procedure RequireCpu(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise ECpuGraph.Create(MessageText);
end;

procedure RequireTVM(const Condition: Boolean; const Error: PTVMCallError);
begin
  if not Condition then
  begin
    if (Error <> nil) and (Error^ <> '') then raise ECpuGraph.Create(string(Error^));
    raise ECpuGraph.Create('TVM failed without an error message');
  end;
end;

function BundleMember(const Folder, Name: string): string;
begin
  RequireCpu((Name <> '') and (Name <> '.') and (Name <> '..') and
    (Pos(':', Name) = 0) and (Pos('/', Name) = 0) and (Pos('\', Name) = 0),
    'Invalid model file name: ' + Name);
  Result := IncludeTrailingPathDelimiter(Folder) + Name;
end;

function ReadModelJSON(const FileName: string): TJSONValue;
var
  Bytes: UTF8String;
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= 64 * 1024 * 1024),
      'Invalid metadata file size: ' + FileName);
    SetLength(Bytes, Stream.Size);
    Stream.ReadBuffer(Bytes[1], Length(Bytes));
    if Copy(Bytes, 1, 3) = #$EF#$BB#$BF then Delete(Bytes, 1, 3);
    Result := ParseCrossJSON(UTF8Decode(Bytes));
    RequireCpu(Result <> nil, 'Invalid model metadata: ' + FileName);
  finally
    Stream.Free;
  end;
end;

function ModelObject(const Value: TJSONValue; const Name: string): TJSONObject;
begin
  RequireCpu(Value is TJSONObject, 'Expected an object for ' + Name);
  Result := TJSONObject(Value);
end;

function ModelArray(const Value: TJSONValue; const Name: string): TJSONArray;
begin
  RequireCpu(Value is TJSONArray, 'Expected an array for ' + Name);
  Result := TJSONArray(Value);
end;

function ModelText(const Value: TJSONValue): string;
begin
  if (Value = nil) or (Value is TJSONNull) then Exit('');
  Result := JSONValueText(Value);
end;

function ModelInt(const Value: TJSONValue; const Name: string): Int64;
begin
  RequireCpu((Value is TJSONNumber) and TryStrToInt64(JSONValueText(Value), Result),
    'Expected an integer for ' + Name);
end;

function ModelFloat(const Value: TJSONValue; const Name: string): Double;
begin
  RequireCpu(
    (Value is TJSONNumber) and TryStrToFloat(JSONValueText(Value), Result, JSONInvariantFormat) and not IsNan(Result) and not IsInfinite(Result),
    'Expected a finite number for ' + Name
  );
end;

function ModelShape(const Value: TJSONValue): TArray<Int64>;
var
  Values: TJSONArray;
  I: Integer;
begin
  Values := ModelArray(Value, 'tensor shape');
  RequireCpu((Values.Count > 0) and (Values.Count <= 8), 'Invalid tensor rank');
  SetLength(Result, Values.Count);
  for I := 0 to High(Result) do Result[I] := ModelInt(Values.Items[I], 'shape dimension');
  TensorElements(Result);
end;

function TensorElements(const Shape: array of Int64): Int64;
var
  Dim: Int64;
begin
  Result := 1;
  for Dim in Shape do
  begin
    RequireCpu((Dim > 0) and (Result <= High(Int64) div 8 div Dim), 'Invalid tensor dimension');
    Result := Result * Dim;
  end;
end;

procedure VerifyArtifacts(const Folder: string; const Hashes: TJSONObject);
var
  I: Integer;
  Name, Digest: string;
  Stream: TFileStream;
begin
  for I := 0 to Hashes.Count - 1 do
  begin
    Name := JSONObjectName(Hashes, I);
    Stream := TFileStream.Create(BundleMember(Folder, Name), fmOpenRead or fmShareDenyWrite);
    try
      Digest := SHA256Stream(Stream);
      RequireCpu(SameText(Digest, JSONValueText(JSONObjectValue(Hashes, I))),
        'Model artifact changed: ' + Name);
    finally
      Stream.Free;
    end;
  end;
end;

constructor TMappedModelFile.Create(const FileName: string);
var
  Stream: TFileStream;
  Mapping: THandle;
begin
  inherited Create;
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    FSize := Stream.Size;
    RequireCpu(FSize > 0, 'Model file is empty: ' + FileName);
    Mapping := CreateFileMapping(Stream.Handle, nil, PAGE_READONLY, 0, 0, nil);
    if Mapping = 0 then RaiseLastOSError;
    try
      FAddress := MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
      if FAddress = nil then RaiseLastOSError;
    finally
      CloseHandle(Mapping);
    end;
  finally
    Stream.Free;
  end;
end;

destructor TMappedModelFile.Destroy;
begin
  if FAddress <> nil then UnmapViewOfFile(FAddress);
  inherited;
end;

function TMappedModelFile.AtOffset(const Offset, ByteCount: Int64): Pointer;
begin
  RequireCpu((Offset >= 0) and (Offset <= FSize) and (ByteCount >= 0) and
    (ByteCount <= FSize - Offset), 'Tensor extends outside the model file');
  Result := PByte(FAddress) + Offset;
end;

constructor TCpuGraph.Create(const LibraryName: string; const DeviceType: TDLDeviceType);
var
  Func: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
  Error: TTVMCallError;
begin
  inherited Create;
  RequireTVM(TTVMRuntime.ModuleLoadFromFile(LibraryName, 'dll', FModule, @Error), @Error);
  RequireTVM(TTVMRuntime.ModuleGetFunction(FModule, 'vm_load_executable', True, Func, @Error), @Error);
  try
    RequireTVM(TTVMRuntime.CallFunc(Func, [], Value, @Error), @Error);
    RequireCpu(Value.TypeIndex = kTVMFFIModule, 'TVM returned an invalid virtual machine');
    FVM := Value.VObj;
  finally
    TTVMRuntime.ObjectRelease(Func);
  end;
  RequireTVM(TTVMRuntime.ModuleGetFunction(FVM, 'vm_initialization', True, Func, @Error), @Error);
  try
    if DeviceType = kDLCUDA then
      RequireTVM(TTVMRuntime.CallFunc(Func, [TTVMRuntime.AnyInt(Ord(kDLCUDA)),
        TTVMRuntime.AnyInt(0), TTVMRuntime.AnyInt(kTVMVMAllocNaive),
        TTVMRuntime.AnyInt(Ord(kDLCPU)), TTVMRuntime.AnyInt(0),
        TTVMRuntime.AnyInt(kTVMVMAllocPooled)], Value, @Error), @Error)
    else
      RequireTVM(TTVMRuntime.CallFunc(Func, [TTVMRuntime.AnyInt(Ord(kDLCPU)),
        TTVMRuntime.AnyInt(0), TTVMRuntime.AnyInt(kTVMVMAllocPooled)], Value, @Error), @Error);
  finally
    TTVMRuntime.ObjectRelease(Func);
  end;
  RequireTVM(TTVMRuntime.ModuleGetFunction(FVM, 'main', True, FMain, @Error), @Error);
end;

destructor TCpuGraph.Destroy;
begin
  TTVMRuntime.ObjectRelease(FMain);
  TTVMRuntime.ObjectRelease(FVM);
  TTVMRuntime.ObjectRelease(FModule);
  inherited;
end;

function CheckGraphStop(Context: Pointer; Args: PTVMFFIAny; NumArgs: Int32;
  Value: PTVMFFIAny): Integer; cdecl;
var
  Graph: TCpuGraph;
begin
  Graph := TCpuGraph(Context);
  try
    if (NumArgs >= 3) and (Args[2].VInt64 <> 0) and Graph.FShouldStop() then
    begin
      Graph.FStopped := True;
      TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr('InterruptedError', 'Generation stopped');
      Exit(-1);
    end;
    Value^ := TTVMRuntime.AnyInt(0);
    Result := 0;
  except
    TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr('RuntimeError', 'Stop callback failed');
    Result := -1;
  end;
end;

function TCpuGraph.Run(const Inputs: array of TTVMFFIObjectHandle;
  const ShouldStop: TCpuGraphStop): TTVMFFIAny;
var
  Args: TArray<TTVMFFIAny>;
  I: Integer;
  Error: TTVMCallError;
  Instrument, SetInstrument: TTVMFFIObjectHandle;
  Ignored: TTVMFFIAny;
begin
  SetLength(Args, Length(Inputs));
  for I := 0 to High(Inputs) do Args[I] := TTVMRuntime.AnyObject(Inputs[I]);
  if not Assigned(ShouldStop) then
  begin
    RequireTVM(TTVMRuntime.CallFunc(FMain, Args, Result, @Error), @Error);
    Exit;
  end;
  FStopped := False;
  FShouldStop := ShouldStop;
  Instrument := nil;
  SetInstrument := nil;
  try
    RequireTVM(TTVMRuntime.ModuleGetFunction(FVM, 'set_instrument', True, SetInstrument, @Error), @Error);
    RequireCpu(Assigned(TTVMRuntime.TModule.TVMFFIFunctionCreate) and Assigned(TTVMRuntime.TModule.TVMFFIErrorSetRaisedFromCStr),
      'TVM cannot install the stop callback');
    RequireCpu(TTVMRuntime.TModule.TVMFFIFunctionCreate(Self, CheckGraphStop, nil, Instrument) = 0,
      'TVM could not create the stop callback');
    RequireTVM(TTVMRuntime.CallFunc(SetInstrument, [TTVMRuntime.AnyObject(Instrument)], Ignored, @Error), @Error);
    try
      if not TTVMRuntime.CallFunc(FMain, Args, Result, @Error) then
      begin
        if FStopped then raise ECpuGraphStopped.Create('Generation stopped');
        raise ECpuGraph.Create(Error);
      end;
    finally
      TTVMRuntime.CallFunc(SetInstrument, [TTVMRuntime.AnyObject(nil)], Ignored);
    end;
  finally
    FShouldStop := nil;
    TTVMRuntime.ObjectRelease(Instrument);
    TTVMRuntime.ObjectRelease(SetInstrument);
  end;
end;

function MappedFloatTensor(const Address: Pointer; const Shape: array of Int64): TTVMFFIObjectHandle;
var
  Tensor: PDLTensor;
  Error: TTVMCallError;
begin
  GetMem(Tensor, SizeOf(TDLTensor) + Length(Shape) * SizeOf(Int64));
  try
    FillChar(Tensor^, SizeOf(TDLTensor), 0);
    Tensor^.data := Address;
    Tensor^.device.device_type := kDLCPU;
    Tensor^.ndim := Length(Shape);
    Tensor^.dtype.code := Ord(kDLFloat);
    Tensor^.dtype.bits := 32;
    Tensor^.dtype.lanes := 1;
    Tensor^.shape := PInt64(PByte(Tensor) + SizeOf(TDLTensor));
    Move(Shape[0], Tensor^.shape^, Length(Shape) * SizeOf(Int64));
    RequireTVM(TTVMRuntime.AdoptCpuTensor(Tensor, Result, @Error), @Error);
  finally
    if Tensor <> nil then FreeMem(Tensor);
  end;
end;

function CpuTensor(const Shape: array of Int64; const Code: TDLDataTypeCode; const Bits: Integer;
  const Data: Pointer): TTVMFFIObjectHandle;
begin
  Result := DeviceTensor(Shape, Code, Bits, Data, kDLCPU);
end;

function DeviceTensor(const Shape: array of Int64; const Code: TDLDataTypeCode; const Bits: Integer;
  const Data: Pointer; const DeviceType: TDLDeviceType): TTVMFFIObjectHandle;
var
  Error: TTVMCallError;
  ByteCount: Int64;
  Zero: TBytes;
begin
  ByteCount := TensorElements(Shape) * (Bits div 8);
  RequireTVM(TTVMRuntime.AllocDeviceTensor(Shape, Ord(Code), Bits, 1, Ord(DeviceType), 0, Result, @Error), @Error);
  try
    if Data <> nil then
      RequireTVM(TTVMRuntime.CopyBytesToTensor(Result, Data, ByteCount, @Error), @Error)
    else if DeviceType = kDLCUDA then
      begin
        SetLength(Zero, ByteCount);
        RequireTVM(TTVMRuntime.CopyBytesToTensor(Result, Pointer(Zero), ByteCount, @Error), @Error);
      end
      else
        FillChar(TTVMRuntime.TensorDLTensor(Result)^.data^, ByteCount, 0);
  except
    TTVMRuntime.ObjectRelease(Result);
    raise;
  end;
end;

function TensorItem(const Items: TTVMFFIObjectHandle; const Index: Integer): TTVMFFIObjectHandle;
var
  Value: TTVMFFIAny;
  Error: TTVMCallError;
begin
  RequireTVM(TTVMRuntime.CallGlobal('ffi.ArrayGetItem', [TTVMRuntime.AnyObject(Items),
    TTVMRuntime.AnyInt(Index)], Value, @Error), @Error);
  RequireCpu(Value.TypeIndex = kTVMFFITensor, 'TVM returned an invalid tensor item');
  Result := Value.VObj;
end;

function ReadFloatTensor(const Handle: TTVMFFIObjectHandle): TArray<Single>;
var
  Tensor: PDLTensor;
  Error: TTVMCallError;
begin
  Tensor := TTVMRuntime.TensorDLTensor(Handle);
  RequireCpu(TTVMRuntime.CheckTensor(Tensor, [kDLFloat], 32), 'Expected a float32 result');
  SetLength(Result, TensorElements(TTVMRuntime.GetTensorShape(Tensor)));
  RequireTVM(TTVMRuntime.CopyBytesFromTensor(Handle, Pointer(Result), Int64(Length(Result)) *
    SizeOf(Single), @Error), @Error);
end;

procedure ReleaseTensors(var Tensors: TTensorHandles);
var
  I: Integer;
begin
  for I := 0 to High(Tensors) do TTVMRuntime.ObjectRelease(Tensors[I]);
  Tensors := nil;
end;

end.
