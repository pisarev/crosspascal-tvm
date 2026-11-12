{ ************************************************************************** }
{                                                                            }
{ TVM.TokenGraph                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.TokenGraph;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, DLPack, TVM.CpuGraph, TVM.CudaExecution;

type
  TTokenGraph = class
  private
    FDecode: TCpuGraph;
    FPrefill: TCpuGraph;
    FDecodeWorkspace: TCudaWorkspace;
    FPrefillWorkspace: TCudaWorkspace;
    FDevice: TDLDeviceType;
    FWeightData: TMappedModelFile;
    FWeights: TTensorHandles;
    FCacheShape: TArray<Int64>;
    FCacheCount: Integer;
    FContext: Integer;
    FChunk: Integer;
    FVocab: Integer;
  public
    constructor Create(const Folder, DecodeLibrary, PrefillLibrary: string;
      const Manifest: TJSONObject; const Context, Chunk: Integer; const ExtraFiles: array of string;
      const Cuda: TCudaExecution = nil);
    destructor Destroy; override;
    function NewCache: TTensorHandles;
    function Evaluate(const Tokens: array of Integer; const Position: Integer;
      var Caches: TTensorHandles; const Stop: TCpuGraphStop): TArray<Single>;
    property Vocabulary: Integer read FVocab;
  end;

implementation

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM;

const
  WeightsFile = 'weights.bin';
  MiB = 1024 * 1024;

constructor TTokenGraph.Create(const Folder, DecodeLibrary, PrefillLibrary: string;
  const Manifest: TJSONObject; const Context, Chunk: Integer; const ExtraFiles: array of string;
  const Cuda: TCudaExecution);
var
  Layout: TJSONArray;
  Hashes, Weight: TJSONObject;
  Shape: TArray<Int64>;
  Name, DataType: string;
  Code: TDLDataTypeCode;
  Bits: Byte;
  Offset, ExpectedOffset, Bytes, Count: Int64;
  CacheBytes, Free, Reserve: UInt64;
  I: Integer;

  procedure CheckHash(const FileName: string);
  begin
    RequireCpu(Length(ModelText(Hashes.GetValue(FileName))) = 64, 'Missing hash for ' + FileName);
  end;

begin
  inherited Create;
  FContext := Context;
  FChunk := Chunk;
  FDevice := kDLCPU;
  if Cuda <> nil then FDevice := kDLCUDA;
  Count := ModelInt(Manifest.GetValue('cache_count'), 'cache count');
  RequireCpu((Count > 0) and (Count <= 256) and not Odd(Count), 'Invalid model cache count');
  FCacheCount := Count;
  FCacheShape := ModelShape(Manifest.GetValue('cache_shape'));
  RequireCpu((Length(FCacheShape) = 4) and (FCacheShape[0] = 1) and
    (FCacheShape[2] = FContext) and (TensorElements(FCacheShape) <= MaxInt div SizeOf(Single)),
    'Invalid model cache shape');
  Count := ModelInt(Manifest.GetValue('vocab_size'), 'vocabulary size');
  RequireCpu((Count > 0) and (Count <= 1048576), 'Invalid model vocabulary size');
  FVocab := Count;
  RequireCpu((ModelText(Manifest.GetValue('library')) = DecodeLibrary) and
    (ModelText(Manifest.GetValue('prefill_library')) = PrefillLibrary), 'Unexpected model library');
  Hashes := ModelObject(Manifest.GetValue('artifact_sha256'), 'model artifact hashes');
  CheckHash(DecodeLibrary);
  CheckHash(PrefillLibrary);
  CheckHash(WeightsFile);
  for Name in ExtraFiles do CheckHash(Name);
  VerifyArtifacts(Folder, Hashes);
  FWeightData := TMappedModelFile.Create(BundleMember(Folder, WeightsFile));
  Layout := ModelArray(Manifest.GetValue('weights'), 'model weight layout');
  RequireCpu((Layout.Count > 0) and (Layout.Count <= 4096), 'Invalid model parameter count');
  Reserve := 0;
  if Cuda <> nil then
  begin
    CacheBytes := UInt64(TensorElements(FCacheShape)) * SizeOf(Single) * UInt64(FCacheCount) * 2;
    Free := Cuda.FreeBytes;
    RequireCpu(Free >= UInt64(FWeightData.Size) + CacheBytes + 256 * MiB,
      'Not enough free CUDA memory for the weights, cache and workspace');
    Reserve := (Free - UInt64(FWeightData.Size) - CacheBytes - 128 * MiB) div 2;
    if Reserve > 512 * MiB then Reserve := 512 * MiB;
  end;
  SetLength(FWeights, Layout.Count);
  ExpectedOffset := 0;
  for I := 0 to High(FWeights) do
  begin
    Weight := ModelObject(Layout.Items[I], 'model weight');
    Name := ModelText(Weight.GetValue('name'));
    Offset := ModelInt(Weight.GetValue('offset'), Name + ' offset');
    Shape := ModelShape(Weight.GetValue('shape'));
    Count := TensorElements(Shape);
    RequireCpu((Length(Shape) > 0) and (Count <= High(Int64) div SizeOf(Single)) and
      (Offset = ExpectedOffset), 'Invalid model weight layout: ' + Name);
    DataType := ModelText(Weight.GetValue('dtype'));
    Code := kDLFloat;
    Bits := 32;
    if DataType = 'int8' then
    begin
      RequireCpu(Cuda <> nil, 'The int8 token graph requires CUDA');
      Code := kDLInt;
      Bits := 8;
    end
    else
      RequireCpu((DataType = '') or (DataType = 'float32'), 'Unknown weight data type: ' + DataType);
    Bytes := Count * (Bits div 8);
    if Cuda = nil then
      FWeights[I] := MappedFloatTensor(FWeightData.AtOffset(Offset, Bytes), Shape)
    else
      FWeights[I] := DeviceTensor(Shape, Code, Bits, FWeightData.AtOffset(Offset, Bytes), FDevice);
    Inc(ExpectedOffset, Bytes);
  end;
  RequireCpu(ExpectedOffset = FWeightData.Size, 'Unexpected model weight file size');
  if Cuda <> nil then
  begin
    Cuda.Synchronize;
    FreeAndNil(FWeightData);
  end;
  FDecode := TCpuGraph.Create(BundleMember(Folder, DecodeLibrary), FDevice);
  FPrefill := TCpuGraph.Create(BundleMember(Folder, PrefillLibrary), FDevice);
  if Cuda <> nil then
  begin
    FDecodeWorkspace := TCudaWorkspace.Create(Cuda, BundleMember(Folder, DecodeLibrary), Reserve);
    FPrefillWorkspace := TCudaWorkspace.Create(Cuda, BundleMember(Folder, PrefillLibrary), Reserve);
  end;
end;

destructor TTokenGraph.Destroy;
begin
  FPrefillWorkspace.Free;
  FDecodeWorkspace.Free;
  FPrefill.Free;
  FDecode.Free;
  ReleaseTensors(FWeights);
  FWeightData.Free;
  inherited;
end;

function TTokenGraph.NewCache: TTensorHandles;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, FCacheCount);
  try
    for I := 0 to High(Result) do
      Result[I] := DeviceTensor(FCacheShape, kDLFloat, 32, nil, FDevice);
  except
    ReleaseTensors(Result);
    raise;
  end;
end;

function TTokenGraph.Evaluate(const Tokens: array of Integer; const Position: Integer;
  var Caches: TTensorHandles; const Stop: TCpuGraphStop): TArray<Single>;
var
  Graph: TCpuGraph;
  Workspace: TCudaWorkspace;
  IDs, Positions: TArray<Int64>;
  Args, NextCaches: TTensorHandles;
  TokenTensor, PositionTensor, Logits: TTVMFFIObjectHandle;
  Reply: TTVMFFIAny;
  I, Count: Integer;
begin
  Count := Length(Tokens);
  RequireCpu(((Count = 1) or (Count = FChunk)) and (Length(Caches) = FCacheCount) and
    (Position >= 0) and (Position <= FContext - Count), 'Invalid token graph request');
  if Count = 1 then
  begin
    Graph := FDecode;
    Workspace := FDecodeWorkspace;
  end
  else begin
    Graph := FPrefill;
    Workspace := FPrefillWorkspace;
  end;
  SetLength(IDs, Count);
  SetLength(Positions, Count);
  for I := 0 to High(Tokens) do
  begin
    RequireCpu((Tokens[I] >= 0) and (Tokens[I] < FVocab), 'Token outside the model vocabulary');
    IDs[I] := Tokens[I];
    Positions[I] := Position + I;
  end;
  SetLength(Args, 2 + Length(Caches) + Length(FWeights));
  TokenTensor := nil;
  PositionTensor := nil;
  try
    TokenTensor := DeviceTensor([1, Count], kDLInt, 64, Pointer(IDs), FDevice);
    PositionTensor := DeviceTensor([Count], kDLInt, 64, Pointer(Positions), FDevice);
    Args[0] := TokenTensor;
    Args[1] := PositionTensor;
    for I := 0 to High(Caches) do Args[2 + I] := Caches[I];
    for I := 0 to High(FWeights) do Args[2 + Length(Caches) + I] := FWeights[I];
    if Workspace = nil then
      Reply := Graph.Run(Args, Stop)
    else
      Reply := Workspace.Run(Graph, Args, Stop);
    RequireCpu(Reply.VObj <> nil, 'TVM returned no token graph output');
    try
      Logits := TensorItem(Reply.VObj, 0);
      try
        Result := ReadFloatTensor(Logits);
      finally
        TTVMRuntime.ObjectRelease(Logits);
      end;
      SetLength(NextCaches, Length(Caches));
      for I := 0 to High(Caches) do NextCaches[I] := TensorItem(Reply.VObj, I + 1);
      ReleaseTensors(Caches);
      Caches := NextCaches;
      NextCaches := nil;
    finally
      TTVMRuntime.ObjectRelease(Reply.VObj);
    end;
  finally
    ReleaseTensors(NextCaches);
    TTVMRuntime.ObjectRelease(PositionTensor);
    TTVMRuntime.ObjectRelease(TokenTensor);
  end;
end;

end.
