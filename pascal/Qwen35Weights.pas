{ ************************************************************************** }
{                                                                            }
{ Qwen35Weights                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit Qwen35Weights;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Generics.Collections, Windows
  {$ELSE}System.SysUtils, System.Generics.Collections, Winapi.Windows{$ENDIF},
  CrossPascal.Json, TVM.CpuGraph, TVM, DLPack, Qwen35PackedCuda;

type
  TQwen35Weight = record
    FileName: string;
    Encoding: string;
    Offset: Int64;
    Bytes: Int64;
    Elements: Int64;
    Shape: TArray<Int64>;
    Format: Integer;
    BlockElements: Integer;
    BlockBytes: Integer;
  end;

  TQwen35WeightFiles = class
  private
    type
      TDecode = procedure(Source: Pointer; Destination: PSingle; Count: Int64); cdecl;
  private
    FDirectory: string;
    FDecoderLibrary: string;
    FFiles: TObjectDictionary<string, TMappedModelFile>;
    FLibrary: HMODULE;
    FPacked: TQwen35PackedCuda;
    function ModelFile(const Name: string): TMappedModelFile;
    function Decoder(const Format: Integer): TDecode;
  public
    constructor Create(const Directory, DecoderLibrary: string; const PackedModule: string = '');
    destructor Destroy; override;
    procedure ConfigurePackedCache(const BudgetBytes: UInt64);
    procedure ReleaseCudaCache;
    function PackedCacheStats: TQwen35PackedCacheStats;
    function Weight(const Value: TJSONValue): TQwen35Weight;
    function RawRange(const Weight: TQwen35Weight; const First, Count: Int64): Pointer;
    function FloatRange(const Weight: TQwen35Weight; const First, Count: Int64): TArray<Single>;
    procedure ReleaseResidentRange(const Weight: TQwen35Weight; const First, Count: Int64);
    function CanPoolCudaTensor(const Weight: TQwen35Weight): Boolean;
    function Tensor(const Weight: TQwen35Weight; const Device: TDLDeviceType;
      const Destination: TTVMFFIObjectHandle = nil): TTVMFFIObjectHandle;
    function TensorSlice(const Weight: TQwen35Weight; const First, Valid: Int64;
      const Shape: TArray<Int64>; const Device: TDLDeviceType;
      const Destination: TTVMFFIObjectHandle = nil): TTVMFFIObjectHandle;
  end;

procedure Qwen35PackedLayout(const Format: Integer; out BlockElements, BlockBytes: Integer);

implementation

const
  QwenReaderSearch: DWORD = $00000900;

procedure Qwen35PackedLayout(const Format: Integer; out BlockElements, BlockBytes: Integer);
begin
  BlockElements := 256;
  case Format of
    0:
      begin
        BlockElements := 1;
        BlockBytes := 4;
      end;
    1, 30:
      begin
        BlockElements := 1;
        BlockBytes := 2;
      end;
    8:
      begin
        BlockElements := 32;
        BlockBytes := 34;
      end;
    11, 21: BlockBytes := 110;
    12: BlockBytes := 144;
    13: BlockBytes := 176;
    14: BlockBytes := 210;
    20:
      begin
        BlockElements := 32;
        BlockBytes := 18;
      end;
    23: BlockBytes := 136;
  else
    raise ECpuGraph.CreateFmt('Unsupported Qwen GGUF storage type %d', [Format]);
  end;
end;

constructor TQwen35WeightFiles.Create(const Directory, DecoderLibrary, PackedModule: string);
begin
  inherited Create;
  FDirectory := ExcludeTrailingPathDelimiter(ExpandFileName(Directory));
  FDecoderLibrary := DecoderLibrary;
  FFiles := TObjectDictionary<string, TMappedModelFile>.Create([doOwnsValues]);
  if PackedModule <> '' then FPacked := TQwen35PackedCuda.Create(PackedModule);
end;

destructor TQwen35WeightFiles.Destroy;
begin
  FPacked.Free;
  FFiles.Free;
  if FLibrary <> 0 then FreeLibrary(FLibrary);
  inherited;
end;

procedure TQwen35WeightFiles.ConfigurePackedCache(const BudgetBytes: UInt64);
begin
  RequireCpu((BudgetBytes = 0) or (FPacked <> nil),
    'Packed CUDA caching requires a packed CUDA decoder');
  if FPacked <> nil then FPacked.ConfigureCache(BudgetBytes);
end;

procedure TQwen35WeightFiles.ReleaseCudaCache;
begin
  if FPacked <> nil then FPacked.ReleaseCachedWeights;
end;

function TQwen35WeightFiles.PackedCacheStats: TQwen35PackedCacheStats;
begin
  Result := Default(TQwen35PackedCacheStats);
  if FPacked <> nil then Result := FPacked.CacheStats;
end;

function TQwen35WeightFiles.ModelFile(const Name: string): TMappedModelFile;
begin
  if FFiles.TryGetValue(Name, Result) then Exit;
  Result := TMappedModelFile.Create(BundleMember(FDirectory, Name));
  try
    FFiles.Add(Name, Result);
  except
    Result.Free;
    raise;
  end;
end;

function TQwen35WeightFiles.Weight(const Value: TJSONValue): TQwen35Weight;
var
  Spec: TJSONObject;
  Data: TMappedModelFile;
  Required: Int64;
begin
  Result := Default(TQwen35Weight);
  Spec := ModelObject(Value, 'Qwen weight descriptor');
  Result.FileName := ModelText(Spec.GetValue('file'));
  Result.Encoding := ModelText(Spec.GetValue('encoding'));
  Result.Offset := ModelInt(Spec.GetValue('offset'), 'weight offset');
  Result.Bytes := ModelInt(Spec.GetValue('bytes'), 'weight bytes');
  Result.Shape := ModelShape(Spec.GetValue('shape'));
  Result.Elements := TensorElements(Result.Shape);
  RequireCpu((Result.Offset >= 0) and (Result.Bytes > 0), 'Invalid Qwen weight range');
  if Result.Encoding = 'bf16' then
  begin
    Result.BlockElements := 1;
    Result.BlockBytes := 2;
    Result.Format := 30;
  end
  else if Result.Encoding = 'float32' then
  begin
    Result.BlockElements := 1;
    Result.BlockBytes := 4;
  end
  else begin
    RequireCpu(Result.Encoding = 'gguf', 'Unsupported Qwen weight encoding');
    Result.Format := ModelInt(Spec.GetValue('format'), 'GGUF storage type');
    Qwen35PackedLayout(Result.Format, Result.BlockElements, Result.BlockBytes);
  end;
  RequireCpu((Result.Elements mod Result.BlockElements = 0) and
    (Result.Elements div Result.BlockElements <= High(Int64) div Result.BlockBytes),
    'Invalid Qwen packed weight dimensions');
  Required := (Result.Elements div Result.BlockElements) * Result.BlockBytes;
  RequireCpu(Result.Bytes = Required, 'Qwen weight shape and byte count differ');
  Data := ModelFile(Result.FileName);
  RequireCpu((Result.Offset <= Data.Size) and (Result.Bytes <= Data.Size - Result.Offset),
    'Qwen weight extends outside its artifact');
end;

function TQwen35WeightFiles.Decoder(const Format: Integer): TDecode;
var
  Name: AnsiString;
begin
  case Format of
    1: Name := 'ggml_fp16_to_fp32_row';
    8: Name := 'dequantize_row_q8_0';
    11: Name := 'dequantize_row_q3_K';
    12: Name := 'dequantize_row_q4_K';
    13: Name := 'dequantize_row_q5_K';
    14: Name := 'dequantize_row_q6_K';
    20: Name := 'dequantize_row_iq4_nl';
    21: Name := 'dequantize_row_iq3_s';
    23: Name := 'dequantize_row_iq4_xs';
  else
    raise ECpuGraph.Create('No tensor decoder for the selected Qwen storage type');
  end;
  if FLibrary = 0 then
  begin
    RequireCpu(FDecoderLibrary <> '', 'A GGUF tensor decoder library is required');
    FLibrary := LoadLibraryExW(PWideChar(ExpandFileName(FDecoderLibrary)), 0, QwenReaderSearch);
    if FLibrary = 0 then
      raise ECpuGraph.CreateFmt('Could not load the model reader or its dependencies (Windows error %d). Verify the model package.',
        [GetLastError]);
  end;
  Result := TDecode(GetProcAddress(FLibrary, PAnsiChar(Name)));
  RequireCpu(Assigned(Result), 'Missing GGUF tensor decode export: ' + string(Name));
end;

function TQwen35WeightFiles.RawRange(const Weight: TQwen35Weight;
  const First, Count: Int64): Pointer;
var
  Offset, Bytes: Int64;
begin
  RequireCpu((First >= 0) and (Count >= 0) and (First <= Weight.Elements) and
    (Count <= Weight.Elements - First), 'Qwen weight element range is out of bounds');
  RequireCpu((Weight.BlockElements > 0) and (Weight.BlockBytes > 0) and
    (First mod Weight.BlockElements = 0) and (Count mod Weight.BlockElements = 0),
    'Qwen weight range is not block aligned');
  if Count = 0 then Exit(nil);
  Offset := (First div Weight.BlockElements) * Weight.BlockBytes;
  Bytes := (Count div Weight.BlockElements) * Weight.BlockBytes;
  RequireCpu((Offset <= Weight.Bytes) and (Bytes <= Weight.Bytes - Offset),
    'Qwen packed range is out of bounds');
  Result := ModelFile(Weight.FileName).AtOffset(Weight.Offset + Offset, Bytes);
end;

function TQwen35WeightFiles.FloatRange(const Weight: TQwen35Weight;
  const First, Count: Int64): TArray<Single>;
var
  Data: Pointer;
  Words: PWord;
  Bits: PCardinal;
  I: NativeInt;
  Decode: TDecode;
begin
  Result := nil;
  RequireCpu(Count <= High(NativeInt) div SizeOf(Single), 'Qwen decoded range is too large');
  Data := RawRange(Weight, First, Count);
  if Count = 0 then Exit;
  SetLength(Result, Count);
  if Weight.Format = 0 then
    Move(Data^, Result[0], Count * SizeOf(Single))
  else if Weight.Format = 30 then
  begin
    Words := Data;
    Bits := PCardinal(Pointer(Result));
    for I := 0 to Count - 1 do Bits[I] := Cardinal(Words[I]) shl 16;
  end
  else begin
    Decode := Decoder(Weight.Format);
    Decode(Data, @Result[0], Count);
  end;
end;

procedure TQwen35WeightFiles.ReleaseResidentRange(const Weight: TQwen35Weight;
  const First, Count: Int64);
var
  Data: Pointer;
  Bytes: NativeUInt;
  Error: Cardinal;
begin
  Data := RawRange(Weight, First, Count);
  if Data = nil then Exit;
  Bytes := NativeUInt(Count div Weight.BlockElements) * Weight.BlockBytes;
  { Windows documents VirtualUnlock on unlocked pages as releasing them from
    this process's working set, returning FALSE/ERROR_NOT_LOCKED. These source
    mappings are never locked. Their addresses and bytes remain valid. }
  if not VirtualUnlock(Data, Bytes) then
  begin
    Error := GetLastError;
    if Error <> ERROR_NOT_LOCKED then RaiseLastOSError(Error);
  end;
end;

function TQwen35WeightFiles.CanPoolCudaTensor(const Weight: TQwen35Weight): Boolean;
begin
  Result := (FPacked <> nil) and (Weight.Encoding = 'gguf') and
    (Weight.Format in [8, 11, 12, 13, 14, 20, 21, 23]);
end;

function TQwen35WeightFiles.Tensor(const Weight: TQwen35Weight; const Device: TDLDeviceType;
  const Destination: TTVMFFIObjectHandle): TTVMFFIObjectHandle;
begin
  Result := TensorSlice(Weight, 0, Weight.Elements, Weight.Shape, Device, Destination);
end;

function TQwen35WeightFiles.TensorSlice(const Weight: TQwen35Weight; const First, Valid: Int64;
  const Shape: TArray<Int64>; const Device: TDLDeviceType;
  const Destination: TTVMFFIObjectHandle): TTVMFFIObjectHandle;
var
  Count: Int64;
  Raw: TBytes;
  Floats: TArray<Single>;
  Data: Pointer;
begin
  Count := TensorElements(Shape);
  RequireCpu((Valid > 0) and (Valid <= Count), 'Invalid padded Qwen tensor slice');
  RequireCpu((Destination = nil) or ((Device = kDLCUDA) and CanPoolCudaTensor(Weight)),
    'A Qwen destination buffer requires packed CUDA weights');
  if Weight.Encoding = 'bf16' then
  begin
    Data := RawRange(Weight, First, Valid);
    if Valid < Count then
    begin
      RequireCpu(Count <= High(NativeInt) div 2, 'Padded BF16 tensor is too large');
      SetLength(Raw, Count * 2);
      Move(Data^, Raw[0], Valid * 2);
      Data := Pointer(Raw);
    end;
    Exit(DeviceTensor(Shape, kDLBfloat, 16, Data, Device));
  end;
  if (Weight.Format = 0) and (Valid = Count) then
  begin
    Data := RawRange(Weight, First, Valid);
    { Generated CPU vector loads assume the TVM allocator's 64-byte alignment.
      GGUF tensor offsets may provide only 32-byte alignment. Copy those
      ranges into an aligned tensor instead of exposing an unsafe mapped view. }
    if (Device = kDLCPU) and (NativeUInt(Data) and 63 = 0) then
      Exit(MappedFloatTensor(Data, Shape));
    Exit(DeviceTensor(Shape, kDLFloat, 32, Data, Device));
  end;
  if (Device = kDLCUDA) and CanPoolCudaTensor(Weight) then
  begin
    Data := RawRange(Weight, First, Valid);
    Exit(FPacked.Tensor(Data, (Valid div Weight.BlockElements) * Weight.BlockBytes,
      Valid, Count, Weight.Format, Shape, Destination));
  end;
  Floats := FloatRange(Weight, First, Valid);
  if Valid < Count then SetLength(Floats, Count);
  Result := DeviceTensor(Shape, kDLFloat, 32, Pointer(Floats), Device);
end;

end.
