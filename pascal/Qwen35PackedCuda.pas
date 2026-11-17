{ ************************************************************************** }
{                                                                            }
{ Qwen35PackedCuda                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit Qwen35PackedCuda;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Windows, Generics.Collections{$ELSE}System.SysUtils, Winapi.Windows, System.Generics.Collections{$ENDIF},
  TVM, DLPack, TVM.CpuGraph;

type
  TQwen35PackedCacheStats = record
    InitialBudgetBytes: UInt64;
    BudgetBytes: UInt64;
    ResidentBytes: UInt64;
    Hits: UInt64;
    Misses: UInt64;
    PressureClears: UInt64;
  end;

  { Loads the converter-built, attributed GGML tensor-decoding kernels.
    Caller enters its CUDA context. This class owns no model/LLM engine. }
  TQwen35PackedCuda = class
  private
    type
      TLoad = function(out Module: Pointer; Image: Pointer): Integer; stdcall;
      TUnload = function(Module: Pointer): Integer; stdcall;
      TFunction = function(out Func: Pointer; Module: Pointer; Name: PAnsiChar): Integer; stdcall;
      TLaunch = function(Func: Pointer; GridX, GridY, GridZ, BlockX, BlockY, BlockZ, SharedBytes: Cardinal;
        Stream, Parameters, Extra: Pointer): Integer; stdcall;
      TZero = function(Address: UInt64; Value: Cardinal; Count: NativeUInt): Integer; stdcall;
      TSync = function: Integer; stdcall;
      TMemoryInfo = function(out FreeBytes, TotalBytes: NativeUInt): Integer; stdcall;
  private
    FDriver: HMODULE;
    FModule: Pointer;
    FUnload: TUnload;
    FFunction: TFunction;
    FLaunch: TLaunch;
    FZero: TZero;
    FSync: TSync;
    FMemoryInfo: TMemoryInfo;
    FFunctions: array[0..30] of Pointer;
    FCache: TDictionary<string, TTVMFFIObjectHandle>;
    FStats: TQwen35PackedCacheStats;
    procedure ClearCache;
    function CacheKey(const Source: Pointer; const SourceBytes: Int64): string;
    function Entry(const Name: AnsiString): Pointer;
    function Kernel(const Format: Integer; out Threads: Cardinal;
      out BlockElements, BlockBytes: Integer): Pointer;
  public
    constructor Create(const ModuleFile: string);
    destructor Destroy; override;
    { Cached source ranges must remain immutable and at the same host address
      for this instance's lifetime. Qwen35Weights owns read-only mappings. }
    procedure ConfigureCache(const BudgetBytes: UInt64);
    procedure ReleaseCachedWeights;
    function Tensor(const Source: Pointer; const SourceBytes, Valid, Count: Int64;
      const Format: Integer; const Shape: TArray<Int64>;
      const Destination: TTVMFFIObjectHandle = nil): TTVMFFIObjectHandle;
    property CacheStats: TQwen35PackedCacheStats read FStats;
  end;

implementation

uses
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF};

const
  CudaSystem32Search: DWORD = $00000800;

procedure CheckCuda(const Code: Integer; const Operation: string);
begin
  RequireCpu(Code = 0, Format('%s failed with CUDA status %d', [Operation, Code]));
end;

function TQwen35PackedCuda.Entry(const Name: AnsiString): Pointer;
begin
  Result := GetProcAddress(FDriver, PAnsiChar(Name));
  RequireCpu(Result <> nil, 'Missing CUDA decoder entry: ' + string(Name));
end;

constructor TQwen35PackedCuda.Create(const ModuleFile: string);
var
  Load: TLoad;
  Image: TBytes;
  Stream: TFileStream;
begin
  inherited Create;
  FDriver := LoadLibraryExW('nvcuda.dll', 0, CudaSystem32Search);
  if FDriver = 0 then RaiseLastOSError;
  Load := TLoad(Entry('cuModuleLoadData'));
  FUnload := TUnload(Entry('cuModuleUnload'));
  FFunction := TFunction(Entry('cuModuleGetFunction'));
  FLaunch := TLaunch(Entry('cuLaunchKernel'));
  FZero := TZero(Entry('cuMemsetD32_v2'));
  FSync := TSync(Entry('cuCtxSynchronize'));
  FMemoryInfo := TMemoryInfo(Entry('cuMemGetInfo_v2'));
  Stream := TFileStream.Create(ModuleFile, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= 64 * 1024 * 1024), 'Invalid packed CUDA module size');
    SetLength(Image, Stream.Size);
    Stream.ReadBuffer(Image[0], Length(Image));
  finally
    Stream.Free;
  end;
  CheckCuda(Load(FModule, Pointer(Image)), 'Load packed CUDA module');
  FCache := TDictionary<string, TTVMFFIObjectHandle>.Create;
end;

destructor TQwen35PackedCuda.Destroy;
begin
  ClearCache;
  FCache.Free;
  if FModule <> nil then FUnload(FModule);
  if FDriver <> 0 then FreeLibrary(FDriver);
  inherited;
end;

procedure TQwen35PackedCuda.ClearCache;
var
  Value, Handle: TTVMFFIObjectHandle;
begin
  if FCache = nil then Exit;
  { Tensor synchronizes before returning or releasing its input, so no cached
    buffer remains in flight when this private operation is called. }
  for Value in FCache.Values do
  begin
    Handle := Value;
    TTVMRuntime.ObjectRelease(Handle);
  end;
  FCache.Clear;
  FStats.ResidentBytes := 0;
end;

procedure TQwen35PackedCuda.ReleaseCachedWeights;
begin
  CheckCuda(FSync(), 'Synchronize before releasing packed CUDA weights');
  ClearCache;
end;

procedure TQwen35PackedCuda.ConfigureCache(const BudgetBytes: UInt64);
begin
  RequireCpu(BudgetBytes <= UInt64(64) * 1024 * 1024 * 1024,
    'Qwen packed cache budget exceeds 64 GiB');
  ClearCache;
  FStats := Default(TQwen35PackedCacheStats);
  FStats.InitialBudgetBytes := BudgetBytes;
  FStats.BudgetBytes := BudgetBytes;
end;

function TQwen35PackedCuda.CacheKey(const Source: Pointer; const SourceBytes: Int64): string;
begin
  Result := IntToHex(NativeUInt(Source), SizeOf(Pointer) * 2) + ':' + IntToStr(SourceBytes);
end;

function TQwen35PackedCuda.Kernel(const Format: Integer; out Threads: Cardinal;
  out BlockElements, BlockBytes: Integer): Pointer;
var
  Name: AnsiString;
begin
  Threads := 32;
  BlockElements := 256;
  case Format of
    8:
      begin
        Name := 'q8_0';
        BlockElements := 32;
        BlockBytes := 34;
      end;
    11:
      begin
        Name := 'q3_K';
        Threads := 64;
        BlockBytes := 110;
      end;
    12:
      begin
        Name := 'q4_K';
        BlockBytes := 144;
      end;
    13:
      begin
        Name := 'q5_K';
        Threads := 64;
        BlockBytes := 176;
      end;
    14:
      begin
        Name := 'q6_K';
        Threads := 64;
        BlockBytes := 210;
      end;
    20:
      begin
        Name := 'iq4_nl';
        BlockElements := 32;
        BlockBytes := 18;
      end;
    21:
      begin
        Name := 'iq3_s';
        BlockBytes := 110;
      end;
    23:
      begin
        Name := 'iq4_xs';
        BlockBytes := 136;
      end;
  else
    raise ECpuGraph.Create('Unsupported packed CUDA tensor type');
  end;
  Name := 'cp_dequant_' + Name;
  if FFunctions[Format] = nil then
    CheckCuda(FFunction(FFunctions[Format], FModule, PAnsiChar(Name)), 'Resolve packed CUDA kernel');
  Result := FFunctions[Format];
end;

function TQwen35PackedCuda.Tensor(const Source: Pointer; const SourceBytes, Valid, Count: Int64;
  const Format: Integer; const Shape: TArray<Int64>;
  const Destination: TTVMFFIObjectHandle): TTVMFFIObjectHandle;
var
  PackedBuffer: TTVMFFIObjectHandle;
  Error: TTVMCallError;
  FunctionHandle: Pointer;
  Input, Output: UInt64;
  Blocks, Grid: Int64;
  Threads: Cardinal;
  BlockElements, BlockBytes: Integer;
  Arguments: array[0..2] of Pointer;
  Key: string;
  Borrowed: Boolean;
  Available, Total: NativeUInt;
  TransientBytes: UInt64;
  Target: PDLTensor;
  I: Integer;
  ExpectedStride: Int64;
begin
  Result := nil;
  PackedBuffer := nil;
  Borrowed := False;
  FunctionHandle := Kernel(Format, Threads, BlockElements, BlockBytes);
  RequireCpu((Source <> nil) and (SourceBytes > 0) and (Valid > 0) and (Valid <= Count) and
    (Count = TensorElements(Shape)) and
    (Valid mod BlockElements = 0), 'Invalid packed CUDA tensor range');
  Blocks := Valid div BlockElements;
  RequireCpu((Blocks <= High(Int64) div BlockBytes) and
    (SourceBytes = Blocks * BlockBytes), 'Packed CUDA source byte count differs');
  if Format = 20 then
    Grid := (Blocks + 7) div 8
  else
    Grid := Blocks;
  RequireCpu((Grid > 0) and (Grid <= High(Integer)), 'Packed CUDA grid is too large');
  RequireCpu(Count <= High(Int64) div SizeOf(Single), 'Packed CUDA result is too large');
  if Destination <> nil then
  begin
    Target := TTVMRuntime.TensorDLTensor(Destination);
    RequireCpu((Target^.data <> nil) and (Target^.device.device_type = kDLCUDA) and
      (Target^.device.device_id = 0) and (Target^.dtype.code = Ord(kDLFloat)) and
      (Target^.dtype.bits = 32) and (Target^.dtype.lanes = 1) and
      (Target^.byte_offset = 0) and
      (Target^.ndim = Length(Shape)) and (Target^.shape <> nil),
      'Incompatible packed CUDA destination tensor');
    for I := 0 to High(Shape) do
      RequireCpu(Target^.shape[I] = Shape[I], 'Packed CUDA destination shape differs');
    ExpectedStride := 1;
    for I := High(Shape) downto 0 do
    begin
      if Target^.strides <> nil then
        RequireCpu(Target^.strides[I] = ExpectedStride,
          'Packed CUDA destination is not contiguous');
      ExpectedStride := ExpectedStride * Shape[I];
    end;
  end;
  Key := '';
  if FStats.BudgetBytes <> 0 then
  begin
    TransientBytes := UInt64(SourceBytes) + 64 * 1024 * 1024;
    if Destination = nil then Inc(TransientBytes, UInt64(Count) * SizeOf(Single));
    CheckCuda(FMemoryInfo(Available, Total), 'Query packed CUDA cache memory');
    if (UInt64(Available) < TransientBytes) and (FCache.Count <> 0) then
    begin
      { Graph allocators and other GPU users can consume more than the load-time
        reserve. Reduce the retained budget instead of refilling a cache that
        would be cleared again on each traversal of the decoder. }
      FStats.BudgetBytes := FStats.ResidentBytes div 2;
      ClearCache;
      Inc(FStats.PressureClears);
    end;
    if FStats.BudgetBytes <> 0 then
    begin
      Key := CacheKey(Source, SourceBytes);
      Borrowed := FCache.TryGetValue(Key, PackedBuffer);
    end;
  end;
  try
    try
      if Borrowed then
        Inc(FStats.Hits)
      else begin
        Inc(FStats.Misses);
        PackedBuffer := DeviceTensor([SourceBytes], kDLUInt, 8, Source, kDLCUDA);
      end;
      if Destination = nil then
        RequireTVM(TTVMRuntime.AllocDeviceTensor(Shape, Ord(kDLFloat), 32, 1, Ord(kDLCUDA), 0, Result, @Error), @Error)
      else begin
        TTVMRuntime.ObjectRetain(Destination);
        Result := Destination;
      end;
      Input := UInt64(NativeUInt(TTVMRuntime.TensorDLTensor(PackedBuffer)^.data));
      Output := UInt64(NativeUInt(TTVMRuntime.TensorDLTensor(Result)^.data));
      Arguments[0] := @Input;
      Arguments[1] := @Output;
      Arguments[2] := @Blocks;
      if Count > Valid then
        CheckCuda(FZero(Output + UInt64(Valid) * 4, 0, Count - Valid), 'Zero packed CUDA tensor padding');
      CheckCuda(FLaunch(FunctionHandle, Grid, 1, 1, Threads, 1, 1, 0, nil, @Arguments[0], nil),
        'Decode packed CUDA tensor');
      CheckCuda(FSync(), 'Synchronize packed CUDA tensor');
      if not Borrowed and (Key <> '') and
        (UInt64(SourceBytes) <= FStats.BudgetBytes - FStats.ResidentBytes) then
        begin
          FCache.Add(Key, PackedBuffer);
          Inc(FStats.ResidentBytes, UInt64(SourceBytes));
          Borrowed := True;
        end;
    except
      if Assigned(FSync) then FSync();
      TTVMRuntime.ObjectRelease(Result);
      raise;
    end;
  finally
    if not Borrowed then TTVMRuntime.ObjectRelease(PackedBuffer);
  end;
end;

end.
