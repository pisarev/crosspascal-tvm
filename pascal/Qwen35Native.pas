{ ************************************************************************** }
{                                                                            }
{ Qwen35Native                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit Qwen35Native;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.Json, DLPack,
  TVM, TVM.CpuGraph, TVM.CudaExecution, QwenCuda, Qwen35Weights, Qwen35PackedCuda,
  Qwen35CudaBuffers, Qwen35Prompt;

type
  TQwen35Timings = record
    Parameters: Double;
    Layers: Double;
    OutputParameters: Double;
    OutputMath: Double;
  end;

  { Text decoder with bounded, layer-wise weight streaming. Source tensors are
    never requantized. Graphs perform all decoder and output-head math in TVM. }
  TQwen35Native = class
  private
    type
      TInput = record
        Name: string;
        Shape: TArray<Int64>;
        DType: Integer;
      end;
      TPlan = class
        Graph: TCpuGraph;
        Inputs: TArray<TInput>;
        Outputs: TArray<string>;
        constructor Create(const Folder: string; const Spec: TJSONObject; const Device: TDLDeviceType);
        destructor Destroy; override;
        function InputIndex(const Name: string): Integer;
        function HasInput(const Name: string): Boolean;
      end;
      TLayer = record
        Plan: TPlan;
        Parameters: TArray<TQwen35Weight>;
        PooledSlots: TArray<Integer>;
        States: TTensorHandles;
      end;
  private
    FFolder: string;
    FDevice: TDLDeviceType;
    FCuda: TCudaExecution;
    FFiles: TQwen35WeightFiles;
    FDecodedPool: TQwen35CudaBufferPool;
    FLinear: TPlan;
    FFull: TPlan;
    FHead: TPlan;
    FPrefillLinear: TPlan;
    FPrefillFull: TPlan;
    FPrefillChunk: Integer;
    FLayers: TArray<TLayer>;
    FEmbedding: TQwen35Weight;
    FNorm: TQwen35Weight;
    FProjection: TQwen35Weight;
    FHidden: Integer;
    FVocab: Integer;
    FEffectiveVocab: Integer;
    FContext: Integer;
    FHeadRows: Integer;
    FPosition: Integer;
    FRopeDelta: Int64;
    FLegacyPositions: Boolean;
    FTimings: TQwen35Timings;
    procedure CheckStop(const Stop: TQwenStopRequested);
    procedure ResetStates;
    procedure ConfigureDecodedPool;
    function ParameterTensor(const Weight: TQwen35Weight; const Slot: Integer): TTVMFFIObjectHandle;
    function Parameters(const LayerIndex: Integer; const Stop: TQwenStopRequested): TTensorHandles;
    procedure ReleaseParameters(const LayerIndex: Integer; var Weights: TTensorHandles);
    function RunLayer(const LayerIndex, CachePosition: Integer; const Rope: TQwen35RopePosition;
      const Hidden: TArray<Single>; const Weights: TTensorHandles;
      const Stop: TQwenStopRequested): TArray<Single>;
    function RunPrefillLayer(const LayerIndex, First: Integer; const Batch: TPlan;
      const Prompt: TQwen35PreparedPrompt; const Hidden: TArray<Single>;
      const Weights: TTensorHandles; const Stop: TQwenStopRequested): TArray<Single>;
    function Logits(const Hidden: TArray<Single>; const Stop: TQwenStopRequested): TArray<Single>;
  public
    constructor Create(const Folder, RuntimeDirectory, DecoderLibrary: string);
    destructor Destroy; override;
    function Prefill(const Tokens: array of Integer; const Stop: TQwenStopRequested = nil;
      const Progress: TQwenProgress = nil): TArray<Single>;
    { Row I is cache position I; visual rows use Prompt.Embeddings instead of
      token embeddings. Decode continues with the prompt's rotary delta. }
    function PrefillPrepared(const Prompt: TQwen35PreparedPrompt;
      const Stop: TQwenStopRequested = nil; const Progress: TQwenProgress = nil): TArray<Single>;
    function Decode(const Token: Integer; const Stop: TQwenStopRequested = nil;
      const Progress: TQwenProgress = nil): TArray<Single>;
    { Discard the previous request and resident decoder buffers before vision.
      Capacities, cache budget and cumulative counters are retained. }
    procedure PrepareImageEncoding;
    property CudaExecution: TCudaExecution read FCuda;
    property HiddenSize: Integer read FHidden;
    property ContextTokens: Integer read FContext;
    property Position: Integer read FPosition;
    property RopeDelta: Int64 read FRopeDelta;
    property Timings: TQwen35Timings read FTimings;
    function PackedCacheStats: TQwen35PackedCacheStats;
    function DecodedPoolStats: TQwen35DecodedPoolStats;
  end;

implementation

uses
  {$IFDEF FPC}Math, Windows{$ELSE}System.Math, Winapi.Windows{$ENDIF};

function IsState(const Name: string): Boolean;
begin
  Result := (Name = 'conv_state') or (Name = 'recurrent_state') or
    (Name = 'key_cache') or (Name = 'value_cache');
end;

function IsPositionInput(const Name: string): Boolean;
begin
  Result := (Name = 'position') or (Name = 'cache_position') or (Name = 'rope_position');
end;

procedure SameShape(const A, B: TArray<Int64>);
var
  I: Integer;
begin
  RequireCpu(Length(A) = Length(B), 'Qwen graph parameter rank differs');
  for I := 0 to High(A) do RequireCpu(A[I] = B[I], 'Qwen graph parameter shape differs');
end;

constructor TQwen35Native.TPlan.Create(const Folder: string; const Spec: TJSONObject;
  const Device: TDLDeviceType);
var
  Values: TJSONArray;
  Row: TJSONObject;
  I, J: Integer;
begin
  inherited Create;
  Values := ModelArray(Spec.GetValue('inputs'), 'graph inputs');
  RequireCpu((Values.Count > 0) and (Values.Count <= 64), 'Invalid Qwen graph input count');
  SetLength(Inputs, Values.Count);
  for I := 0 to Values.Count - 1 do
  begin
    Row := ModelObject(Values.Items[I], 'graph input');
    Inputs[I].Name := ModelText(Row.GetValue('name'));
    RequireCpu(Inputs[I].Name <> '', 'Empty Qwen input name');
    for J := 0 to I - 1 do
      RequireCpu(Inputs[J].Name <> Inputs[I].Name, 'Duplicate Qwen graph input');
    Inputs[I].Shape := ModelShape(Row.GetValue('shape'));
    Inputs[I].DType := ModelInt(Row.GetValue('dtype'), 'graph input type');
    RequireCpu(Inputs[I].DType in [1, 7, 16], 'Unsupported Qwen graph input type');
  end;
  Values := ModelArray(Spec.GetValue('outputs'), 'graph outputs');
  RequireCpu((Values.Count > 0) and (Values.Count <= 3), 'Invalid Qwen graph outputs');
  SetLength(Outputs, Values.Count);
  for I := 0 to Values.Count - 1 do Outputs[I] := ModelText(Values.Items[I]);
  Graph := TCpuGraph.Create(BundleMember(Folder, ModelText(Spec.GetValue('library'))), Device);
end;

destructor TQwen35Native.TPlan.Destroy;
begin
  Graph.Free;
  inherited;
end;

function TQwen35Native.TPlan.HasInput(const Name: string): Boolean;
var
  I: Integer;
begin
  for I := 0 to High(Inputs) do
    if Inputs[I].Name = Name then Exit(True);
  Result := False;
end;

function TQwen35Native.TPlan.InputIndex(const Name: string): Integer;
var
  I: Integer;
begin
  for I := 0 to High(Inputs) do
    if Inputs[I].Name = Name then Exit(I);
  raise ECpuGraph.Create('Missing Qwen graph input: ' + Name);
end;

constructor TQwen35Native.Create(const Folder, RuntimeDirectory, DecoderLibrary: string);
var
  Data: TJSONValue;
  Root, Config, Graphs, Row, Values, Hashes, PrefillSpec: TJSONObject;
  Layers: TJSONArray;
  I, J: Integer;
  RequestedCache, Chunk: Int64;
  CacheBudget, CacheReserve, LayerBytes, MaximumLayerBytes, StateBytes, Available: UInt64;
  UnpooledBytes, MaximumUnpooledBytes: UInt64;
  Name, Kind, PackedModule: string;
  Entered: Boolean;

  function ReadWeight(const Value: TJSONValue): TQwen35Weight;
  begin
    Result := FFiles.Weight(Value);
    RequireCpu(Hashes.GetValue(Result.FileName) <> nil, 'Unhashed Qwen parameter artifact');
  end;

  function ReadPlan(const Value: TJSONValue): TPlan;
  var
    Spec: TJSONObject;
  begin
    Spec := ModelObject(Value, 'compiled graph');
    RequireCpu(Hashes.GetValue(ModelText(Spec.GetValue('library'))) <> nil, 'Unhashed Qwen graph');
    Result := TPlan.Create(FFolder, Spec, FDevice);
  end;

  procedure RequirePositionInput(const Name: string; const Width: Int64);
  var
    Index: Integer;
  begin
    Index := FFull.InputIndex(Name);
    RequireCpu((FFull.Inputs[Index].DType = 7) and (Length(FFull.Inputs[Index].Shape) = 1) and
      (FFull.Inputs[Index].Shape[0] = Width), 'Invalid Qwen position graph input: ' + Name);
  end;

  { Batch graphs differ from the single-step graphs only in the hidden and
    position inputs; parameters, states and outputs match by name. }
  procedure MatchPrefill(const Batch, Step: TPlan);
  var
    K, Index, Matches: Integer;
    Item: string;
  begin
    RequireCpu((Length(Batch.Inputs) = Length(Step.Inputs)) and
      (Length(Batch.Outputs) = Length(Step.Outputs)), 'Qwen prefill graph interface differs');
    for K := 0 to High(Batch.Inputs) do
    begin
      Item := Batch.Inputs[K].Name;
      Index := Step.InputIndex(Item);
      if Item = 'hidden' then
        RequireCpu((Batch.Inputs[K].DType = 1) and (Length(Batch.Inputs[K].Shape) = 3) and
          (Batch.Inputs[K].Shape[0] = 1) and (Batch.Inputs[K].Shape[1] = FPrefillChunk) and
          (Batch.Inputs[K].Shape[2] = FHidden), 'Invalid Qwen prefill hidden input')
      else if Item = 'cache_position' then
        RequireCpu((Batch.Inputs[K].DType = 7) and (Length(Batch.Inputs[K].Shape) = 1) and
          (Batch.Inputs[K].Shape[0] = FPrefillChunk), 'Invalid Qwen prefill cache positions')
      else if Item = 'rope_position' then
        RequireCpu((Batch.Inputs[K].DType = 7) and (Length(Batch.Inputs[K].Shape) = 2) and
          (Batch.Inputs[K].Shape[0] = FPrefillChunk) and (Batch.Inputs[K].Shape[1] = 3),
          'Invalid Qwen prefill rope positions')
      else begin
        RequireCpu(not IsPositionInput(Item) and (Batch.Inputs[K].DType = Step.Inputs[Index].DType),
          'Qwen prefill graph input differs: ' + Item);
        SameShape(Batch.Inputs[K].Shape, Step.Inputs[Index].Shape);
      end;
    end;
    for K := 0 to High(Batch.Outputs) do
    begin
      Item := Batch.Outputs[K];
      Matches := 0;
      for Index := 0 to High(Step.Outputs) do
        if Step.Outputs[Index] = Item then Inc(Matches);
      for Index := 0 to K - 1 do
        if Batch.Outputs[Index] = Item then Inc(Matches);
      RequireCpu(
        (Matches = 1) and ((Item = 'updated_hidden') or ((Copy(Item, 1, 8) = 'updated_') and IsState(Copy(Item, 9, MaxInt)) and Batch.HasInput(Copy(Item, 9, MaxInt)))),
        'Qwen prefill graph output differs: ' + Item
      );
    end;
  end;
begin
  inherited Create;
  FPosition := -1;
  FFolder := ExpandFileName(Folder);
  Data := ReadLanguageManifest(FFolder);
  Entered := False;
  try
    Root := ModelObject(Data, 'Qwen manifest');
    RequireCpu(
      (ModelText(Root.GetValue('schema')) = 'crosspascal.qwen35.text.v1') and (ModelText(Root.GetValue('architecture')) = 'qwen3_5'),
      'Expected a native dense Qwen 3.5/3.8 text bundle'
    );
    if ModelText(Root.GetValue('device')) = 'cuda' then
      FDevice := kDLCUDA
    else begin
      RequireCpu(ModelText(Root.GetValue('device')) = 'cpu', 'Unknown Qwen target device');
      FDevice := kDLCPU;
    end;
    Config := ModelObject(Root.GetValue('config'), 'Qwen dimensions');
    FHidden := ModelInt(Config.GetValue('hidden'), 'hidden width');
    FVocab := ModelInt(Config.GetValue('vocab'), 'vocabulary');
    FEffectiveVocab := ModelInt(Root.GetValue('effective_vocab'), 'tokenizer vocabulary');
    FContext := ModelInt(Root.GetValue('context'), 'compiled context');
    FHeadRows := ModelInt(Root.GetValue('head_rows'), 'head rows');
    RequireCpu(
      (FHidden > 0) and (FHidden <= 65536) and (FVocab > 0) and (FVocab <= 16777216) and (FEffectiveVocab > 0) and (FEffectiveVocab <= FVocab) and (FContext > 0) and (FContext <= 1048576) and (FHeadRows > 0) and (FHeadRows <= FVocab),
      'Invalid Qwen runtime dimensions'
    );
    Hashes := ModelObject(Root.GetValue('artifact_sha256'), 'artifact hashes');
    VerifyArtifacts(FFolder, Hashes);
    InitializeRuntime(IncludeTrailingPathDelimiter(RuntimeDirectory), FDevice = kDLCUDA);
    if FDevice = kDLCUDA then
    begin
      FCuda := TCudaExecution.Create;
      FCuda.RequireArchitecture(ModelText(Root.GetValue('cuda_arch')));
      FCuda.Enter;
      Entered := True;
    end;
    PackedModule := ModelText(Root.GetValue('packed_decoder'));
    if PackedModule <> '' then
    begin
      RequireCpu((FDevice = kDLCUDA) and (Hashes.GetValue(PackedModule) <> nil),
        'Unhashed or incompatible packed CUDA decoder');
      PackedModule := BundleMember(FFolder, PackedModule);
    end;
    FFiles := TQwen35WeightFiles.Create(FFolder, DecoderLibrary, PackedModule);
    Graphs := ModelObject(Root.GetValue('graphs'), 'decoder graph families');
    FLinear := ReadPlan(Graphs.GetValue('linear'));
    FFull := ReadPlan(Graphs.GetValue('full'));
    FHead := ReadPlan(Graphs.GetValue('head'));
    { Legacy graphs use one scalar position for cache and RoPE; current graphs
      take a cache index and three MRoPE axes. }
    FLegacyPositions := FFull.HasInput('position');
    if FLegacyPositions then
    begin
      RequireCpu(not FFull.HasInput('cache_position') and not FFull.HasInput('rope_position'),
        'Qwen full-attention graph mixes position contracts');
      RequirePositionInput('position', 1);
    end
    else begin
      RequirePositionInput('cache_position', 1);
      RequirePositionInput('rope_position', 3);
    end;
    for I := 0 to High(FLinear.Inputs) do
      RequireCpu(not IsPositionInput(FLinear.Inputs[I].Name),
        'Qwen linear-attention graph has a position input');
    { Optional batched prefill; without it prefill stays row by row. }
    if Root.GetValue('prefill_graphs') <> nil then
    begin
      PrefillSpec := ModelObject(Root.GetValue('prefill_graphs'), 'prefill graphs');
      Chunk := ModelInt(PrefillSpec.GetValue('chunk_size'), 'prefill chunk size');
      RequireCpu((Chunk >= 2) and (Chunk <= 64) and (Chunk <= FContext) and not FLegacyPositions,
        'Invalid Qwen prefill graphs');
      FPrefillChunk := Chunk;
      FPrefillLinear := ReadPlan(PrefillSpec.GetValue('linear'));
      FPrefillFull := ReadPlan(PrefillSpec.GetValue('full'));
      MatchPrefill(FPrefillLinear, FLinear);
      MatchPrefill(FPrefillFull, FFull);
    end;
    Values := ModelObject(Root.GetValue('weights'), 'global parameters');
    FEmbedding := ReadWeight(Values.GetValue('embed_tokens.weight'));
    FNorm := ReadWeight(Values.GetValue('norm.weight'));
    FProjection := ReadWeight(Values.GetValue('lm_head.weight'));
    SameShape(FEmbedding.Shape, [FVocab, FHidden]);
    SameShape(FNorm.Shape, [FHidden]);
    SameShape(FProjection.Shape, [FVocab, FHidden]);
    Layers := ModelArray(Root.GetValue('layers'), 'decoder layers');
    RequireCpu((Layers.Count > 0) and (Layers.Count <= 1024) and (Layers.Count = ModelInt(Config.GetValue('layers'), 'layer count')),
      'Invalid decoder layer count');
    SetLength(FLayers, Layers.Count);
    for I := 0 to Layers.Count - 1 do
    begin
      Row := ModelObject(Layers.Items[I], 'decoder layer');
      Kind := ModelText(Row.GetValue('kind'));
      if Kind = 'linear' then
        FLayers[I].Plan := FLinear
      else begin
        RequireCpu(Kind = 'full', 'Unknown Qwen layer kind');
        FLayers[I].Plan := FFull;
      end;
      Values := ModelObject(Row.GetValue('weights'), 'layer weights');
      SetLength(FLayers[I].Parameters, Length(FLayers[I].Plan.Inputs));
      SetLength(FLayers[I].States, Length(FLayers[I].Plan.Inputs));
      for J := 0 to High(FLayers[I].Parameters) do
      begin
        Name := FLayers[I].Plan.Inputs[J].Name;
        if (Name = 'hidden') or IsPositionInput(Name) or IsState(Name) then Continue;
        FLayers[I].Parameters[J] := ReadWeight(Values.GetValue(Name));
        SameShape(FLayers[I].Parameters[J].Shape, FLayers[I].Plan.Inputs[J].Shape);
        RequireCpu((FLayers[I].Plan.Inputs[J].DType = 16) = (FLayers[I].Parameters[J].Encoding = 'bf16'),
          'Qwen parameter storage differs from graph');
      end;
    end;
    if Root.GetValue('cuda_decoded_buffer_pool') <> nil then
    begin
      RequireCpu(JSONIsBoolean(Root.GetValue('cuda_decoded_buffer_pool')),
        'Qwen decoded buffer pool must be a JSON boolean');
      if JSONIsTrue(Root.GetValue('cuda_decoded_buffer_pool')) then
      begin
        RequireCpu((FDevice = kDLCUDA) and (PackedModule <> ''),
          'Decoded buffer pooling requires a packed CUDA decoder');
        ConfigureDecodedPool;
      end;
    end;
    RequestedCache := 0;
    if Root.GetValue('cuda_packed_cache_bytes') <> nil then
      RequestedCache := ModelInt(Root.GetValue('cuda_packed_cache_bytes'), 'packed cache budget');
    RequireCpu((RequestedCache >= 0) and (RequestedCache <= Int64(64) * 1024 * 1024 * 1024),
      'Invalid Qwen packed CUDA cache budget');
    if RequestedCache <> 0 then
    begin
      RequireCpu((FDevice = kDLCUDA) and (PackedModule <> ''),
        'Packed caching requires a CUDA GGUF decoder');
      MaximumLayerBytes := 0;
      MaximumUnpooledBytes := 0;
      StateBytes := 0;
      for I := 0 to High(FLayers) do
      begin
        LayerBytes := 0;
        UnpooledBytes := 0;
        for J := 0 to High(FLayers[I].Parameters) do
        begin
          if FLayers[I].Parameters[J].FileName <> '' then
          begin
            Inc(LayerBytes, UInt64(FLayers[I].Parameters[J].Elements) * SizeOf(Single));
            if (FDecodedPool = nil) or (FLayers[I].PooledSlots[J] < 0) then
              Inc(UnpooledBytes, UInt64(FLayers[I].Parameters[J].Elements) * SizeOf(Single));
          end;
          if IsState(FLayers[I].Plan.Inputs[J].Name) then
            Inc(StateBytes, UInt64(TensorElements(FLayers[I].Plan.Inputs[J].Shape)) * SizeOf(Single));
        end;
        if LayerBytes > MaximumLayerBytes then MaximumLayerBytes := LayerBytes;
        if UnpooledBytes > MaximumUnpooledBytes then MaximumUnpooledBytes := UnpooledBytes;
      end;
      if (FDecodedPool <> nil) and
        (FDecodedPool.Stats.PlannedBytes + MaximumUnpooledBytes > MaximumLayerBytes) then
          MaximumLayerBytes := FDecodedPool.Stats.PlannedBytes + MaximumUnpooledBytes;
      { Retain space for a complete decoded layer, old/new recurrent and KV
        state, and one GiB of graph workspaces and device headroom. }
      CacheReserve := MaximumLayerBytes + StateBytes * 2 + UInt64(1024) * 1024 * 1024;
      Available := FCuda.FreeBytes;
      CacheBudget := 0;
      if Available > CacheReserve then CacheBudget := Available - CacheReserve;
      if CacheBudget > UInt64(RequestedCache) then CacheBudget := UInt64(RequestedCache);
      FFiles.ConfigurePackedCache(CacheBudget);
    end;
  finally
    if Entered then FCuda.Leave;
    Data.Free;
  end;
end;

procedure TQwen35Native.PrepareImageEncoding;
begin
  if FCuda <> nil then FCuda.Enter;
  try
    if FCuda <> nil then FCuda.Synchronize;
    ResetStates;
    if FDecodedPool <> nil then FDecodedPool.ReleaseStorage;
    FFiles.ReleaseCudaCache;
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
end;

function TQwen35Native.PackedCacheStats: TQwen35PackedCacheStats;
begin
  Result := FFiles.PackedCacheStats;
end;

function TQwen35Native.DecodedPoolStats: TQwen35DecodedPoolStats;
begin
  Result := Default(TQwen35DecodedPoolStats);
  if FDecodedPool <> nil then Result := FDecodedPool.Stats;
end;

procedure TQwen35Native.ConfigureDecodedPool;
var
  Capacities: TArray<UInt64>;
  Order: TArray<Integer>;
  I, J, K, Index: Integer;
  Bytes: UInt64;
begin
  RequireCpu(FFiles.CanPoolCudaTensor(FProjection), 'Qwen output head cannot use decoded buffers');
  Capacities := nil;
  for I := 0 to High(FLayers) do
  begin
    SetLength(FLayers[I].PooledSlots, Length(FLayers[I].Parameters));
    SetLength(Order, 0);
    for J := 0 to High(FLayers[I].Parameters) do
    begin
      FLayers[I].PooledSlots[J] := -1;
      if not FFiles.CanPoolCudaTensor(FLayers[I].Parameters[J]) then Continue;
      Index := Length(Order);
      SetLength(Order, Index + 1);
      { Stable descending size order pairs simultaneously live weights with
        distinct slots and bounds each slot across all decoder layers. }
      while (Index > 0) and
        (FLayers[I].Parameters[Order[Index - 1]].Elements < FLayers[I].Parameters[J].Elements) do
        begin
          Order[Index] := Order[Index - 1];
          Dec(Index);
        end;
      Order[Index] := J;
    end;
    if Length(Capacities) < Length(Order) then SetLength(Capacities, Length(Order));
    for K := 0 to High(Order) do
    begin
      J := Order[K];
      FLayers[I].PooledSlots[J] := K;
      RequireCpu(FLayers[I].Parameters[J].Elements <= High(Int64) div SizeOf(Single),
        'Qwen decoded parameter is too large');
      Bytes := UInt64(FLayers[I].Parameters[J].Elements) * SizeOf(Single);
      if Bytes > Capacities[K] then Capacities[K] := Bytes;
    end;
  end;
  if Length(Capacities) = 0 then SetLength(Capacities, 1);
  Bytes := UInt64(FHeadRows) * UInt64(FHidden) * SizeOf(Single);
  if Bytes > Capacities[0] then Capacities[0] := Bytes;
  FDecodedPool := TQwen35CudaBufferPool.Create(Capacities);
end;

function TQwen35Native.ParameterTensor(const Weight: TQwen35Weight;
  const Slot: Integer): TTVMFFIObjectHandle;
var
  Destination: TTVMFFIObjectHandle;
begin
  Destination := nil;
  try
    if Slot >= 0 then Destination := FDecodedPool.Rent(Slot, Weight.Shape);
    Result := FFiles.Tensor(Weight, FDevice, Destination);
  finally
    TTVMRuntime.ObjectRelease(Destination);
  end;
end;

destructor TQwen35Native.Destroy;
begin
  if FCuda <> nil then FCuda.Enter;
  try
    if FCuda <> nil then FCuda.Synchronize;
    ResetStates;
    FHead.Free;
    FPrefillFull.Free;
    FPrefillLinear.Free;
    FFull.Free;
    FLinear.Free;
    FDecodedPool.Free;
    FFiles.Free;
  finally
    if FCuda <> nil then FCuda.Leave;
    FCuda.Free;
  end;
  inherited;
end;

procedure TQwen35Native.CheckStop(const Stop: TQwenStopRequested);
begin
  if Assigned(Stop) and Stop() then raise ECpuGraphStopped.Create('Generation stopped');
end;

procedure TQwen35Native.ResetStates;
var
  I, J: Integer;
begin
  for I := 0 to High(FLayers) do
    for J := 0 to High(FLayers[I].States) do TTVMRuntime.ObjectRelease(FLayers[I].States[J]);
  FPosition := -1;
  FRopeDelta := 0;
end;

function TQwen35Native.Parameters(const LayerIndex: Integer;
  const Stop: TQwenStopRequested): TTensorHandles;
var
  I, Slot: Integer;
  Started: UInt64;
begin
  Started := GetTickCount64;
  SetLength(Result, Length(FLayers[LayerIndex].Parameters));
  if FDecodedPool <> nil then FDecodedPool.BeginGroup;
  try
    for I := 0 to High(Result) do
    begin
      CheckStop(Stop);
      if FLayers[LayerIndex].Parameters[I].FileName <> '' then
      begin
        Slot := -1;
        if FDecodedPool <> nil then Slot := FLayers[LayerIndex].PooledSlots[I];
        Result[I] := ParameterTensor(FLayers[LayerIndex].Parameters[I], Slot);
      end;
    end;
  except
    if FCuda <> nil then FCuda.Synchronize;
    ReleaseTensors(Result);
    if FDecodedPool <> nil then FDecodedPool.EndGroup;
    raise;
  end;
  FTimings.Parameters := FTimings.Parameters + (GetTickCount64 - Started) / 1000.0;
end;

procedure TQwen35Native.ReleaseParameters(const LayerIndex: Integer; var Weights: TTensorHandles);
var
  I: Integer;
begin
  if FCuda <> nil then FCuda.Synchronize;
  ReleaseTensors(Weights);
  if FDecodedPool <> nil then FDecodedPool.EndGroup;
  for I := 0 to High(FLayers[LayerIndex].Parameters) do
    if FLayers[LayerIndex].Parameters[I].FileName <> '' then
      FFiles.ReleaseResidentRange(FLayers[LayerIndex].Parameters[I], 0,
        FLayers[LayerIndex].Parameters[I].Elements);
end;

function TQwen35Native.RunLayer(const LayerIndex, CachePosition: Integer;
  const Rope: TQwen35RopePosition; const Hidden: TArray<Single>; const Weights: TTensorHandles;
  const Stop: TQwenStopRequested): TArray<Single>;
var
  Plan: TPlan;
  Args: TTensorHandles;
  Reply: TTVMFFIAny;
  Output, HiddenInput, PositionInput, RopeInput: TTVMFFIObjectHandle;
  Index, I: Integer;
  TokenPosition: Int64;
  RopeValues: array[0..2] of Int64;
  Name: string;
  Started: UInt64;
begin
  Started := GetTickCount64;
  CheckStop(Stop);
  RequireCpu(Length(Hidden) = FHidden, 'Decoder input hidden width differs');
  Plan := FLayers[LayerIndex].Plan;
  Args := Copy(Weights);
  HiddenInput := nil;
  PositionInput := nil;
  RopeInput := nil;
  Output := nil;
  Reply := Default(TTVMFFIAny);
  try
    for I := 0 to High(Args) do
    begin
      Name := Plan.Inputs[I].Name;
      if Name = 'hidden' then
      begin
        HiddenInput := DeviceTensor(Plan.Inputs[I].Shape, kDLFloat, 32, Pointer(Hidden), FDevice);
        Args[I] := HiddenInput;
      end
      else if (Name = 'position') or (Name = 'cache_position') then
      begin
        { A legacy scalar position drives both the cache and RoPE, so it can
          only represent diagonal text positions without a rotary delta. }
        RequireCpu(
          (Name = 'cache_position') or ((Rope[0] = CachePosition) and (Rope[1] = CachePosition) and (Rope[2] = CachePosition)),
          'This Qwen bundle has a text-only position graph; rebuild its graphs for multimodal positions'
        );
        TokenPosition := CachePosition;
        PositionInput := DeviceTensor(Plan.Inputs[I].Shape, kDLInt, 64, @TokenPosition, FDevice);
        Args[I] := PositionInput;
      end
      else if Name = 'rope_position' then
      begin
        RopeValues[0] := Rope[0];
        RopeValues[1] := Rope[1];
        RopeValues[2] := Rope[2];
        RopeInput := DeviceTensor(Plan.Inputs[I].Shape, kDLInt, 64, @RopeValues[0], FDevice);
        Args[I] := RopeInput;
      end
      else if IsState(Name) then
      begin
        if FLayers[LayerIndex].States[I] = nil then
          FLayers[LayerIndex].States[I] := DeviceTensor(Plan.Inputs[I].Shape, kDLFloat, 32, nil, FDevice);
        Args[I] := FLayers[LayerIndex].States[I];
      end;
      RequireCpu(Args[I] <> nil, 'Unbound Qwen graph input: ' + Name);
    end;
    Reply := Plan.Graph.Run(Args, Stop);
    for I := 0 to High(Plan.Outputs) do
    begin
      Output := TensorItem(Reply.VObj, I);
      Name := Plan.Outputs[I];
      if Name = 'updated_hidden' then
        Result := ReadFloatTensor(Output)
      else begin
        RequireCpu(Copy(Name, 1, 8) = 'updated_', 'Unknown decoder output');
        Delete(Name, 1, 8);
        RequireCpu(IsState(Name), 'Unknown decoder state output');
        Index := Plan.InputIndex(Name);
        TTVMRuntime.ObjectRelease(FLayers[LayerIndex].States[Index]);
        FLayers[LayerIndex].States[Index] := Output;
        Output := nil;
      end;
      if Output <> nil then TTVMRuntime.ObjectRelease(Output);
    end;
    RequireCpu(Length(Result) = FHidden, 'Decoder hidden width differs');
    CheckStop(Stop);
  finally
    if FCuda <> nil then FCuda.Synchronize;
    if Output <> nil then TTVMRuntime.ObjectRelease(Output);
    if Reply.VObj <> nil then TTVMRuntime.ObjectRelease(Reply.VObj);
    TTVMRuntime.ObjectRelease(RopeInput);
    TTVMRuntime.ObjectRelease(PositionInput);
    TTVMRuntime.ObjectRelease(HiddenInput);
    FTimings.Layers := FTimings.Layers + (GetTickCount64 - Started) / 1000.0;
  end;
end;

{ Runs prompt rows First .. First + FPrefillChunk - 1 through one batch graph.
  Weights and states are bound by name through the single-step plan, whose
  input indexes own FLayers states. Replaced states are released only after
  device synchronization. Returns FPrefillChunk * FHidden updated values. }
function TQwen35Native.RunPrefillLayer(const LayerIndex, First: Integer; const Batch: TPlan;
  const Prompt: TQwen35PreparedPrompt; const Hidden: TArray<Single>; const Weights: TTensorHandles;
  const Stop: TQwenStopRequested): TArray<Single>;
var
  Plan: TPlan;
  Args, Previous: TTensorHandles;
  Reply: TTVMFFIAny;
  Output, HiddenInput, PositionInput, RopeInput: TTVMFFIObjectHandle;
  CachePositions, RopePositions: TArray<Int64>;
  Index, I, K: Integer;
  Name: string;
  Started: UInt64;
begin
  Started := GetTickCount64;
  Result := nil;
  CheckStop(Stop);
  RequireCpu((First >= 0) and (First <= Length(Prompt.Rows) - FPrefillChunk) and
    (Length(Hidden) = FPrefillChunk * FHidden), 'Invalid Qwen prefill chunk');
  Plan := FLayers[LayerIndex].Plan;
  SetLength(Args, Length(Batch.Inputs));
  SetLength(Previous, Length(Plan.Inputs));
  HiddenInput := nil;
  PositionInput := nil;
  RopeInput := nil;
  Output := nil;
  Reply := Default(TTVMFFIAny);
  try
    for I := 0 to High(Args) do
    begin
      Name := Batch.Inputs[I].Name;
      if Name = 'hidden' then
      begin
        HiddenInput := DeviceTensor(Batch.Inputs[I].Shape, kDLFloat, 32, Pointer(Hidden), FDevice);
        Args[I] := HiddenInput;
      end
      else if Name = 'cache_position' then
      begin
        SetLength(CachePositions, FPrefillChunk);
        for K := 0 to FPrefillChunk - 1 do CachePositions[K] := First + K;
        PositionInput := DeviceTensor(Batch.Inputs[I].Shape, kDLInt, 64, Pointer(CachePositions), FDevice);
        Args[I] := PositionInput;
      end
      else if Name = 'rope_position' then
      begin
        SetLength(RopePositions, FPrefillChunk * 3);
        for K := 0 to FPrefillChunk - 1 do
        begin
          RopePositions[K * 3] := Prompt.Rows[First + K].Rope[0];
          RopePositions[K * 3 + 1] := Prompt.Rows[First + K].Rope[1];
          RopePositions[K * 3 + 2] := Prompt.Rows[First + K].Rope[2];
        end;
        RopeInput := DeviceTensor(Batch.Inputs[I].Shape, kDLInt, 64, Pointer(RopePositions), FDevice);
        Args[I] := RopeInput;
      end
      else begin
        Index := Plan.InputIndex(Name);
        if IsState(Name) then
        begin
          if FLayers[LayerIndex].States[Index] = nil then
            FLayers[LayerIndex].States[Index] := DeviceTensor(Plan.Inputs[Index].Shape, kDLFloat,
              32, nil, FDevice);
          Args[I] := FLayers[LayerIndex].States[Index];
        end
        else
          Args[I] := Weights[Index];
      end;
      RequireCpu(Args[I] <> nil, 'Unbound Qwen prefill graph input: ' + Name);
    end;
    Reply := Batch.Graph.Run(Args, Stop);
    for I := 0 to High(Batch.Outputs) do
    begin
      Output := TensorItem(Reply.VObj, I);
      Name := Batch.Outputs[I];
      if Name = 'updated_hidden' then
        Result := ReadFloatTensor(Output)
      else begin
        RequireCpu(Copy(Name, 1, 8) = 'updated_', 'Unknown decoder output');
        Delete(Name, 1, 8);
        RequireCpu(IsState(Name), 'Unknown decoder state output');
        Index := Plan.InputIndex(Name);
        Previous[Index] := FLayers[LayerIndex].States[Index];
        FLayers[LayerIndex].States[Index] := Output;
        Output := nil;
      end;
      TTVMRuntime.ObjectRelease(Output);
      Output := nil;
    end;
    RequireCpu(Length(Result) = FPrefillChunk * FHidden, 'Decoder prefill hidden width differs');
    CheckStop(Stop);
  finally
    if FCuda <> nil then FCuda.Synchronize;
    TTVMRuntime.ObjectRelease(Output);
    if Reply.VObj <> nil then TTVMRuntime.ObjectRelease(Reply.VObj);
    ReleaseTensors(Previous);
    TTVMRuntime.ObjectRelease(RopeInput);
    TTVMRuntime.ObjectRelease(PositionInput);
    TTVMRuntime.ObjectRelease(HiddenInput);
    FTimings.Layers := FTimings.Layers + (GetTickCount64 - Started) / 1000.0;
  end;
end;

function TQwen35Native.Logits(const Hidden: TArray<Single>;
  const Stop: TQwenStopRequested): TArray<Single>;
var
  Args: TTensorHandles;
  Reply: TTVMFFIAny;
  Values: TArray<Single>;
  First, Valid, Take, I, MatrixIndex: Integer;
  Started: UInt64;
  Destination: TTVMFFIObjectHandle;
begin
  SetLength(Result, FEffectiveVocab);
  SetLength(Args, Length(FHead.Inputs));
  Reply := Default(TTVMFFIAny);
  MatrixIndex := FHead.InputIndex('lm_head.weight');
  try
    Args[FHead.InputIndex('hidden')] := DeviceTensor([1, 1, FHidden], kDLFloat, 32, Pointer(Hidden), FDevice);
    Args[FHead.InputIndex('norm.weight')] := FFiles.Tensor(FNorm, FDevice);
    First := 0;
    while First < FEffectiveVocab do
    begin
      CheckStop(Stop);
      Valid := Min(FHeadRows, FVocab - First);
      if FDecodedPool <> nil then FDecodedPool.BeginGroup;
      try
        Started := GetTickCount64;
        Destination := nil;
        try
          if FDecodedPool <> nil then Destination := FDecodedPool.Rent(0, [FHeadRows, FHidden]);
          Args[MatrixIndex] := FFiles.TensorSlice(FProjection, Int64(First) * FHidden,
            Int64(Valid) * FHidden, [FHeadRows, FHidden], FDevice, Destination);
        finally
          TTVMRuntime.ObjectRelease(Destination);
        end;
        FTimings.OutputParameters := FTimings.OutputParameters +
          (GetTickCount64 - Started) / 1000.0;
        Started := GetTickCount64;
        Reply := FHead.Graph.Run(Args, Stop);
        RequireCpu(Reply.TypeIndex = kTVMFFITensor, 'Expected a single output-head tensor');
        Values := ReadFloatTensor(Reply.VObj);
        RequireCpu(Length(Values) = FHeadRows, 'Qwen output tile width differs');
        Take := Min(FHeadRows, FEffectiveVocab - First);
        for I := 0 to Take - 1 do
        begin
          RequireCpu(not IsNan(Values[I]) and not IsInfinite(Values[I]), 'Non-finite Qwen logits');
          Result[First + I] := Values[I];
        end;
        FTimings.OutputMath := FTimings.OutputMath + (GetTickCount64 - Started) / 1000.0;
      finally
        if FCuda <> nil then FCuda.Synchronize;
        if Reply.VObj <> nil then TTVMRuntime.ObjectRelease(Reply.VObj);
        TTVMRuntime.ObjectRelease(Args[MatrixIndex]);
        if FDecodedPool <> nil then FDecodedPool.EndGroup;
        FFiles.ReleaseResidentRange(FProjection, Int64(First) * FHidden, Int64(Valid) * FHidden);
      end;
      Inc(First, FHeadRows);
    end;
  finally
    if FCuda <> nil then FCuda.Synchronize;
    ReleaseTensors(Args);
  end;
end;

function TQwen35Native.Prefill(const Tokens: array of Integer; const Stop: TQwenStopRequested;
  const Progress: TQwenProgress): TArray<Single>;
var
  Prompt: TQwen35PreparedPrompt;
begin
  RequireCpu((Length(Tokens) > 0) and (Length(Tokens) <= FContext), 'Prompt exceeds compiled Qwen context');
  Prompt := Default(TQwen35PreparedPrompt);
  Prompt.AppendText(Tokens);
  Result := PrefillPrepared(Prompt, Stop, Progress);
end;

function TQwen35Native.PrefillPrepared(const Prompt: TQwen35PreparedPrompt;
  const Stop: TQwenStopRequested; const Progress: TQwenProgress): TArray<Single>;
var
  Rows: TArray<TArray<Single>>;
  Weights: TTensorHandles;
  Batch: TPlan;
  Block, Updated: TArray<Single>;
  I, J, K: Integer;
  Offset: NativeInt;
begin
  FTimings := Default(TQwen35Timings);
  if FCuda <> nil then FCuda.Enter;
  try
    ResetStates;
    ValidateQwen35Prompt(Prompt, FHidden, FEffectiveVocab, FContext);
    if FLegacyPositions then
      for I := 0 to High(Prompt.Rows) do
        RequireCpu(
          (Prompt.Rows[I].Rope[0] = I) and (Prompt.Rows[I].Rope[1] = I) and (Prompt.Rows[I].Rope[2] = I),
          'This Qwen bundle has a text-only position graph; rebuild its graphs for multimodal positions'
        );
    SetLength(Rows, Length(Prompt.Rows));
    for I := 0 to High(Prompt.Rows) do
    begin
      CheckStop(Stop);
      if Prompt.Rows[I].Visual < 0 then
        Rows[I] := FFiles.FloatRange(FEmbedding, Int64(Prompt.Rows[I].Token) * FHidden, FHidden)
      else begin
        Offset := NativeInt(Prompt.Rows[I].Visual) * FHidden;
        SetLength(Rows[I], FHidden);
        Move(Prompt.Embeddings[Offset], Rows[I][0], NativeInt(FHidden) * SizeOf(Single));
      end;
    end;
    try
      for I := 0 to High(FLayers) do
      begin
        CheckStop(Stop);
        if Assigned(Progress) then
          Progress(Format('Prefill layer %d/%d', [I + 1, Length(FLayers)]));
        Weights := Parameters(I, Stop);
        try
          { Full chunks from row 0 run in the batch graph; the remainder runs
            row by row without padding rows. Weights stay resident for both. }
          if FLayers[I].Plan = FLinear then
            Batch := FPrefillLinear
          else
            Batch := FPrefillFull;
          J := 0;
          if Batch <> nil then
            while J <= Length(Rows) - FPrefillChunk do
            begin
              SetLength(Block, FPrefillChunk * FHidden);
              for K := 0 to FPrefillChunk - 1 do
              begin
                RequireCpu(Length(Rows[J + K]) = FHidden, 'Decoder input hidden width differs');
                Move(Rows[J + K][0], Block[K * FHidden], NativeInt(FHidden) * SizeOf(Single));
              end;
              Updated := RunPrefillLayer(I, J, Batch, Prompt, Block, Weights, Stop);
              for K := 0 to FPrefillChunk - 1 do Rows[J + K] := Copy(Updated, K * FHidden, FHidden);
              Inc(J, FPrefillChunk);
            end;
          while J < Length(Rows) do
          begin
            Rows[J] := RunLayer(I, J, Prompt.Rows[J].Rope, Rows[J], Weights, Stop);
            Inc(J);
          end;
        finally
          ReleaseParameters(I, Weights);
        end;
      end;
      Result := Logits(Rows[High(Rows)], Stop);
      FPosition := Length(Prompt.Rows);
      FRopeDelta := Qwen35RopeDelta(Prompt);
    except
      if FCuda <> nil then FCuda.Synchronize;
      ResetStates;
      raise;
    end;
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
end;

function TQwen35Native.Decode(const Token: Integer; const Stop: TQwenStopRequested;
  const Progress: TQwenProgress): TArray<Single>;
var
  Hidden: TArray<Single>;
  Weights: TTensorHandles;
  Rope: TQwen35RopePosition;
  Next: Int64;
  I: Integer;
begin
  RequireCpu((FPosition >= 0) and (FPosition < FContext),
    'Qwen requires prefill or a longer compiled context');
  Next := Int64(FPosition) + FRopeDelta;
  RequireCpu((Next >= 0) and (Next < Qwen35MaxRopePosition), 'Qwen rotary position is out of range');
  Rope[0] := Next;
  Rope[1] := Next;
  Rope[2] := Next;
  RequireCpu((Token >= 0) and (Token < FEffectiveVocab), 'Generated token is outside tokenizer vocabulary');
  if FCuda <> nil then FCuda.Enter;
  try
    try
      CheckStop(Stop);
      Hidden := FFiles.FloatRange(FEmbedding, Int64(Token) * FHidden, FHidden);
      for I := 0 to High(FLayers) do
      begin
        if Assigned(Progress) then Progress(Format('Decode layer %d/%d', [I + 1, Length(FLayers)]));
        Weights := Parameters(I, Stop);
        try
          Hidden := RunLayer(I, FPosition, Rope, Hidden, Weights, Stop);
        finally
          ReleaseParameters(I, Weights);
        end;
      end;
      Result := Logits(Hidden, Stop);
      Inc(FPosition);
    except
      if FCuda <> nil then FCuda.Synchronize;
      ResetStates;
      raise;
    end;
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
end;

end.
