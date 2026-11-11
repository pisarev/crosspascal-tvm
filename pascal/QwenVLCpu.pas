{ ************************************************************************** }
{                                                                            }
{ QwenVLCpu                                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit QwenVLCpu;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes, Generics.Collections
  {$ELSE}System.SysUtils, System.Classes, System.Generics.Collections{$ENDIF},
  CrossPascal.Json, DLPack, QwenTokenizer, TVM, TVM.CpuGraph, TVM.CudaExecution, QwenCuda;

type
  TQwenVLCpu = class
  private type
    TParameter = record
      FileData: TMappedModelFile;
      Shape: TArray<Int64>;
      Bits: Integer;
    end;
    TUploadedWeights = TDictionary<TMappedModelFile, TTVMFFIObjectHandle>;
    TGraph = record
      VM: TCpuGraph;
      Weights: TTensorHandles;
      LibraryName: string;
      Parameters: TArray<TParameter>;
      Workspace: TCudaWorkspace;
    end;
    TPosition = array[0..2] of Int64;
    TPrompt = record
      Tokens: TArray<Integer>;
      Rotary: TArray<TPosition>;
      Features: TArray<TArray<Single>>;
      NextPosition: Int64;
      procedure AppendText(const IDs: array of Integer);
      procedure AppendImage(const ID, Count, Grid, Width: Integer; const Encoded: TArray<Single>);
    end;
  private
    FFolder: string;
    FCVLibrary: string;
    FLastHash: string;
    FMaps: TStringList;
    FTokenizer: TQwenTokenizer;
    FDecode: TGraph;
    FPrefill: TGraph;
    FVision: TGraph;
    FEmbedding: TMappedModelFile;
    FLastFeatures: TArray<Single>;
    FCacheShape: TArray<Int64>;
    FContext: Integer;
    FChunk: Integer;
    FCacheCount: Integer;
    FSize: Integer;
    FPatch: Integer;
    FMerge: Integer;
    FTemporal: Integer;
    FSoftTokens: Integer;
    FWidth: Integer;
    FVocab: Integer;
    FImageStart: Integer;
    FImageEnd: Integer;
    FImageToken: Integer;
    FStops: TArray<Integer>;
    FMean: array[0..2] of Single;
    FStd: array[0..2] of Single;
    FScale: Single;
    FLoadSeconds: Double;
    FCuda: TCudaExecution;
    FDevice: TDLDeviceType;
    FDeviceLabel: string;
    FBackend: string;
    function WeightFile(const Spec: TJSONObject; out Shape: TArray<Int64>;
      out Bits: Integer): TMappedModelFile;
    procedure LoadGraph(const Value: TJSONValue; const Hashes: TJSONObject;
      const InputCount: Integer; var Graph: TGraph);
    procedure LoadCudaWeights(var Graph: TGraph; const Uploaded: TUploadedWeights);
    procedure LoadCudaTextWeights;
    procedure ReleaseCudaWeights(var Graph: TGraph);
    procedure ReleaseGraphs;
    function ImageFeatures(const FileName: string; const Stop: TQwenStopRequested; out Hash: string;
      out Reused: Boolean): TArray<Single>;
    function TextStep(const Graph: TGraph; const Prompt: TPrompt; const Position, Count: Integer;
      var Caches: TTensorHandles): TArray<Single>;
    function GenerateInContext(const Prompt: string; const Options: TQwenGenerationOptions;
      const Images: array of string; const Progress, OnText: TQwenProgress;
      const ShouldStop: TQwenStopRequested): string;
  public
    constructor Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
    destructor Destroy; override;
    function Generate(const Prompt: string; const Options: TQwenGenerationOptions;
      const Images: array of string; const Progress, OnText: TQwenProgress;
      const ShouldStop: TQwenStopRequested): string;
    property ContextTokens: Integer read FContext;
  end;

implementation

uses
  {$IFDEF FPC}Windows, Math{$ELSE}Winapi.Windows, System.Math{$ENDIF}, CrossVision.Mat,
  TVM.ImageInput, CrossPascal.SHA256;

const
  MiB = 1024 * 1024;

procedure StopIfRequested(const Stop: TQwenStopRequested);
begin
  if Assigned(Stop) and Stop() then raise ECpuGraphStopped.Create('Generation stopped');
end;

procedure TQwenVLCpu.TPrompt.AppendText(const IDs: array of Integer);
var
  Start, I, J: Integer;
begin
  Start := Length(Tokens);
  SetLength(Tokens, Start + Length(IDs));
  SetLength(Rotary, Length(Tokens));
  for I := 0 to High(IDs) do
  begin
    Tokens[Start + I] := IDs[I];
    for J := 0 to 2 do Rotary[Start + I][J] := NextPosition;
    Inc(NextPosition);
  end;
end;

procedure TQwenVLCpu.TPrompt.AppendImage(const ID, Count, Grid, Width: Integer;
  const Encoded: TArray<Single>);
var
  Start, I: Integer;
begin
  Start := Length(Tokens);
  SetLength(Tokens, Start + Count);
  SetLength(Rotary, Length(Tokens));
  SetLength(Features, Length(Tokens));
  for I := 0 to Count - 1 do
  begin
    Tokens[Start + I] := ID;
    Rotary[Start + I][0] := NextPosition;
    Rotary[Start + I][1] := NextPosition + I div Grid;
    Rotary[Start + I][2] := NextPosition + I mod Grid;
    Features[Start + I] := Copy(Encoded, I * Width, Width);
  end;
  Inc(NextPosition, Grid);
end;

function TQwenVLCpu.WeightFile(const Spec: TJSONObject; out Shape: TArray<Int64>;
  out Bits: Integer): TMappedModelFile;
var
  Name, Path, DataType: string;
  I: Integer;
  Stream: TFileStream;
begin
  Name := ModelText(Spec.GetValue('file'));
  Shape := ModelShape(Spec.GetValue('shape'));
  DataType := ModelText(Spec.GetValue('dtype'));
  RequireCpu((DataType = 'float32') or ((DataType = 'int8') and (FCuda <> nil)),
    'Unsupported Qwen-VL weight data type: ' + DataType);
  Bits := 32;
  if DataType = 'int8' then Bits := 8;
  I := FMaps.IndexOf(Name);
  if I >= 0 then
    Result := TMappedModelFile(FMaps.Objects[I])
  else begin
    Path := BundleMember(FFolder, Name);
    Stream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
    try
      RequireCpu(SameText(SHA256Stream(Stream), ModelText(Spec.GetValue('sha256'))),
        'Qwen-VL weight file changed: ' + Name);
      Result := TMappedModelFile.Create(Path);
    finally
      Stream.Free;
    end;
    try
      FMaps.AddObject(Name, Result);
    except
      Result.Free;
      raise;
    end;
  end;
  RequireCpu(Result.Size = TensorElements(Shape) * (Bits div 8),
    'Incorrect Qwen-VL weight size: ' + Name);
end;

procedure TQwenVLCpu.LoadGraph(const Value: TJSONValue; const Hashes: TJSONObject;
  const InputCount: Integer; var Graph: TGraph);
var
  Spec, Params: TJSONObject;
  Inputs, Runtime: TJSONArray;
  Name, LibraryName: string;
  Parameter: TParameter;
  I: Integer;
begin
  Spec := ModelObject(Value, 'Qwen-VL graph');
  Inputs := ModelArray(Spec.GetValue('input_names'), 'graph input names');
  Runtime := ModelArray(Spec.GetValue('runtime_inputs'), 'graph runtime inputs');
  RequireCpu((Runtime.Count = InputCount) and (Inputs.Count > InputCount),
    'Qwen-VL graph has the wrong number of inputs');
  for I := 0 to InputCount - 1 do
    RequireCpu(ModelText(Runtime.Items[I]) = ModelText(Inputs.Items[I]), 'Invalid Qwen-VL input order');
  LibraryName := ModelText(Spec.GetValue('library'));
  RequireCpu((Hashes.GetValue(LibraryName) <> nil) and (Hashes.GetValue(ModelText(Spec.GetValue('onnx'))) <> nil),
    'Qwen-VL graph hashes are incomplete');
  Params := ModelObject(Spec.GetValue('parameters'), 'graph parameters');
  if FCuda = nil then SetLength(Graph.Weights, Inputs.Count - InputCount)
  else SetLength(Graph.Parameters, Inputs.Count - InputCount);
  for I := InputCount to Inputs.Count - 1 do
  begin
    Name := ModelText(Inputs.Items[I]);
    Parameter.FileData := WeightFile(ModelObject(Params.GetValue(Name), Name), Parameter.Shape,
      Parameter.Bits);
    if FCuda = nil then
      Graph.Weights[I - InputCount] := MappedFloatTensor(
        Parameter.FileData.AtOffset(0, Parameter.FileData.Size), Parameter.Shape)
    else
      Graph.Parameters[I - InputCount] := Parameter;
  end;
  Graph.LibraryName := BundleMember(FFolder, LibraryName);
  Graph.VM := TCpuGraph.Create(Graph.LibraryName, FDevice);
end;

procedure TQwenVLCpu.LoadCudaWeights(var Graph: TGraph; const Uploaded: TUploadedWeights);
var
  Parameter: TParameter;
  Handle: TTVMFFIObjectHandle;
  Required, Free: UInt64;
  Code: TDLDataTypeCode;
  I: Integer;
begin
  Required := 0;
  for Parameter in Graph.Parameters do
    if not Uploaded.ContainsKey(Parameter.FileData) then Inc(Required, Parameter.FileData.Size);
  Free := FCuda.FreeBytes;
  RequireCpu(Free >= Required + 384 * MiB,
    Format('Qwen-VL needs %.2f GiB of free CUDA memory for this stage. Available: %.2f GiB.',
      [(Required + 384 * MiB) / 1073741824.0, Free / 1073741824.0]));
  SetLength(Graph.Weights, Length(Graph.Parameters));
  try
    for I := 0 to High(Graph.Parameters) do
    begin
      Parameter := Graph.Parameters[I];
      if Uploaded.TryGetValue(Parameter.FileData, Handle) then
      begin
        TTVMRuntime.ObjectRetain(Handle);
        Graph.Weights[I] := Handle;
        Continue;
      end;
      Code := kDLFloat;
      if Parameter.Bits = 8 then Code := kDLInt;
      Handle := DeviceTensor(Parameter.Shape, Code, Parameter.Bits,
        Parameter.FileData.AtOffset(0, Parameter.FileData.Size), kDLCUDA);
      Graph.Weights[I] := Handle;
      Uploaded.Add(Parameter.FileData, Handle);
    end;
    FCuda.Synchronize;
    Graph.Workspace := TCudaWorkspace.Create(FCuda, Graph.LibraryName, 256 * MiB);
  except
    ReleaseCudaWeights(Graph);
    raise;
  end;
end;

procedure TQwenVLCpu.LoadCudaTextWeights;
var
  Uploaded: TUploadedWeights;
begin
  if Length(FDecode.Weights) > 0 then Exit;
  Uploaded := TUploadedWeights.Create;
  try
    try
      LoadCudaWeights(FDecode, Uploaded);
      if FPrefill.VM <> nil then LoadCudaWeights(FPrefill, Uploaded);
    except
      ReleaseCudaWeights(FPrefill);
      ReleaseCudaWeights(FDecode);
      raise;
    end;
  finally
    Uploaded.Free;
  end;
end;

procedure TQwenVLCpu.ReleaseCudaWeights(var Graph: TGraph);
begin
  FreeAndNil(Graph.Workspace);
  ReleaseTensors(Graph.Weights);
end;

constructor TQwenVLCpu.Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
var
  Data: TJSONValue;
  Model, Hashes, Graphs, Config, VisionConfig, TextConfig: TJSONObject;
  Shape: TArray<Int64>;
  Stops: TJSONValue;
  Mean, Std: TJSONArray;
  I, Bits: Integer;
  Started: UInt64;
begin
  inherited Create;
  Started := GetTickCount64;
  FFolder := ExpandFileName(Folder);
  FDevice := kDLCPU;
  FDeviceLabel := 'CPU';
  FCVLibrary := IncludeTrailingPathDelimiter(RuntimeRoot) + 'runtime\gemma-native\crosscv.dll';
  FMaps := TStringList.Create;
  FMaps.OwnsObjects := True;
  FMaps.Sorted := True;
  Data := ReadLanguageManifest(FFolder);
  try
    Model := ModelObject(Data, 'Qwen-VL model');
    FBackend := ModelText(Model.GetValue('backend'));
    RequireCpu((ModelText(Model.GetValue('schema')) = 'crosspascal.qwen2_5_vl.v1') and
      ((FBackend = 'TVM LLVM CPU FP32') or (FBackend = 'TVM CUDA W8A32')),
      'Expected a Qwen2.5-VL TVM bundle');
    if FBackend = 'TVM CUDA W8A32' then
    begin
      FCuda := TCudaExecution.Create;
      FCuda.RequireArchitecture(ModelText(Model.GetValue('cuda_arch')));
      FDevice := kDLCUDA;
      FDeviceLabel := 'CUDA';
    end;
    FContext := ModelInt(Model.GetValue('context'), 'context');
    FCacheCount := ModelInt(Model.GetValue('cache_count'), 'cache count');
    FCacheShape := ModelShape(Model.GetValue('cache_shape'));
    FSize := ModelInt(Model.GetValue('image_size'), 'image size');
    FSoftTokens := ModelInt(Model.GetValue('soft_tokens'), 'image token count');
    RequireCpu((FContext > 1) and (FContext <= 131072) and (FCacheCount > 0) and
      (FCacheCount <= 256) and (FSize > 0) and (FSize <= 4096) and
      (FSoftTokens > 0) and (FSoftTokens < FContext), 'Invalid Qwen-VL profile');
    Hashes := ModelObject(Model.GetValue('artifact_sha256'), 'Qwen-VL artifact hashes');
    if Assigned(Progress) then Progress('Verifying the Qwen-VL ' + FDeviceLabel + ' bundle...');
    VerifyArtifacts(FFolder, Hashes);
    QwenCuda.InitializeRuntime(IncludeTrailingPathDelimiter(RuntimeRoot) +
      'crosspascal-tvm\runtime\windows-x64-cuda\', FCuda <> nil);
    Graphs := ModelObject(Model.GetValue('graphs'), 'Qwen-VL graphs');
    if FCuda <> nil then FCuda.Enter;
    try
      LoadGraph(Graphs.GetValue('text'), Hashes, FCacheCount + 3, FDecode);
      LoadGraph(Graphs.GetValue('vision'), Hashes, 1, FVision);
      if Graphs.GetValue('prefill') <> nil then
      begin
        FChunk := ModelInt(Model.GetValue('prefill_chunk'), 'prefill chunk');
        RequireCpu((FChunk > 1) and (FChunk <= FContext), 'Invalid Qwen-VL prefill chunk');
        LoadGraph(Graphs.GetValue('prefill'), Hashes, FCacheCount + 3, FPrefill);
      end;
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    Data.Free;
  end;
  Data := ReadModelJSON(BundleMember(FFolder, 'parameters.json'));
  try
    FEmbedding := WeightFile(
      ModelObject(
        ModelObject(ModelObject(Data, 'parameters').GetValue('weights'), 'weights').GetValue('model.language_model.embed_tokens.weight'),
        'embedding table'
      ),
      Shape, Bits
    );
    RequireCpu((Bits = 32) and (Length(Shape) = 2) and (Shape[0] <= 1000000) and (Shape[1] <= 65536),
      'Invalid Qwen-VL embedding table');
    FVocab := Shape[0];
    FWidth := Shape[1];
  finally
    Data.Free;
  end;
  FTokenizer := TQwenTokenizer.Create;
  Data := ReadModelJSON(BundleMember(FFolder, 'tokenizer.json'));
  try
    FTokenizer.LoadByteLevelJSON(ModelObject(Data, 'Qwen-VL tokenizer'));
  finally
    Data.Free;
  end;
  Data := ReadModelJSON(BundleMember(FFolder, 'config.json'));
  try
    Config := ModelObject(Data, 'Qwen-VL config');
    VisionConfig := ModelObject(Config.GetValue('vision_config'), 'vision config');
    FPatch := ModelInt(VisionConfig.GetValue('patch_size'), 'patch size');
    FMerge := ModelInt(VisionConfig.GetValue('spatial_merge_size'), 'spatial merge');
    FTemporal := ModelInt(VisionConfig.GetValue('temporal_patch_size'), 'temporal patch');
    RequireCpu((FPatch > 0) and (FPatch <= 128) and (FMerge > 0) and (FMerge <= 16) and
      (FTemporal > 0) and (FTemporal <= 8) and (FSize mod (FPatch * FMerge) = 0) and
      (Sqr(FSize div (FPatch * FMerge)) = FSoftTokens), 'Invalid Qwen-VL patch layout');
    FImageStart := ModelInt(Config.GetValue('vision_start_token_id'), 'vision start token');
    FImageEnd := ModelInt(Config.GetValue('vision_end_token_id'), 'vision end token');
    FImageToken := ModelInt(Config.GetValue('image_token_id'), 'image token');
    TextConfig := Config;
    if Config.GetValue('text_config') <> nil then
      TextConfig := ModelObject(Config.GetValue('text_config'), 'text config');
    Stops := TextConfig.GetValue('eos_token_id');
    if Stops = nil then Stops := Config.GetValue('eos_token_id');
    if Stops is TJSONArray then
    begin
      SetLength(FStops, TJSONArray(Stops).Count);
      for I := 0 to High(FStops) do FStops[I] := ModelInt(TJSONArray(Stops).Items[I], 'EOS token');
    end
    else
      FStops := [ModelInt(Stops, 'EOS token')];
    FStops := FStops + [FTokenizer.SpecialId('<|im_end|>')];
  finally
    Data.Free;
  end;
  Data := ReadModelJSON(BundleMember(FFolder, 'preprocessor_config.json'));
  try
    Config := ModelObject(Data, 'image preprocessor');
    FScale := 1 / 255;
    if Config.GetValue('rescale_factor') <> nil then
      FScale := ModelFloat(Config.GetValue('rescale_factor'), 'image rescale factor');
    Mean := ModelArray(Config.GetValue('image_mean'), 'image mean');
    Std := ModelArray(Config.GetValue('image_std'), 'image standard deviation');
    RequireCpu((Mean.Count = 3) and (Std.Count = 3), 'Expected RGB normalization');
    for I := 0 to 2 do
    begin
      FMean[I] := ModelFloat(Mean.Items[I], 'image mean');
      FStd[I] := ModelFloat(Std.Items[I], 'image standard deviation');
      RequireCpu(FStd[I] > 0, 'Image standard deviation must be positive');
    end;
  finally
    Data.Free;
  end;
  FLoadSeconds := (GetTickCount64 - Started) / 1000.0;
  if Assigned(Progress) then Progress('Model loaded on ' + FDeviceLabel + ' through TVM');
end;

destructor TQwenVLCpu.Destroy;
begin
  try
    if FCuda <> nil then FCuda.Enter;
    try
      ReleaseGraphs;
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    FCuda.Free;
    FTokenizer.Free;
    FMaps.Free;
    inherited;
  end;
end;

procedure TQwenVLCpu.ReleaseGraphs;
begin
  ReleaseCudaWeights(FVision);
  ReleaseCudaWeights(FPrefill);
  ReleaseCudaWeights(FDecode);
  FVision.VM.Free;
  FPrefill.VM.Free;
  FDecode.VM.Free;
end;

function TQwenVLCpu.ImageFeatures(const FileName: string; const Stop: TQwenStopRequested;
  out Hash: string; out Reused: Boolean): TArray<Single>;
var
  Bytes, Pixels: TBytes;
  Image: IMat;
  Patches: TArray<Single>;
  Tensor: TTVMFFIObjectHandle;
  Reply: TTVMFFIAny;
  Shape: PDLTensor;
  Uploaded: TUploadedWeights;
  W, H, PadX, PadY, Grid, PatchWidth, At, I, BX, BY, MX, MY, C, T, X, Y, RX, RY: Integer;
  Raw: Single;
begin
  Bytes := ReadImageBytes(FileName);
  Hash := SHA256Bytes(Bytes);
  Reused := Hash = FLastHash;
  if Reused then Exit(FLastFeatures);
  StopIfRequested(Stop);
  Image := DecodeColorImage(Bytes, FCVLibrary);
  W := FSize;
  H := FSize;
  if Image.Width > Image.Height then
    H := Max(1, Round(Image.Height * FSize / Image.Width))
  else
    W := Max(1, Round(Image.Width * FSize / Image.Height));
  PadX := Round((FSize - W) / 2);
  PadY := Round((FSize - H) / 2);
  Pixels := ResizeColorImage(Image, W, H, True);
  Grid := FSize div FPatch;
  PatchWidth := 3 * FTemporal * Sqr(FPatch);
  SetLength(Patches, Sqr(Grid) * PatchWidth);
  At := 0;
  for BY := 0 to Grid div FMerge - 1 do
    for BX := 0 to Grid div FMerge - 1 do
      for MY := 0 to FMerge - 1 do
        for MX := 0 to FMerge - 1 do
          for C := 0 to 2 do
            for T := 0 to FTemporal - 1 do
              for Y := 0 to FPatch - 1 do
                for X := 0 to FPatch - 1 do
                begin
                  RX := (BX * FMerge + MX) * FPatch + X - PadX;
                  RY := (BY * FMerge + MY) * FPatch + Y - PadY;
                  Raw := 0;
                  if (RX >= 0) and (RX < W) and (RY >= 0) and (RY < H) then
                    Raw := Pixels[(RY * W + RX) * 3 + C] * FScale;
                  Patches[At] := (Raw - FMean[C]) / FStd[C];
                  Inc(At);
                end;
  StopIfRequested(Stop);
  if FCuda <> nil then
  begin
    ReleaseCudaWeights(FPrefill);
    ReleaseCudaWeights(FDecode);
    Uploaded := TUploadedWeights.Create;
    try
      LoadCudaWeights(FVision, Uploaded);
    finally
      Uploaded.Free;
    end;
  end;
  Tensor := nil;
  try
    Tensor := DeviceTensor([Sqr(Grid), PatchWidth], kDLFloat, 32, Pointer(Patches), FDevice);
    if FVision.Workspace = nil then
      Reply := FVision.VM.Run(TTensorHandles.Create(Tensor) + FVision.Weights, TCpuGraphStop(Stop))
    else
      Reply := FVision.Workspace.Run(FVision.VM, TTensorHandles.Create(Tensor) + FVision.Weights,
        TCpuGraphStop(Stop));
    RequireCpu(Reply.TypeIndex = kTVMFFITensor, 'Expected Qwen-VL vision tensor');
    try
      Shape := TTVMRuntime.TensorDLTensor(Reply.VObj);
      RequireCpu((Shape <> nil) and (Shape^.ndim = 2) and (Shape^.shape[0] = FSoftTokens) and
        (Shape^.shape[1] = FWidth), 'Invalid Qwen-VL image features');
      Result := ReadFloatTensor(Reply.VObj);
    finally
      TTVMRuntime.ObjectRelease(Reply.VObj);
    end;
  finally
    TTVMRuntime.ObjectRelease(Tensor);
    if FCuda <> nil then ReleaseCudaWeights(FVision);
  end;
  StopIfRequested(Stop);
  for I := 0 to High(Result) do
    RequireCpu(not IsNan(Result[I]) and not IsInfinite(Result[I]), 'Non-finite Qwen-VL image features');
  FLastHash := Hash;
  FLastFeatures := Result;
end;

function TQwenVLCpu.TextStep(const Graph: TGraph; const Prompt: TPrompt;
  const Position, Count: Integer; var Caches: TTensorHandles): TArray<Single>;
var
  Inputs, NextCaches: TTensorHandles;
  Rows: TArray<Single>;
  Rotary, Positions: TArray<Int64>;
  Data: Pointer;
  I, J, Token, Index: Integer;
  Reply: TTVMFFIAny;
  Logits: TTVMFFIObjectHandle;
begin
  SetLength(Rows, Count * FWidth);
  SetLength(Rotary, Count * 3);
  SetLength(Positions, Count);
  for I := 0 to Count - 1 do
  begin
    Index := Position + I;
    Token := Prompt.Tokens[Index];
    if (Index < Length(Prompt.Features)) and (Prompt.Features[Index] <> nil) then
      Data := Pointer(Prompt.Features[Index])
    else begin
      RequireCpu((Token >= 0) and (Token < FVocab), 'Token outside the Qwen-VL embedding table');
      Data := FEmbedding.AtOffset(Int64(Token) * FWidth * SizeOf(Single), FWidth * SizeOf(Single));
    end;
    Move(Data^, Rows[I * FWidth], FWidth * SizeOf(Single));
    for J := 0 to 2 do Rotary[J * Count + I] := Prompt.Rotary[Index][J];
    Positions[I] := Index;
  end;
  SetLength(Inputs, 3);
  try
    Inputs[0] := DeviceTensor([1, Count, FWidth], kDLFloat, 32, Pointer(Rows), FDevice);
    Inputs[1] := DeviceTensor([3, 1, Count], kDLInt, 64, Pointer(Rotary), FDevice);
    Inputs[2] := DeviceTensor([Count], kDLInt, 64, Pointer(Positions), FDevice);
    if Graph.Workspace = nil then
      Reply := Graph.VM.Run(Inputs + Caches + Graph.Weights)
    else
      Reply := Graph.Workspace.Run(Graph.VM, Inputs + Caches + Graph.Weights);
    RequireCpu(Reply.VObj <> nil, 'TVM returned no Qwen-VL graph outputs');
    try
      SetLength(NextCaches, Length(Caches));
      for I := 0 to High(Caches) do NextCaches[I] := TensorItem(Reply.VObj, I + 1);
      if Position + Count = Length(Prompt.Tokens) then
      begin
        Logits := TensorItem(Reply.VObj, 0);
        try
          Result := ReadFloatTensor(Logits);
        finally
          TTVMRuntime.ObjectRelease(Logits);
        end;
      end
      else
        Result := nil;
      ReleaseTensors(Caches);
      Caches := NextCaches;
      NextCaches := nil;
    finally
      TTVMRuntime.ObjectRelease(Reply.VObj);
    end;
  finally
    ReleaseTensors(NextCaches);
    ReleaseTensors(Inputs);
  end;
end;

function TQwenVLCpu.Generate(const Prompt: string; const Options: TQwenGenerationOptions;
  const Images: array of string; const Progress, OnText: TQwenProgress;
  const ShouldStop: TQwenStopRequested): string;
begin
  if FCuda <> nil then FCuda.Enter;
  try
    Result := GenerateInContext(Prompt, Options, Images, Progress, OnText, ShouldStop);
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
end;

function TQwenVLCpu.GenerateInContext(const Prompt: string; const Options: TQwenGenerationOptions;
  const Images: array of string; const Progress, OnText: TQwenProgress;
  const ShouldStop: TQwenStopRequested): string;
const
  EffortBudget: array[TQwenAnswerEffort] of Integer = (0, 128, 512, 1024);
  EffortName: array[TQwenAnswerEffort] of string = ('full_context', 'brief', 'balanced', 'detailed');
var
  Input: TPrompt;
  Output, TextIDs: TArray<Integer>;
  Caches: TTensorHandles;
  Features, Logits: TArray<Single>;
  Seen: TArray<Boolean>;
  Text, Answer, Path, Hash, StopReason, LimitReason: string;
  Count, Position, PromptCount, Limit, Token, I, EOS: Integer;
  Started, Encoded, First, ImageStarted: UInt64;
  Sampled, Reused, EndOfSequence: Boolean;
  State: Cardinal;
  Report, ImageReport, Settings: TJSONObject;
  ImageReports, TokenIDs, Modalities: TJSONArray;
begin
  ValidateQwenGenerationOptions(Options);
  RequireCpu(Trim(Prompt) <> '', 'Enter a non-empty prompt');
  Started := GetTickCount64;
  Encoded := Started;
  First := 0;
  PromptCount := 0;
  StopReason := 'context_limit';
  LimitReason := StopReason;
  Sampled := (Options.Sampling = qsSample) or ((Options.Sampling = qsDefault) and
    (Options.MaxNewTokens = 0));
  if Options.Seed >= 0 then
    State := Cardinal(Options.Seed)
  else
    State := Cardinal(GetTickCount64 xor UInt64(GetCurrentThreadId));
  Report := TJSONObject.Create;
  try
    ImageReports := TJSONArray.Create;
    Report.AddPair('images', ImageReports);
    Settings := TJSONObject.Create;
    Report.AddPair('settings', Settings);
    if Sampled then
      Settings.AddPair('sampling', 'sample')
    else
      Settings.AddPair('sampling', 'greedy');
    Settings.AddPair('seed', JSONUnsigned(State));
    Settings.AddPair('temperature', JSONFloat(Options.Temperature));
    Settings.AddPair('top_p', JSONFloat(Options.TopP));
    Settings.AddPair('top_k', JSONInteger(Options.TopK));
    Settings.AddPair('repetition_penalty', JSONFloat(Options.RepetitionPenalty));
    Settings.AddPair('effort', EffortName[Options.Effort]);
    try
      StopIfRequested(ShouldStop);
      Text := Prompt;
      if Options.SystemPrompt <> '' then Text := Options.SystemPrompt + #10#10 + Prompt;
      Input := Default(TPrompt);
      if not Options.PlainCompletion then
        Input.AppendText(FTokenizer.Encode('<|im_start|>system' + #10 + 'You are a helpful assistant.' +
          '<|im_end|>' + #10 + '<|im_start|>user' + #10));
      TextIDs := FTokenizer.Encode(Text);
      RequireCpu(Length(Input.Tokens) + Length(TextIDs) + Int64(Length(Images)) * (FSoftTokens + 2) + 12 < FContext,
        'Text and images exceed the compiled context');
      for Path in Images do
      begin
        if Assigned(Progress) then Progress('Encoding image on ' + FDeviceLabel + '...');
        ImageStarted := GetTickCount64;
        Features := ImageFeatures(Path, ShouldStop, Hash, Reused);
        ImageReport := TJSONObject.Create;
        ImageReports.AddElement(ImageReport);
        ImageReport.AddPair('sha256', Hash);
        ImageReport.AddPair('reused', JSONBoolean(Reused));
        ImageReport.AddPair('seconds', JSONFloat((GetTickCount64 - ImageStarted) / 1000.0));
        if FCuda = nil then ImageReport.AddPair('backend', FBackend)
        else ImageReport.AddPair('backend', 'TVM CUDA FP32');
        Input.AppendText([FImageStart]);
        Input.AppendImage(FImageToken, FSoftTokens, FSize div (FPatch * FMerge), FWidth, Features);
        Input.AppendText([FImageEnd]);
      end;
      Input.AppendText(TextIDs);
      if not Options.PlainCompletion then
        Input.AppendText(FTokenizer.Encode('<|im_end|>' + #10 + '<|im_start|>assistant' + #10));
      Encoded := GetTickCount64;
      PromptCount := Length(Input.Tokens);
      RequireCpu((PromptCount > 0) and (PromptCount < FContext), 'Prompt does not fit the compiled context');
      Limit := FContext - PromptCount;
      if (Options.MaxNewTokens > 0) and (Options.MaxNewTokens < Limit) then
      begin
        Limit := Options.MaxNewTokens;
        LimitReason := 'manual_limit';
      end
      else if (Options.MaxNewTokens = 0) and (EffortBudget[Options.Effort] > 0) and
        (EffortBudget[Options.Effort] < Limit) then
          begin
            Limit := EffortBudget[Options.Effort];
            LimitReason := 'effort_limit';
          end;
      if Assigned(Progress) then Progress('Processing the prompt on ' + FDeviceLabel + '...');
      if FCuda <> nil then LoadCudaTextWeights;
      SetLength(Caches, FCacheCount);
      for I := 0 to High(Caches) do
        Caches[I] := DeviceTensor(FCacheShape, kDLFloat, 32, nil, FDevice);
      Position := 0;
      while Position < FContext - 1 do
      begin
        StopIfRequested(ShouldStop);
        Count := 1;
        if (FPrefill.VM <> nil) and (PromptCount - Position >= FChunk) then Count := FChunk;
        if Count > 1 then
          Logits := TextStep(FPrefill, Input, Position, Count, Caches)
        else
          Logits := TextStep(FDecode, Input, Position, Count, Caches);
        StopIfRequested(ShouldStop);
        Inc(Position, Count);
        if Position < PromptCount then Continue;
        RequireCpu(Length(Logits) > 0, 'TVM returned empty Qwen-VL logits');
        Token := 0;
        for I := 0 to High(Logits) do
        begin
          RequireCpu(not IsNan(Logits[I]) and not IsInfinite(Logits[I]), 'Non-finite Qwen-VL logits');
          if Logits[I] > Logits[Token] then Token := I;
        end;
        if First = 0 then
        begin
          First := GetTickCount64;
          SetLength(Seen, Length(Logits));
          for I in Input.Tokens do if (I >= 0) and (I < Length(Seen)) then Seen[I] := True;
        end;
        if Sampled then Token := SampleToken(Logits, Seen, Options, State);
        EndOfSequence := False;
        for EOS in FStops do if EOS = Token then EndOfSequence := True;
        if EndOfSequence then
        begin
          StopReason := 'eos';
          Break;
        end;
        Seen[Token] := True;
        Output := Output + [Token];
        Input.AppendText([Token]);
        Answer := FTokenizer.DecodePrefix(Output);
        if Assigned(OnText) then OnText(Answer);
        if Length(Output) >= Limit then
        begin
          StopReason := LimitReason;
          Break;
        end;
      end;
    except
      on ECpuGraphStopped do StopReason := 'user_stop';
    end;
    Report.AddPair('text', Answer);
    Report.AddPair('stop', StopReason);
    Report.AddPair('backend', FBackend);
    if FCuda <> nil then
    begin
      Report.AddPair('device_name', FCuda.DeviceName);
      Report.AddPair('compiled_arch', FCuda.Architecture);
    end;
    Report.AddPair('load_seconds', JSONFloat(FLoadSeconds));
    Report.AddPair('encoding_seconds', JSONFloat((Encoded - Started) / 1000.0));
    Report.AddPair('seconds', JSONFloat((GetTickCount64 - Started) / 1000.0));
    if First = 0 then
      Report.AddPair('first_token_seconds', TJSONNull.Create)
    else
      Report.AddPair('first_token_seconds', JSONFloat((First - Started) / 1000.0));
    Report.AddPair('prompt_tokens', JSONInteger(PromptCount));
    Report.AddPair('generated_tokens', JSONInteger(Length(Output)));
    Report.AddPair('model_reused', JSONBoolean(True));
    TokenIDs := TJSONArray.Create;
    Report.AddPair('tokens', TokenIDs);
    for Token in Output do TokenIDs.AddElement(JSONInteger(Token));
    Modalities := TJSONArray.Create;
    Report.AddPair('modalities', Modalities);
    Modalities.AddElement(TJSONString.Create('text'));
    if Length(Images) > 0 then Modalities.AddElement(TJSONString.Create('image'));
    Result := Report.ToJSON;
  finally
    ReleaseTensors(Caches);
    Report.Free;
  end;
end;

end.
