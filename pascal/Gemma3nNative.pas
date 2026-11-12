{ ************************************************************************** }
{                                                                            }
{ Gemma3nNative                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit Gemma3nNative;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  TVM, DLPack, TVM.CpuGraph, TVM.CudaExecution, CrossPascal.Json,
  CrossPascal.SentencePiece, QwenCuda, Gemma3nVision;

type
  TGemma3nNative = class
  private type
    TEmbeddingTable = record
      Data: TMappedModelFile;
      Rows: Integer;
      Width: Integer;
    end;
    TCudaWeight = record
      Data: Pointer;
      Shape: TArray<Int64>;
      Code: TDLDataTypeCode;
      Bits: Integer;
    end;
    TPrompt = record
      Tokens: TArray<Integer>;
      Embeddings: TArray<TArray<Single>>;
      procedure Add(const Token: Integer); overload;
      procedure Add(const IDs: array of Integer); overload;
    end;
  private
    FFolder: string;
    FSourceHash: string;
    FBackend: string;
    FCudaArch: string;
    FFiles: TStringList;
    FTokenizer: TSentencePiece;
    FDecode: TCpuGraph;
    FPrefill: TCpuGraph;
    FDecodeWeights: TTensorHandles;
    FPrefillWeights: TTensorHandles;
    FVision: TGemma3nVision;
    FTables: array[0..1] of TEmbeddingTable;
    FCacheShape: TArray<Int64>;
    FCacheCount: Integer;
    FChunk: Integer;
    FContext: Integer;
    FHostEmbeddings: Boolean;
    FDeviceType: TDLDeviceType;
    FCuda: TCudaExecution;
    FWorkspace: TCudaWorkspace;
    FCudaWeights: TArray<TCudaWeight>;
    FBos: Integer;
    FEos: Integer;
    FStartTurn: Integer;
    FEndTurn: Integer;
    FStartImage: Integer;
    FEndImage: Integer;
    FSoftImage: Integer;
    FLoadSeconds: Double;
    function WeightFile(const Name: string): TMappedModelFile;
    procedure ReadWeights(const Graph, Weights: TJSONObject; var Args: TTensorHandles);
    procedure LoadCuda(const Manifest, Weights: TJSONObject);
    procedure ReleaseCudaWeights;
    procedure UploadCudaWeights(const Progress: TQwenProgress);
    function Evaluate(const Graph: TCpuGraph; const Weights: TTensorHandles; const Prompt: TPrompt;
      const Position, Count: Integer; var Caches: TTensorHandles): TArray<Single>;
    function BuildPrompt(const Text: string; const Images: array of string; const Chat: Boolean;
      const Progress: TQwenProgress; const ShouldStop: TQwenStopRequested;
      const Reports: TJSONArray): TPrompt;
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
  {$IFDEF FPC}Windows, Math{$ELSE}Winapi.Windows, System.Math{$ENDIF};

type
  EGenerationStopped = class(Exception);

procedure CheckStop(const ShouldStop: TQwenStopRequested);
begin
  if Assigned(ShouldStop) and ShouldStop() then
    raise EGenerationStopped.Create('Generation stopped');
end;

procedure TGemma3nNative.TPrompt.Add(const Token: Integer);
var
  Index: Integer;
begin
  Index := Length(Tokens);
  SetLength(Tokens, Index + 1);
  Tokens[Index] := Token;
end;

procedure TGemma3nNative.TPrompt.Add(const IDs: array of Integer);
var
  Index: Integer;
begin
  if Length(IDs) = 0 then Exit;
  Index := Length(Tokens);
  SetLength(Tokens, Index + Length(IDs));
  Move(IDs[0], Tokens[Index], Length(IDs) * SizeOf(Integer));
end;

function TGemma3nNative.WeightFile(const Name: string): TMappedModelFile;
var
  Index: Integer;
begin
  Index := FFiles.IndexOf(Name);
  if Index >= 0 then Exit(TMappedModelFile(FFiles.Objects[Index]));
  Result := TMappedModelFile.Create(BundleMember(FFolder, Name));
  try
    FFiles.AddObject(Name, Result);
  except
    Result.Free;
    raise;
  end;
end;

procedure TGemma3nNative.ReadWeights(const Graph, Weights: TJSONObject; var Args: TTensorHandles);
var
  Inputs, RuntimeInputs: TJSONArray;
  Params, Binding, Spec: TJSONObject;
  Shape, FullShape: TArray<Int64>;
  Slice: TJSONValue;
  FileData: TMappedModelFile;
  Offset, Bytes, SliceIndex: Int64;
  I, J: Integer;
  InputName, DType: string;
begin
  Inputs := ModelArray(Graph.GetValue('input_names'), 'graph input names');
  RuntimeInputs := ModelArray(Graph.GetValue('runtime_inputs'), 'graph runtime inputs');
  RequireCpu((RuntimeInputs.Count = FCacheCount + 2 + Ord(FHostEmbeddings)) and
    (Inputs.Count > RuntimeInputs.Count),
    'Unexpected Gemma 3n graph input count');
  if FHostEmbeddings then
    RequireCpu((JSONValueText(RuntimeInputs.Items[0]) = 'embedding') and
      (JSONValueText(RuntimeInputs.Items[1]) = 'per_layer_embedding') and
      (JSONValueText(RuntimeInputs.Items[2]) = 'position'), 'Unexpected Gemma 3n embedding graph signature')
  else
    RequireCpu((JSONValueText(RuntimeInputs.Items[0]) = 'ids') and (JSONValueText(RuntimeInputs.Items[1]) = 'position'),
      'Unexpected Gemma 3n token graph signature');
  for I := 0 to RuntimeInputs.Count - 1 do
    RequireCpu(JSONValueText(RuntimeInputs.Items[I]) = JSONValueText(Inputs.Items[I]),
      'Model runtime input order does not match the graph');
  Params := ModelObject(Graph.GetValue('parameters'), 'graph parameters');
  SetLength(Args, Inputs.Count - RuntimeInputs.Count);
  for I := RuntimeInputs.Count to Inputs.Count - 1 do
  begin
    InputName := JSONValueText(Inputs.Items[I]);
    Binding := ModelObject(Params.GetValue(InputName), InputName);
    Spec := ModelObject(Weights.GetValue(JSONValueText(Binding.GetValue('tensor'))), InputName);
    DType := ModelText(Spec.GetValue('dtype'));
    RequireCpu((DType = '') or (DType = 'float32'), 'Unsupported weight type: ' + DType);
    FullShape := ModelShape(Spec.GetValue('shape'));
    FileData := WeightFile(JSONValueText(Spec.GetValue('file')));
    RequireCpu(FileData.Size = TensorElements(FullShape) * SizeOf(Single),
      'Incorrect file size for ' + InputName);
    Shape := FullShape;
    Offset := 0;
    Slice := Binding.GetValue('slice');
    if (Slice <> nil) and not (Slice is TJSONNull) then
    begin
      SliceIndex := ModelInt(Slice, InputName + ' slice');
      RequireCpu((Length(FullShape) > 1) and (SliceIndex >= 0) and
        (SliceIndex < FullShape[0]), 'Invalid weight slice for ' + InputName);
      Shape := Copy(FullShape, 1, Length(FullShape) - 1);
      Offset := SliceIndex * TensorElements(Shape) * SizeOf(Single);
    end;
    if Binding.GetValue('shape') <> nil then
    begin
      FullShape := ModelShape(Binding.GetValue('shape'));
      RequireCpu(Length(FullShape) = Length(Shape), 'Weight rank does not match ' + InputName);
      for J := 0 to High(Shape) do
        RequireCpu(FullShape[J] = Shape[J], 'Weight shape does not match ' + InputName);
    end;
    Bytes := TensorElements(Shape) * SizeOf(Single);
    Args[I - RuntimeInputs.Count] := MappedFloatTensor(FileData.AtOffset(Offset, Bytes), Shape);
  end;
end;

procedure TGemma3nNative.LoadCuda(const Manifest, Weights: TJSONObject);
const
  MaximumWorkspace = UInt64(1024) * 1024 * 1024;
  DeviceHeadroom = UInt64(64) * 1024 * 1024;
var
  Bindings, Runtime: TJSONArray;
  Original, Quantized, Binding, Spec, SliceBinding: TJSONObject;
  Uploads: TArray<TCudaWeight>;
  Slice: TJSONValue;
  Name, Kind, Path: string;
  FileData: TMappedModelFile;
  I, RuntimeCount: Integer;
  Offset, Index, Bytes: Int64;
  Required, Free, CacheStride: UInt64;
begin
  Runtime := ModelArray(Manifest.GetValue('runtime_inputs'), 'runtime inputs');
  RuntimeCount := Runtime.Count;
  RequireCpu(
    FHostEmbeddings and (RuntimeCount = FCacheCount + 3) and (ModelText(Runtime.Items[0]) = 'embedding') and (ModelText(Runtime.Items[1]) = 'per_layer_embedding') and (ModelText(Runtime.Items[2]) = 'position'),
    'Unexpected Gemma 3n CUDA graph signature'
  );
  Bindings := ModelArray(Manifest.GetValue('gpu_bindings'), 'CUDA input bindings');
  RequireCpu(Bindings.Count > RuntimeCount, 'Missing CUDA weight bindings');
  Original := ModelObject(Manifest.GetValue('parameters'), 'graph parameters');
  Quantized := ModelObject(ModelObject(Manifest.GetValue('quantization'), 'quantization').GetValue('weights'),
    'quantized weights');
  SetLength(Uploads, Bindings.Count - RuntimeCount);
  CacheStride := UInt64(FCacheCount) * SizeOf(Single) * 3;
  Required := UInt64(TensorElements(FCacheShape));
  RequireCpu(Required <= (UInt64(High(UInt64)) - DeviceHeadroom) div CacheStride,
    'CUDA cache size overflow');
  Required := DeviceHeadroom + Required * CacheStride;
  for I := 0 to High(Uploads) do
  begin
    Binding := ModelObject(Bindings.Items[RuntimeCount + I], 'CUDA weight binding');
    Name := ModelText(Binding.GetValue('source'));
    Kind := ModelText(Binding.GetValue('kind'));
    Uploads[I].Code := kDLFloat;
    Uploads[I].Bits := 32;
    Offset := 0;
    if (Kind = 'q8') or (Kind = 'scale') then
    begin
      Spec := ModelObject(Quantized.GetValue(Name), Name);
      Uploads[I].Shape := ModelShape(Spec.GetValue('shape'));
      if Kind = 'scale' then
        SetLength(Uploads[I].Shape, 1)
      else begin
        Uploads[I].Code := kDLInt;
        Uploads[I].Bits := 8;
      end;
      FileData := WeightFile(ModelText(Spec.GetValue(Kind)));
      RequireCpu(FileData.Size = TensorElements(Uploads[I].Shape) * (Uploads[I].Bits div 8),
        'Unexpected quantized weight size: ' + Name);
    end
    else begin
      RequireCpu(Kind = 'original', 'Unknown CUDA weight binding: ' + Kind);
      SliceBinding := ModelObject(Original.GetValue(Name), Name);
      Spec := ModelObject(Weights.GetValue(ModelText(SliceBinding.GetValue('tensor'))), Name);
      RequireCpu((ModelText(Spec.GetValue('dtype')) = '') or
        (ModelText(Spec.GetValue('dtype')) = 'float32'), 'Expected float32 CUDA weight: ' + Name);
      Uploads[I].Shape := ModelShape(Spec.GetValue('shape'));
      FileData := WeightFile(ModelText(Spec.GetValue('file')));
      RequireCpu(FileData.Size = TensorElements(Uploads[I].Shape) * SizeOf(Single),
        'Unexpected CUDA weight size: ' + Name);
      Slice := SliceBinding.GetValue('slice');
      if (Slice <> nil) and not (Slice is TJSONNull) then
      begin
        Index := ModelInt(Slice, Name + ' slice');
        RequireCpu((Length(Uploads[I].Shape) > 1) and (Index >= 0) and (Index < Uploads[I].Shape[0]),
          'CUDA weight slice is out of range: ' + Name);
        Uploads[I].Shape := Copy(Uploads[I].Shape, 1, Length(Uploads[I].Shape) - 1);
        Offset := Index * TensorElements(Uploads[I].Shape) * SizeOf(Single);
      end;
    end;
    Bytes := TensorElements(Uploads[I].Shape) * (Uploads[I].Bits div 8);
    Uploads[I].Data := FileData.AtOffset(Offset, Bytes);
    RequireCpu(UInt64(Bytes) <= High(UInt64) - Required, 'CUDA weight byte count overflow');
    Inc(Required, Bytes);
  end;
  Path := BundleMember(FFolder, ModelText(Manifest.GetValue('library')));
  FCudaArch := ModelText(Manifest.GetValue('cuda_arch'));
  FCudaWeights := Uploads;
  FCuda := TCudaExecution.Create;
  FCuda.RequireArchitecture(FCudaArch);
  FCuda.Enter;
  try
    Free := FCuda.FreeBytes;
    RequireCpu(Free >= Required,
      Format('Gemma 3n requires %.2f GiB of free CUDA memory. Available: %.2f GiB.', [Required / 1073741824.0, Free / 1073741824.0]));
    Free := Free - Required + DeviceHeadroom div 2;
    if Free > MaximumWorkspace then Free := MaximumWorkspace;
    FDecode := TCpuGraph.Create(Path, kDLCUDA);
    FWorkspace := TCudaWorkspace.Create(FCuda, Path, Free);
    UploadCudaWeights(nil);
  finally
    FCuda.Leave;
  end;
end;

procedure TGemma3nNative.ReleaseCudaWeights;
begin
  if Length(FDecodeWeights) = 0 then Exit;
  FCuda.Synchronize;
  ReleaseTensors(FDecodeWeights);
end;

procedure TGemma3nNative.UploadCudaWeights(const Progress: TQwenProgress);
var
  I: Integer;
begin
  if Length(FDecodeWeights) <> 0 then Exit;
  if Assigned(Progress) then Progress('Uploading Gemma 3n weights to CUDA...');
  SetLength(FDecodeWeights, Length(FCudaWeights));
  try
    for I := 0 to High(FCudaWeights) do
      FDecodeWeights[I] := DeviceTensor(FCudaWeights[I].Shape, FCudaWeights[I].Code,
        FCudaWeights[I].Bits, FCudaWeights[I].Data, kDLCUDA);
    FCuda.Synchronize;
  except
    ReleaseTensors(FDecodeWeights);
    raise;
  end;
end;

constructor TGemma3nNative.Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
const
  TableName: array[0..1] of string = ('embed_tokens.weight', 'embed_tokens_per_layer.weight');
var
  ModelData, RecordData: TJSONValue;
  Manifest, RecordObj, Prefill, Weights, Tables, Spec, Hashes: TJSONObject;
  Shape: TArray<Int64>;
  NativeDir, RuntimeDir, LibraryName, PrefillOnnx, VisionManifest: string;
  I: Integer;
  Started: UInt64;
begin
  inherited Create;
  Started := GetTickCount64;
  FFolder := ExpandFileName(Folder);
  FFiles := TStringList.Create;
  FFiles.OwnsObjects := True;
  FFiles.Sorted := True;
  NativeDir := IncludeTrailingPathDelimiter(RuntimeRoot) + 'runtime\gemma-native\';
  RuntimeDir := IncludeTrailingPathDelimiter(RuntimeRoot) +
    'crosspascal-tvm\runtime\windows-x64-cuda\';
  ModelData := ReadLanguageManifest(FFolder);
  try
    Manifest := ModelObject(ModelData, 'Gemma 3n manifest');
    FBackend := ModelText(Manifest.GetValue('backend'));
    RequireCpu(
      (JSONValueText(Manifest.GetValue('schema')) = 'crosspascal.gemma3n.text.v1') and (JSONValueText(Manifest.GetValue('architecture')) = 'gemma3n') and ((FBackend = 'TVM LLVM CPU FP32') or (FBackend = 'TVM CUDA W8A32')),
      'Expected a native Gemma 3n bundle'
    );
    if FBackend = 'TVM CUDA W8A32' then
      FDeviceType := kDLCUDA
    else
      FDeviceType := kDLCPU;
    FHostEmbeddings := JSONIsTrue(Manifest.GetValue('host_embeddings'));
    FSourceHash := JSONValueText(Manifest.GetValue('source_sha256'));
    FContext := ModelInt(Manifest.GetValue('context'), 'context');
    FCacheCount := ModelInt(Manifest.GetValue('cache_count'), 'cache count');
    RequireCpu((FContext >= 2) and (FContext <= 131072) and (FCacheCount > 0) and
      (FCacheCount <= 256), 'Invalid Gemma 3n context or cache count');
    FCacheShape := ModelShape(Manifest.GetValue('cache_shape'));
    Hashes := ModelObject(Manifest.GetValue('artifact_sha256'), 'artifact hashes');
    if Assigned(Progress) then Progress('Verifying the Gemma 3n bundle...');
    VerifyArtifacts(FFolder, Hashes);
    RecordData := ReadModelJSON(BundleMember(FFolder, 'parameters.json'));
    try
      RecordObj := ModelObject(RecordData, 'parameter record');
      Weights := ModelObject(RecordObj.GetValue('weights'), 'weights');
      if FHostEmbeddings then
      begin
        Tables := ModelObject(RecordObj.GetValue('parameters'), 'host embedding tables');
        for I := 0 to High(FTables) do
        begin
          Spec := ModelObject(Tables.GetValue(TableName[I]), TableName[I]);
          Spec := ModelObject(Weights.GetValue(JSONValueText(Spec.GetValue('tensor'))), TableName[I]);
          Shape := ModelShape(Spec.GetValue('shape'));
          RequireCpu((Length(Shape) = 2) and (Shape[0] <= 1000000) and (Shape[1] <= 65536),
            'Invalid host embedding table shape');
          FTables[I].Rows := Shape[0];
          FTables[I].Width := Shape[1];
          FTables[I].Data := WeightFile(JSONValueText(Spec.GetValue('file')));
          RequireCpu(FTables[I].Data.Size = TensorElements(Shape) * SizeOf(Single),
            'Host embedding table size does not match the manifest');
        end;
      end;
      QwenCuda.InitializeRuntime(RuntimeDir, FDeviceType = kDLCUDA);
      FTokenizer := TSentencePiece.Create(NativeDir + 'crosspascal_sentencepiece.dll',
        BundleMember(FFolder, 'tokenizer.model'));
      FBos := FTokenizer.ID('<bos>');
      FEos := FTokenizer.ID('<eos>');
      FStartTurn := FTokenizer.ID('<start_of_turn>');
      FEndTurn := FTokenizer.ID('<end_of_turn>');
      if FDeviceType = kDLCUDA then
        LoadCuda(Manifest, Weights)
      else begin
        ReadWeights(Manifest, Weights, FDecodeWeights);
        FDecode := TCpuGraph.Create(BundleMember(FFolder, 'gemma3n-cpu.dll'));
      end;
      if Manifest.GetValue('prefill') <> nil then
      begin
        Prefill := ModelObject(Manifest.GetValue('prefill'), 'prefill');
        FChunk := ModelInt(Prefill.GetValue('chunk'), 'prefill chunk');
        LibraryName := JSONValueText(Prefill.GetValue('library'));
        PrefillOnnx := ModelText(Prefill.GetValue('onnx'));
        RequireCpu(
          FHostEmbeddings and (FChunk > 1) and (FChunk <= FContext) and (Hashes.GetValue(LibraryName) <> nil) and ((PrefillOnnx = '') or (Hashes.GetValue(PrefillOnnx) <> nil)),
          'Invalid prefill graph'
        );
        ReadWeights(Prefill, Weights, FPrefillWeights);
        FPrefill := TCpuGraph.Create(BundleMember(FFolder, LibraryName));
      end;
      VisionManifest := ModelText(Manifest.GetValue('vision_manifest'));
      if VisionManifest <> '' then
      begin
        RequireCpu(FHostEmbeddings and (Hashes.GetValue(VisionManifest) <> nil),
          'Missing Gemma 3n vision manifest hash');
        FStartImage := FTokenizer.ID('<start_of_image>');
        FEndImage := FTokenizer.ID('<end_of_image>');
        FSoftImage := FTokenizer.ID('<image_soft_token>');
        RequireCpu((FStartImage >= 0) and (FEndImage >= 0) and (FSoftImage >= 0),
          'Gemma 3n tokenizer does not contain the vision markers');
        FVision := TGemma3nVision.Create(FFolder, NativeDir + 'crosscv.dll', FTables[0].Width,
          FCuda, VisionManifest);
      end;
    finally
      RecordData.Free;
    end;
  finally
    ModelData.Free;
  end;
  FLoadSeconds := (GetTickCount64 - Started) / 1000.0;
  if Assigned(Progress) then
    if FCuda <> nil then
      Progress('Model loaded on CUDA through TVM')
    else
      Progress('Model loaded on CPU through TVM');
end;

destructor TGemma3nNative.Destroy;
begin
  try
    if FCuda <> nil then FCuda.Enter;
    try
      FVision.Free;
      FWorkspace.Free;
      FPrefill.Free;
      FDecode.Free;
      ReleaseTensors(FPrefillWeights);
      ReleaseTensors(FDecodeWeights);
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    FCuda.Free;
    FTokenizer.Free;
    FFiles.Free;
    inherited;
  end;
end;

function TGemma3nNative.BuildPrompt(const Text: string; const Images: array of string;
  const Chat: Boolean; const Progress: TQwenProgress; const ShouldStop: TQwenStopRequested;
  const Reports: TJSONArray): TPrompt;
var
  Ending: TArray<Integer>;
  Features, EndEmbedding: TArray<Single>;
  ImagePath, Hash: string;
  Reused: Boolean;
  Index, J: Integer;
  ImageReport: TJSONObject;
  Started: UInt64;
begin
  Result := Default(TPrompt);
  RequireCpu(Trim(Text) <> '', 'Enter a non-empty text prompt');
  if FBos >= 0 then Result.Add(FBos);
  if Chat then
  begin
    RequireCpu((FStartTurn >= 0) and (FEndTurn >= 0), 'This tokenizer has no Gemma chat markers');
    Result.Add(FStartTurn);
  end;
  if Length(Images) = 0 then
  begin
    if Chat then
      Result.Add(FTokenizer.Encode('user' + #10 + Text))
    else
      Result.Add(FTokenizer.Encode(Text));
  end
  else begin
    RequireCpu(FVision <> nil, 'This bundle does not contain a vision encoder');
    if Chat then Result.Add(FTokenizer.Encode('user' + #10));
    Ending := FTokenizer.Encode(#10 + Text);
    RequireCpu(Length(Result.Tokens) + Int64(Length(Images)) * (FVision.SoftTokens + 2) +
      Length(Ending) + 12 < FContext, 'Text and images exceed the compiled context');
    SetLength(Result.Embeddings, Length(Result.Tokens) + Length(Images) * (FVision.SoftTokens + 2));
    EndEmbedding := FVision.EndEmbedding(FEndImage);
    for ImagePath in Images do
    begin
      CheckStop(ShouldStop);
      if Assigned(Progress) then
        if FVision.Backend = 'TVM CUDA FP32' then
          Progress('Encoding image on CUDA...')
        else
          Progress('Encoding image on CPU...');
      Started := GetTickCount64;
      Features := FVision.Encode(ImagePath, ShouldStop, Hash, Reused, ReleaseCudaWeights);
      CheckStop(ShouldStop);
      ImageReport := TJSONObject.Create;
      Reports.AddElement(ImageReport);
      ImageReport.AddPair('sha256', Hash);
      ImageReport.AddPair('reused', JSONBoolean(Reused));
      ImageReport.AddPair('seconds', JSONFloat((GetTickCount64 - Started) / 1000.0));
      ImageReport.AddPair('backend', FVision.Backend);
      Result.Add(FStartImage);
      for J := 0 to FVision.SoftTokens - 1 do
      begin
        Index := Length(Result.Tokens);
        Result.Embeddings[Index] := Copy(Features, J * FTables[0].Width, FTables[0].Width);
        Result.Add(FSoftImage);
      end;
      Result.Embeddings[Length(Result.Tokens)] := EndEmbedding;
      Result.Add(FEndImage);
    end;
    Result.Add(Ending);
  end;
  if Chat then
  begin
    Result.Add(FEndTurn);
    Result.Add(FTokenizer.Encode(#10));
    Result.Add(FStartTurn);
    Result.Add(FTokenizer.Encode('model' + #10));
  end;
  RequireCpu((Length(Result.Tokens) > 0) and (Length(Result.Tokens) < FContext),
    'Prompt does not fit the compiled context');
end;

function TGemma3nNative.Evaluate(const Graph: TCpuGraph; const Weights: TTensorHandles; const Prompt: TPrompt;
  const Position, Count: Integer; var Caches: TTensorHandles): TArray<Single>;
var
  Args, Inputs, NextCaches: TTensorHandles;
  Values, Override: TArray<Single>;
  Positions: TArray<Int64>;
  InputToken: Int64;
  Data: Pointer;
  I, J, Token, Table, Width: Integer;
  Scale: Single;
  Reply: TTVMFFIAny;
  Logits: TTVMFFIObjectHandle;
begin
  SetLength(Inputs, 2 + Ord(FHostEmbeddings));
  SetLength(Args, Length(Inputs) + Length(Caches) + Length(Weights));
  try
    if FHostEmbeddings then
      for Table := 0 to 1 do
      begin
        Width := FTables[Table].Width;
        SetLength(Values, Count * Width);
        Scale := 1 / Sqrt(FTables[0].Width);
        for I := 0 to Count - 1 do
        begin
          Token := Prompt.Tokens[Position + I];
          Override := nil;
          if Position + I < Length(Prompt.Embeddings) then
            Override := Prompt.Embeddings[Position + I];
          if (Table = 0) and (Override <> nil) then
          begin
            for J := 0 to Width - 1 do Values[I * Width + J] := Override[J] * Scale;
            Continue;
          end;
          if (Override <> nil) and ((Token < 0) or (Token >= FTables[Table].Rows)) then Token := 0;
          RequireCpu((Token >= 0) and (Token < FTables[Table].Rows),
            'Token outside the host embedding table');
          Data := FTables[Table].Data.AtOffset(Int64(Token) * Width * SizeOf(Single), Width * SizeOf(Single));
          Move(Data^, Values[I * Width], Width * SizeOf(Single));
        end;
        Inputs[Table] := DeviceTensor([1, Count, Width], kDLFloat, 32, Pointer(Values), FDeviceType);
      end
    else begin
      RequireCpu(Count = 1, 'The token graph accepts one token at a time');
      InputToken := Prompt.Tokens[Position];
      Inputs[0] := CpuTensor([1, 1], kDLInt, 64, @InputToken);
    end;
    SetLength(Positions, Count);
    for I := 0 to Count - 1 do Positions[I] := Position + I;
    Inputs[High(Inputs)] := DeviceTensor([Count], kDLInt, 64, Pointer(Positions), FDeviceType);
    for I := 0 to High(Inputs) do Args[I] := Inputs[I];
    for I := 0 to High(Caches) do Args[Length(Inputs) + I] := Caches[I];
    for I := 0 to High(Weights) do Args[Length(Inputs) + Length(Caches) + I] := Weights[I];
    if FWorkspace <> nil then
      Reply := FWorkspace.Run(Graph, Args)
    else
      Reply := Graph.Run(Args);
    RequireCpu(Reply.VObj <> nil, 'TVM returned no text graph outputs');
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

function TGemma3nNative.Generate(const Prompt: string; const Options: TQwenGenerationOptions;
  const Images: array of string; const Progress, OnText: TQwenProgress;
  const ShouldStop: TQwenStopRequested): string;
const
  EffortBudget: array[TQwenAnswerEffort] of Integer = (0, 128, 512, 1024);
  EffortName: array[TQwenAnswerEffort] of string = ('full_context', 'brief', 'balanced', 'detailed');
var
  State: Cardinal;
  PromptData: TPrompt;
  Caches: TTensorHandles;
  Logits: TArray<Single>;
  Seen: TArray<Boolean>;
  Output: TPrompt;
  Started, Encoded, First: UInt64;
  Count, Position, PromptCount, Limit, Token, I: Integer;
  Sampled, CudaEntered: Boolean;
  Text, Answer, StopReason, LimitReason: string;
  Report, Settings: TJSONObject;
  ImageReports, Modalities, IDs, PromptIDs: TJSONArray;
begin
  ValidateQwenGenerationOptions(Options);
  Started := GetTickCount64;
  Encoded := Started;
  First := 0;
  StopReason := 'context_limit';
  LimitReason := StopReason;
  PromptCount := 0;
  Text := Prompt;
  if Options.SystemPrompt <> '' then Text := Options.SystemPrompt + #10#10 + Prompt;
  Sampled := (Options.Sampling = qsSample) or ((Options.Sampling = qsDefault) and
    (Options.MaxNewTokens = 0));
  if Options.Seed >= 0 then
    State := Cardinal(Options.Seed)
  else
    State := Cardinal(GetTickCount64 xor UInt64(GetCurrentThreadId));
  Report := TJSONObject.Create;
  CudaEntered := False;
  try
    if FCuda <> nil then
    begin
      FCuda.Enter;
      CudaEntered := True;
    end;
    ImageReports := TJSONArray.Create;
    Report.AddPair('images', ImageReports);
    Settings := TJSONObject.Create;
    Report.AddPair('generation_options', Settings);
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
      CheckStop(ShouldStop);
      PromptData := BuildPrompt(Text, Images, not Options.PlainCompletion, Progress, ShouldStop,
        ImageReports);
      Encoded := GetTickCount64;
      PromptCount := Length(PromptData.Tokens);
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
      if FCuda <> nil then UploadCudaWeights(Progress);
      if Assigned(Progress) then
        if FCuda <> nil then
          Progress('Processing the prompt on CUDA...')
        else
          Progress('Processing the prompt on CPU...');
      SetLength(Caches, FCacheCount);
      for I := 0 to High(Caches) do
        Caches[I] := DeviceTensor(FCacheShape, kDLFloat, 32, nil, FDeviceType);
      Position := 0;
      while Position < FContext - 1 do
      begin
        CheckStop(ShouldStop);
        Count := 1;
        if (FPrefill <> nil) and (PromptCount - Position >= FChunk) then Count := FChunk;
        if Count > 1 then
          Logits := Evaluate(FPrefill, FPrefillWeights, PromptData, Position, Count, Caches)
        else
          Logits := Evaluate(FDecode, FDecodeWeights, PromptData, Position, Count, Caches);
        CheckStop(ShouldStop);
        Inc(Position, Count);
        if Position < PromptCount then Continue;
        RequireCpu(Length(Logits) > 0, 'TVM returned empty logits');
        Token := 0;
        for I := 0 to High(Logits) do
        begin
          RequireCpu(not IsNan(Logits[I]) and not IsInfinite(Logits[I]), 'TVM returned non-finite logits');
          if Logits[I] > Logits[Token] then Token := I;
        end;
        if First = 0 then
        begin
          First := GetTickCount64;
          SetLength(Seen, Length(Logits));
          for I in PromptData.Tokens do if (I >= 0) and (I < Length(Seen)) then Seen[I] := True;
        end;
        if Sampled then Token := SampleToken(Logits, Seen, Options, State);
        if (Token = FEos) or (Token = FEndTurn) then
        begin
          StopReason := 'eos';
          Break;
        end;
        Seen[Token] := True;
        Output.Add(Token);
        PromptData.Add(Token);
        Answer := FTokenizer.Decode(Output.Tokens);
        if Assigned(OnText) then OnText(Answer);
        if Length(Output.Tokens) >= Limit then
        begin
          StopReason := LimitReason;
          Break;
        end;
      end;
    except
      on EGenerationStopped do StopReason := 'user_stop';
      on ECpuGraphStopped do StopReason := 'user_stop';
    end;
    Report.AddPair('text', Answer);
    Report.AddPair('stop', StopReason);
    Report.AddPair('backend', FBackend);
    if FCuda <> nil then Report.AddPair('compiled_arch', FCudaArch);
    Report.AddPair('source_sha256', FSourceHash);
    Report.AddPair('load_seconds', JSONFloat(FLoadSeconds));
    Report.AddPair('encoding_seconds', JSONFloat((Encoded - Started) / 1000.0));
    Report.AddPair('total_seconds', JSONFloat((GetTickCount64 - Started) / 1000.0));
    if First = 0 then
      Report.AddPair('first_token_seconds', TJSONNull.Create)
    else
      Report.AddPair('first_token_seconds', JSONFloat((First - Started) / 1000.0));
    Report.AddPair('prompt_tokens', JSONInteger(PromptCount));
    Report.AddPair('generated_tokens', JSONInteger(Length(Output.Tokens)));
    Report.AddPair('model_reused', JSONBoolean(True));
    IDs := TJSONArray.Create;
    Report.AddPair('token_ids', IDs);
    for Token in Output.Tokens do IDs.AddElement(JSONInteger(Token));
    PromptIDs := TJSONArray.Create;
    Report.AddPair('prompt_token_ids', PromptIDs);
    for I := 0 to PromptCount - 1 do PromptIDs.AddElement(JSONInteger(PromptData.Tokens[I]));
    Modalities := TJSONArray.Create;
    Report.AddPair('modalities', Modalities);
    Modalities.AddElement(TJSONString.Create('text'));
    if Length(Images) > 0 then Modalities.AddElement(TJSONString.Create('image'));
    Result := Report.ToJSON;
  finally
    try
      ReleaseTensors(Caches);
    finally
      Report.Free;
      if CudaEntered then FCuda.Leave;
    end;
  end;
end;

end.
