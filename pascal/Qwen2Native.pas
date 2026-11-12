{ ************************************************************************** }
{                                                                            }
{ Qwen2Native                                                                }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit Qwen2Native;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  QwenCuda, QwenTokenizer, TVM.CudaExecution, TVM.TokenGraph;

type
  TQwen2Native = class
  private
    FCuda: TCudaExecution;
    FGraph: TTokenGraph;
    FTokenizer: TQwenTokenizer;
    FInfo: TQwenCudaResult;
    FContext: Integer;
    FChunk: Integer;
    FEOS: Integer;
    FEndText: Integer;
  public
    constructor Create(const Folder, Runtime: string; const Progress: TQwenProgress);
    destructor Destroy; override;
    function Generate(const Prompt: string; const Options: TQwenGenerationOptions;
      const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): TQwenCudaResult;
    property ContextTokens: Integer read FContext;
  end;

implementation

uses
  {$IFDEF FPC}Classes, Math, SysUtils, Windows{$ELSE}System.Classes, System.Math, System.SysUtils, Winapi.Windows{$ENDIF},
  CrossPascal.Json, CrossPascal.SHA256, TVM.CpuGraph;

constructor TQwen2Native.Create(const Folder, Runtime: string; const Progress: TQwenProgress);
var
  Data, Tokens: TJSONValue;
  Model, Hashes: TJSONObject;
  Context, Chunk: Int64;
  Available, Total: UInt64;
  Architecture, Path: string;
  Stream: TFileStream;
begin
  inherited Create;
  Path := BundleMember(Folder, 'model.json');
  Stream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    FInfo.ModelMetadataSHA256 := SHA256Stream(Stream);
    Data := ReadModelJSON(Path);
  finally
    Stream.Free;
  end;
  try
    Model := ModelObject(Data, 'Qwen2 manifest');
    RequireCpu(
      (ModelText(Model.GetValue('schema')) = 'crosspascal.qwen2.text.v1') and (ModelText(Model.GetValue('backend')) = 'TVM CUDA FP32'),
      'Expected a native Qwen2 CUDA bundle'
    );
    Context := ModelInt(Model.GetValue('context_window_size'), 'Qwen2 context');
    Chunk := ModelInt(Model.GetValue('prefill_chunk_size'), 'Qwen2 prefill chunk');
    RequireCpu((Context >= 2) and (Context <= 131072) and (Chunk >= 2) and (Chunk <= Context),
      'Invalid Qwen2 compiled context');
    FContext := Context;
    FChunk := Chunk;
    FInfo.ContextTokens := FContext;
    FInfo.ProfileID := 'crosspascal.qwen2.text.v1';
    Hashes := ModelObject(Model.GetValue('artifact_sha256'), 'Qwen2 hashes');
    FInfo.ModelSHA256 := ModelText(Hashes.GetValue('qwen2-cuda.dll'));
    FInfo.ParametersSHA256 := ModelText(Hashes.GetValue('weights.bin'));
    if Assigned(Progress) then Progress('Verifying the Qwen2 CUDA bundle...');
    InitializeRuntime(IncludeTrailingPathDelimiter(Runtime), True);
    FCuda := TCudaExecution.Create;
    FInfo.DeviceName := FCuda.DeviceName;
    FInfo.ComputeVersion := FCuda.ComputeVersion;
    Architecture := 'sm_' + Copy(FInfo.ComputeVersion, 1, Pos('.', FInfo.ComputeVersion) - 1) +
      Copy(FInfo.ComputeVersion, Pos('.', FInfo.ComputeVersion) + 1, MaxInt);
    RequireCpu(Architecture = ModelText(Model.GetValue('cuda_arch')),
      'The Qwen2 bundle targets another CUDA compute architecture');
    FCuda.Enter;
    try
      FCuda.MemoryBytes(Available, Total);
      FInfo.FreeDeviceBytes := Available;
      FInfo.TotalDeviceBytes := Total;
      FGraph := TTokenGraph.Create(Folder, 'qwen2-cuda.dll', 'qwen2-prefill-cuda.dll',
        Model, FContext, FChunk, ['tokenizer.json', 'config.json'], FCuda);
    finally
      FCuda.Leave;
    end;
  finally
    Data.Free;
  end;
  FTokenizer := TQwenTokenizer.Create;
  Tokens := ReadModelJSON(BundleMember(Folder, 'tokenizer.json'));
  try
    FTokenizer.LoadByteLevelJSON(ModelObject(Tokens, 'Qwen2 tokenizer'));
  finally
    Tokens.Free;
  end;
  FEOS := FTokenizer.SpecialId('<|im_end|>');
  FEndText := FTokenizer.SpecialId('<|endoftext|>');
  RequireCpu((FEOS >= 0) and (FEOS < FGraph.Vocabulary) and (FEndText >= 0) and
    (FEndText < FGraph.Vocabulary), 'Qwen2 tokenizer has no chat end tokens');
  if Assigned(Progress) then Progress('Model loaded on CUDA through TVM');
end;

destructor TQwen2Native.Destroy;
begin
  try
    if FGraph <> nil then
    begin
      FCuda.Enter;
      try
        FreeAndNil(FGraph);
      finally
        FCuda.Leave;
      end;
    end;
  finally
    FCuda.Free;
    FTokenizer.Free;
    inherited;
  end;
end;

function TQwen2Native.Generate(const Prompt: string; const Options: TQwenGenerationOptions;
  const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): TQwenCudaResult;
const
  EffortBudget: array[TQwenAnswerEffort] of Integer = (0, 128, 512, 1024);
var
  Caches: TTensorHandles;
  Current: TArray<Integer>;
  Logits: TArray<Single>;
  Seen: TArray<Boolean>;
  Chat: string;
  State: Cardinal;
  Started: UInt64;
  Position, Limit, I, Token, Count: Integer;
begin
  ValidateQwenGenerationOptions(Options);
  RequireCpu(Trim(Prompt) <> '', 'Enter a nonempty prompt.');
  Result := FInfo;
  Result.ModelReused := True;
  Result.GenerationOptions := Options;
  Result.Prompt := Prompt;
  Chat := Prompt;
  if not Options.PlainCompletion then
  begin
    Chat := '<|im_start|>user'#10 + Prompt + '<|im_end|>'#10'<|im_start|>assistant'#10;
    if Options.SystemPrompt <> '' then
      Chat := '<|im_start|>system'#10 + Options.SystemPrompt + '<|im_end|>'#10 + Chat;
  end;
  Result.PromptTokenIDs := FTokenizer.Encode(Chat);
  Count := Length(Result.PromptTokenIDs);
  RequireCpu((Count > 0) and (Count < FContext), 'Prompt does not fit the bundle context');
  Result.AvailableNewTokens := FContext - Count;
  Limit := Result.AvailableNewTokens;
  Result.StopReason := 'context_limit';
  if (Options.MaxNewTokens > 0) and (Options.MaxNewTokens < Limit) then
  begin
    Limit := Options.MaxNewTokens;
    Result.StopReason := 'max_new_tokens';
  end
  else if (Options.MaxNewTokens = 0) and (EffortBudget[Options.Effort] > 0) and
    (EffortBudget[Options.Effort] < Limit) then
      begin
        Limit := EffortBudget[Options.Effort];
        Result.StopReason := 'effort_limit';
      end;
  Result.Sampled := (Options.Sampling = qsSample) or ((Options.Sampling = qsDefault) and
    (Options.MaxNewTokens = 0));
  if Result.Sampled then
  begin
    if Options.Seed >= 0 then
      State := Cardinal(Options.Seed)
    else
      State := Cardinal(GetTickCount64);
    Result.SamplingSeed := State;
    SetLength(Seen, FGraph.Vocabulary);
    for Token in Result.PromptTokenIDs do
      if (Token >= 0) and (Token < Length(Seen)) then Seen[Token] := True;
  end;
  Started := GetTickCount64;
  Position := 0;
  if Assigned(Progress) then Progress('Processing the prompt on CUDA...');
  FCuda.Enter;
  try
    Caches := FGraph.NewCache;
    try
      Current := Copy(Result.PromptTokenIDs, 0, FChunk);
      while Length(Result.GeneratedTokenIDs) < Limit do
      begin
        if Assigned(ShouldStop) and ShouldStop() then
        begin
          Result.StopReason := 'cancelled';
          Break;
        end;
        if Length(Current) <> FChunk then SetLength(Current, 1);
        try
          Logits := FGraph.Evaluate(Current, Position, Caches, TCpuGraphStop(ShouldStop));
        except
          on ECpuGraphStopped do
          begin
            Result.StopReason := 'cancelled';
            Break;
          end;
        end;
        Inc(Position, Length(Current));
        if Position < Count then
        begin
          Current := Copy(Result.PromptTokenIDs, Position, FChunk);
          Continue;
        end;
        RequireCpu(Length(Logits) = FGraph.Vocabulary, 'Unexpected Qwen2 logits count');
        Token := 0;
        for I := 0 to High(Logits) do
        begin
          RequireCpu(not IsNan(Logits[I]) and not IsInfinite(Logits[I]), 'Non-finite Qwen2 logits');
          if Logits[I] > Logits[Token] then Token := I;
        end;
        if Result.Sampled then
        begin
          Token := SampleToken(Logits, Seen, Options, State);
          Seen[Token] := True;
        end;
        Result.GeneratedTokenIDs := Result.GeneratedTokenIDs + [Token];
        if (Token = FEOS) or (Token = FEndText) then
        begin
          Result.StopReason := 'eos';
          Break;
        end;
        Result.Answer := FTokenizer.DecodePrefix(Result.GeneratedTokenIDs);
        if Assigned(OnText) then OnText(Result.Answer);
        Current := [Token];
      end;
    finally
      ReleaseTensors(Caches);
    end;
  finally
    FCuda.Leave;
  end;
  Result.ElapsedMilliseconds := GetTickCount64 - Started;
end;

end.
