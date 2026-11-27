{ ************************************************************************** }
{                                                                            }
{ Qwen35Session                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit Qwen35Session;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.NativeIO, {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, QwenCuda,
  Qwen35Native, QwenTokenizer, CrossPascal.Json, Qwen35Vision;

type
  TQwen35Session = class
  private
    FModel: TQwen35Native;
    FTokenizer: TQwenTokenizer;
    FSourceHashes: TJSONObject;
    FBackend: string;
    FDevice: string;
    FSourceRepository: string;
    FSourceRevision: string;
    FLoadSeconds: Double;
    FEOS: Integer;
    FEndText: Integer;
    FContext: Integer;
    FVision: TQwen35Vision;
    FVisionStart: Integer;
    FVisionEnd: Integer;
    FImagePad: Integer;
  public
    constructor Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
    destructor Destroy; override;
    function Generate(const Prompt: string; const Options: TQwenGenerationOptions;
      const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): string; overload;
    function Generate(const Prompt: string; const Options: TQwenGenerationOptions;
      const Images: array of string; const Progress, OnText: TQwenProgress;
      const ShouldStop: TQwenStopRequested): string; overload;
    property ContextTokens: Integer read FContext;
  end;

implementation

uses
  {$IFDEF FPC}Windows, Classes, Math{$ELSE}Winapi.Windows, System.Classes, System.Math{$ENDIF},
  TVM.CpuGraph, TVM.CudaExecution, CrossPascal.SHA256, Qwen35PackedCuda, Qwen35CudaBuffers,
  Qwen35Prompt;

constructor TQwen35Session.Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
var
  Data: TJSONValue;
  Manifest, Hashes, Sources: TJSONObject;
  RuntimeDirectory, Decoder, TokenizerFile, Name, Storage, VisionManifest, CVLibrary: string;
  Stream: TStream;
  I: Integer;
  Started: UInt64;
begin
  inherited Create;
  Started := GetTickCount64;
  Data := ReadLanguageManifest(Folder);
  try
    Manifest := ModelObject(Data, 'Qwen manifest');
    RequireCpu(ModelText(Manifest.GetValue('schema')) = 'crosspascal.qwen35.text.v1',
      'Expected a native Qwen 3.5/3.8 text bundle');
    FDevice := ModelText(Manifest.GetValue('device'));
    RequireCpu((FDevice = 'cuda') or (FDevice = 'cpu'), 'Unknown Qwen target device');
    Storage := ModelText(Manifest.GetValue('weight_storage'));
    if Storage = 'bfloat16' then
      FBackend := 'TVM ' + UpperCase(FDevice) + ' BF16 streamed'
    else begin
      RequireCpu((Storage = 'float32') and (ModelText(Manifest.GetValue('source_convention')) = 'gguf'),
        'Unknown Qwen source storage');
      FBackend := 'TVM ' + UpperCase(FDevice) + ' mixed GGUF streamed';
    end;
    Hashes := ModelObject(Manifest.GetValue('artifact_sha256'), 'artifact hashes');
    Sources := ModelObject(Manifest.GetValue('source_files'), 'source weight files');
    FSourceHashes := TJSONObject.Create;
    for I := 0 to Sources.Count - 1 do
    begin
      Name := JSONObjectName(Sources, I);
      RequireCpu(Hashes.GetValue(Name) <> nil, 'Missing Qwen source weight digest');
      FSourceHashes.AddPair(Name, ModelText(Hashes.GetValue(Name)));
    end;
    FSourceRepository := ModelText(Manifest.GetValue('source_repository'));
    FSourceRevision := ModelText(Manifest.GetValue('source_revision'));
    TokenizerFile := ModelText(Manifest.GetValue('tokenizer'));
    if TokenizerFile = '' then TokenizerFile := 'tokenizer.json';
    RequireCpu(Hashes.GetValue(TokenizerFile) <> nil, 'Missing Qwen tokenizer hash');
    if Assigned(Progress) then Progress('Verifying the Qwen tokenizer...');
    Stream := TNativeReadStream.Create(BundleMember(Folder, TokenizerFile));
    try
      RequireCpu(SameText(SHA256Stream(Stream), ModelText(Hashes.GetValue(TokenizerFile))),
        'Qwen tokenizer changed');
    finally
      Stream.Free;
    end;
    FTokenizer := TQwenTokenizer.Create;
    FTokenizer.LoadByteLevelFile(BundleMember(Folder, TokenizerFile));
    RequireCpu(FTokenizer.TokenCount = ModelInt(Manifest.GetValue('effective_vocab'),
      'tokenizer vocabulary'), 'Qwen tokenizer vocabulary differs from manifest');
    FEOS := FTokenizer.SpecialId('<|im_end|>');
    FEndText := FTokenizer.SpecialId('<|endoftext|>');
    RequireCpu((FEOS >= 0) and (FEndText >= 0), 'Qwen tokenizer lacks end markers');
    Decoder := ModelText(Manifest.GetValue('tensor_decoder'));
    if Decoder <> '' then
    begin
      RequireCpu(Hashes.GetValue(Decoder) <> nil, 'Missing Qwen tensor decoder hash');
      Decoder := BundleMember(Folder, Decoder);
    end
    else if ModelText(Manifest.GetValue('source_convention')) = 'gguf' then
    begin
      Decoder := IncludeTrailingPathDelimiter(RuntimeRoot) + 'runtime\qwen35-native\ggml-base.dll';
      RequireCpu(FileExists(Decoder), 'A model reader component is missing. Run Prepare.cmd again.');
    end;
    RuntimeDirectory := IncludeTrailingPathDelimiter(RuntimeRoot);
    if not FileExists(RuntimeDirectory + 'tvm_ffi.dll') then
      RuntimeDirectory := RuntimeDirectory + 'crosspascal-tvm\runtime\windows-x64-cuda\';
    if Assigned(Progress) then Progress('Verifying and loading the Qwen source weights...');
    FModel := TQwen35Native.Create(Folder, RuntimeDirectory, Decoder);
    FContext := FModel.ContextTokens;
    VisionManifest := ModelText(Manifest.GetValue('vision_manifest'));
    if VisionManifest <> '' then
    begin
      BundleMember(Folder, VisionManifest);
      FVisionStart := FTokenizer.SpecialId('<|vision_start|>');
      FVisionEnd := FTokenizer.SpecialId('<|vision_end|>');
      FImagePad := FTokenizer.SpecialId('<|image_pad|>');
      RequireCpu(
        (FVisionStart >= 0) and (FVisionStart < FTokenizer.TokenCount) and (FVisionEnd >= 0) and (FVisionEnd < FTokenizer.TokenCount) and (FImagePad >= 0) and (FImagePad < FTokenizer.TokenCount),
        'Qwen tokenizer lacks vision markers'
      );
      CVLibrary := IncludeTrailingPathDelimiter(RuntimeRoot) + 'runtime\gemma-native\crosscv.dll';
      RequireCpu(FileExists(CVLibrary), 'An image reader component is missing. Run Prepare.cmd again.');
      if Assigned(Progress) then Progress('Verifying and loading the Qwen vision encoder...');
      FVision := TQwen35Vision.Create(Folder, CVLibrary, FModel.HiddenSize, FModel.CudaExecution,
        VisionManifest);
    end;
  finally
    Data.Free;
  end;
  FLoadSeconds := (GetTickCount64 - Started) / 1000.0;
  if Assigned(Progress) then
    Progress('Qwen model loaded on ' + UpperCase(FDevice) + ' through TVM');
end;

destructor TQwen35Session.Destroy;
begin
  try
    FreeAndNil(FVision);
  finally
    FModel.Free;
    FTokenizer.Free;
    FSourceHashes.Free;
    inherited;
  end;
end;

function TQwen35Session.Generate(const Prompt: string; const Options: TQwenGenerationOptions;
  const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): string;
var
  NoImages: TArray<string>;
begin
  NoImages := nil;
  Result := Generate(Prompt, Options, NoImages, Progress, OnText, ShouldStop);
end;

function TQwen35Session.Generate(const Prompt: string; const Options: TQwenGenerationOptions;
  const Images: array of string; const Progress, OnText: TQwenProgress;
  const ShouldStop: TQwenStopRequested): string;
const
  EffortBudget: array[TQwenAnswerEffort] of Integer = (0, 128, 512, 1024);
  EffortNames: array[TQwenAnswerEffort] of string = ('full_context', 'brief', 'balanced', 'detailed');
var
  Tokens, Output, PrefixIDs, NewlineIDs, TailIDs: TArray<Integer>;
  Logits, Features: TArray<Single>;
  Seen: TArray<Boolean>;
  Positions: TArray<TQwen35RopePosition>;
  ImageHashes: TArray<string>;
  ImageReused: TArray<Boolean>;
  ImageSeconds: TArray<Double>;
  Prepared: TQwen35PreparedPrompt;
  Chat, Answer, Reason, Prefix, Tail, Hash: string;
  Limit, I, J, Token, GridRows, GridColumns, Y, X: Integer;
  Required, Base: Int64;
  State: Cardinal;
  Started, First, ImageStarted, PreparedAt: UInt64;
  Sampled, HasImages, Reused: Boolean;
  Report, Settings, Sources, Cache, Pool, Entry: TJSONObject;
  CacheStats: TQwen35PackedCacheStats;
  PoolStats: TQwen35DecodedPoolStats;
  IDs, PromptIDs, Modalities, ImageList: TJSONArray;
begin
  ValidateQwenGenerationOptions(Options);
  RequireCpu(Trim(Prompt) <> '', 'Enter a nonempty prompt.');
  HasImages := Length(Images) > 0;
  GridRows := 0;
  GridColumns := 0;
  if not HasImages then
  begin
    Chat := Prompt;
    if not Options.PlainCompletion then
    begin
      Chat := '<|im_start|>user'#10 + Prompt + '<|im_end|>'#10'<|im_start|>assistant'#10;
      if Options.SystemPrompt <> '' then
        Chat := '<|im_start|>system'#10 + Options.SystemPrompt + '<|im_end|>'#10 + Chat;
      Chat := Chat + '<think>'#10#10'</think>'#10#10;
    end;
    Tokens := FTokenizer.Encode(Chat);
    RequireCpu((Length(Tokens) > 0) and (Length(Tokens) < FContext),
      'Prompt does not fit the bundle context');
  end
  else begin
    RequireCpu(FVision <> nil, 'This Qwen bundle has no vision encoder. Send text only or load a bundle with a vision_manifest.');
    Prefix := '';
    Tail := #10 + Prompt;
    if not Options.PlainCompletion then
    begin
      Prefix := '<|im_start|>user'#10;
      if Options.SystemPrompt <> '' then
        Prefix := '<|im_start|>system'#10 + Options.SystemPrompt + '<|im_end|>'#10 + Prefix;
      Tail := Tail + '<|im_end|>'#10'<|im_start|>assistant'#10'<think>'#10#10'</think>'#10#10;
    end;
    PrefixIDs := FTokenizer.Encode(Prefix);
    NewlineIDs := FTokenizer.Encode(UnicodeString(#10));
    TailIDs := FTokenizer.Encode(Tail);
    GridRows := FVision.GridHeight div FVision.MergeSize;
    GridColumns := FVision.GridWidth div FVision.MergeSize;
    RequireCpu((GridRows > 0) and (GridColumns > 0) and (Int64(GridRows) * GridColumns = FVision.SoftTokens),
      'Qwen vision merger grid differs from its soft token count');
    Required := Int64(Length(PrefixIDs)) + Int64(Length(Images)) * (Int64(FVision.SoftTokens) + 2) +
      Int64(Length(Images) - 1) * Length(NewlineIDs) + Length(TailIDs);
    RequireCpu(Required < FContext, Format('The prompt with %d image(s) needs %d input tokens ' +
      '(%d per image), but this Qwen bundle was compiled for %d context tokens and needs at least ' +
      'one more token for the answer. Shorten the prompt or send fewer images.',
      [Length(Images), Required, FVision.SoftTokens + 2, FContext]));
    SetLength(Tokens, Integer(Required));
    J := 0;
    for Token in PrefixIDs do
    begin
      Tokens[J] := Token;
      Inc(J);
    end;
    for I := 0 to High(Images) do
    begin
      if I > 0 then
        for Token in NewlineIDs do
        begin
          Tokens[J] := Token;
          Inc(J);
        end;
      Tokens[J] := FVisionStart;
      Inc(J);
      for X := 1 to FVision.SoftTokens do
      begin
        Tokens[J] := FImagePad;
        Inc(J);
      end;
      Tokens[J] := FVisionEnd;
      Inc(J);
    end;
    for Token in TailIDs do
    begin
      Tokens[J] := Token;
      Inc(J);
    end;
    RequireCpu(J = Length(Tokens), 'Qwen image prompt planning is inconsistent');
  end;
  Limit := FContext - Length(Tokens);
  Reason := 'context_limit';
  if (Options.MaxNewTokens > 0) and (Options.MaxNewTokens < Limit) then
  begin
    Limit := Options.MaxNewTokens;
    Reason := 'max_new_tokens';
  end
  else
    if (Options.MaxNewTokens = 0) and (EffortBudget[Options.Effort] > 0) and
      (EffortBudget[Options.Effort] < Limit) then
      begin
        Limit := EffortBudget[Options.Effort];
        Reason := 'effort_limit';
      end;
  Sampled := (Options.Sampling = qsSample) or ((Options.Sampling = qsDefault) and
    (Options.MaxNewTokens = 0));
  if Options.Seed >= 0 then
    State := Cardinal(Options.Seed)
  else
    State := Cardinal(GetTickCount64);
  SetLength(Seen, FTokenizer.TokenCount);
  for Token in Tokens do Seen[Token] := True;
  Started := GetTickCount64;
  First := 0;
  PreparedAt := 0;
  Answer := '';
  if not HasImages and Assigned(Progress) then
    Progress('Processing the prompt on ' + UpperCase(FDevice) + '...');
  try
    if HasImages then
    begin
      FModel.PrepareImageEncoding;
      Prepared := Default(TQwen35PreparedPrompt);
      Prepared.AppendText(PrefixIDs);
      SetLength(Positions, FVision.SoftTokens);
      for I := 0 to High(Images) do
      begin
        if I > 0 then Prepared.AppendText(NewlineIDs);
        Prepared.AppendText([FVisionStart]);
        if Assigned(Progress) then
          Progress(Format('Encoding image %d/%d on %s...', [I + 1, Length(Images), UpperCase(FDevice)]));
        ImageStarted := GetTickCount64;
        Features := FVision.Encode(Images[I], ShouldStop, Hash, Reused);
        ImageHashes := ImageHashes + [Hash];
        ImageReused := ImageReused + [Reused];
        ImageSeconds := ImageSeconds + [(GetTickCount64 - ImageStarted) / 1000.0];
        Base := Prepared.NextRope;
        for Y := 0 to GridRows - 1 do
          for X := 0 to GridColumns - 1 do
          begin
            Positions[Y * GridColumns + X][0] := Base;
            Positions[Y * GridColumns + X][1] := Base + Y;
            Positions[Y * GridColumns + X][2] := Base + X;
          end;
        Prepared.AppendVisualRows(FImagePad, Positions, Features, FModel.HiddenSize);
        Features := nil;
        Prepared.AppendText([FVisionEnd]);
      end;
      Prepared.AppendText(TailIDs);
      RequireCpu(Length(Prepared.Rows) = Length(Tokens), 'Qwen image prompt differs from its token budget');
      for I := 0 to High(Prepared.Rows) do
        RequireCpu(Prepared.Rows[I].Token = Tokens[I], 'Qwen image prompt differs from its planned tokens');
      PreparedAt := GetTickCount64;
      if Assigned(Progress) then Progress('Processing the prompt on ' + UpperCase(FDevice) + '...');
      Logits := FModel.PrefillPrepared(Prepared, ShouldStop, Progress);
      Prepared := Default(TQwen35PreparedPrompt);
    end
    else
      Logits := FModel.Prefill(Tokens, ShouldStop, Progress);
    while Length(Output) < Limit do
    begin
      if Assigned(ShouldStop) and ShouldStop() then
      begin
        Reason := 'user_stop';
        Break;
      end;
      RequireCpu(Length(Logits) = Length(Seen), 'Qwen returned a different vocabulary');
      Token := 0;
      for I := 1 to High(Logits) do if Logits[I] > Logits[Token] then Token := I;
      if Sampled then Token := SampleToken(Logits, Seen, Options, State);
      if First = 0 then First := GetTickCount64;
      Output := Output + [Token];
      if (Token = FEOS) or (Token = FEndText) then
      begin
        Reason := 'eos';
        Break;
      end;
      Seen[Token] := True;
      Answer := FTokenizer.DecodePrefix(Output);
      if Assigned(OnText) then OnText(Answer);
      if Length(Output) = Limit then Break;
      Logits := FModel.Decode(Token, ShouldStop, Progress);
    end;
  except
    on ECpuGraphStopped do Reason := 'user_stop';
  end;
  if Reason <> 'user_stop' then Answer := FTokenizer.Decode(Output);
  Report := TJSONObject.Create;
  try
    Report.AddPair('text', Answer);
    Report.AddPair('stop', Reason);
    Report.AddPair('backend', FBackend);
    Report.AddPair('architecture', 'qwen3_5');
    Report.AddPair('source_repository', FSourceRepository);
    Report.AddPair('source_revision', FSourceRevision);
    Sources := TJSONObject.Create;
    Report.AddPair('source_weight_file_sha256', Sources);
    for I := 0 to FSourceHashes.Count - 1 do
      Sources.AddPair(JSONObjectName(FSourceHashes, I), ModelText(JSONObjectValue(FSourceHashes, I)));
    Report.AddPair('experimental', JSONBoolean(True));
    CacheStats := FModel.PackedCacheStats;
    Cache := TJSONObject.Create;
    Report.AddPair('packed_cuda_cache', Cache);
    Cache.AddPair('counters_scope', 'model_lifetime');
    Cache.AddPair('initial_budget_bytes', JSONUnsigned(CacheStats.InitialBudgetBytes));
    Cache.AddPair('budget_bytes', JSONUnsigned(CacheStats.BudgetBytes));
    Cache.AddPair('resident_bytes', JSONUnsigned(CacheStats.ResidentBytes));
    Cache.AddPair('hits', JSONUnsigned(CacheStats.Hits));
    Cache.AddPair('misses', JSONUnsigned(CacheStats.Misses));
    Cache.AddPair('pressure_clears', JSONUnsigned(CacheStats.PressureClears));
    PoolStats := FModel.DecodedPoolStats;
    Pool := TJSONObject.Create;
    Report.AddPair('decoded_cuda_buffers', Pool);
    Pool.AddPair('enabled', JSONBoolean(PoolStats.PlannedBytes <> 0));
    Pool.AddPair('counters_scope', 'model_lifetime');
    Pool.AddPair('planned_bytes', JSONUnsigned(PoolStats.PlannedBytes));
    Pool.AddPair('allocated_bytes', JSONUnsigned(PoolStats.AllocatedBytes));
    Pool.AddPair('allocations', JSONUnsigned(PoolStats.Allocations));
    Pool.AddPair('rents', JSONUnsigned(PoolStats.Rents));
    Pool.AddPair('groups', JSONUnsigned(PoolStats.Groups));
    Report.AddPair('model_reused', JSONBoolean(True));
    Report.AddPair('load_seconds', JSONFloat(FLoadSeconds));
    Report.AddPair('prompt_tokens', JSONInteger(Length(Tokens)));
    Report.AddPair('generated_tokens', JSONInteger(Length(Output)));
    Report.AddPair('total_seconds', JSONFloat((GetTickCount64 - Started) / 1000.0));
    if First = 0 then
      Report.AddPair('first_token_seconds', TJSONNull.Create)
    else
      Report.AddPair('first_token_seconds', JSONFloat((First - Started) / 1000.0));
    IDs := TJSONArray.Create;
    PromptIDs := TJSONArray.Create;
    Report.AddPair('token_ids', IDs);
    Report.AddPair('prompt_token_ids', PromptIDs);
    for Token in Output do IDs.AddElement(JSONInteger(Token));
    for Token in Tokens do PromptIDs.AddElement(JSONInteger(Token));
    Modalities := TJSONArray.Create;
    Modalities.AddElement(TJSONString.Create('text'));
    if Length(ImageHashes) > 0 then Modalities.AddElement(TJSONString.Create('image'));
    Report.AddPair('modalities', Modalities);
    if HasImages then
    begin
      ImageList := TJSONArray.Create;
      Report.AddPair('images', ImageList);
      for I := 0 to High(ImageHashes) do
      begin
        Entry := TJSONObject.Create;
        ImageList.AddElement(Entry);
        Entry.AddPair('sha256', ImageHashes[I]);
        Entry.AddPair('reused', JSONBoolean(ImageReused[I]));
        Entry.AddPair('seconds', JSONFloat(ImageSeconds[I]));
        Entry.AddPair('backend', FVision.Backend);
      end;
      if PreparedAt = 0 then PreparedAt := GetTickCount64;
      Report.AddPair('encoding_seconds', JSONFloat((PreparedAt - Started) / 1000.0));
    end;
    Settings := TJSONObject.Create;
    Report.AddPair('generation_options', Settings);
    if Sampled then
      Settings.AddPair('sampling', 'sample')
    else
      Settings.AddPair('sampling', 'greedy');
    Settings.AddPair('temperature', JSONFloat(Options.Temperature));
    Settings.AddPair('top_p', JSONFloat(Options.TopP));
    Settings.AddPair('top_k', JSONInteger(Options.TopK));
    Settings.AddPair('repetition_penalty', JSONFloat(Options.RepetitionPenalty));
    Settings.AddPair('effort', EffortNames[Options.Effort]);
    Settings.AddPair('plain_completion', JSONBoolean(Options.PlainCompletion));
    Result := Report.ToJSON;
  finally
    Report.Free;
  end;
end;

end.
