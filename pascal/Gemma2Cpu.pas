{ ************************************************************************** }
{                                                                            }
{ Gemma2Cpu                                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit Gemma2Cpu;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, QwenCuda, QwenModel, GemmaTokenizer,
  TVM.CudaExecution, TVM.TokenGraph;

type
  TGemma2Cpu = class
  private
    FModel: TQwenModel;
    FNative: TTokenGraph;
    FTokenizer: TGemmaTokenizer;
    FContext: Integer;
    FChunk: Integer;
    FBos: Integer;
    FStopIDs: TArray<Integer>;
    FCuda: TCudaExecution;
    FBackend: string;
    FDevice: string;
    function GenerateInContext(const Prompt: string; const Options: TQwenGenerationOptions;
      const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): string;
  public
    constructor Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
    destructor Destroy; override;
    function Generate(const Prompt: string; const Options: TQwenGenerationOptions;
      const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): string;
    property ContextTokens: Integer read FContext;
  end;

implementation

uses
  {$IFDEF FPC}Windows, Math{$ELSE}Winapi.Windows, System.Math{$ENDIF}, TVM, TVM.CpuGraph,
  CrossPascal.Json;

procedure AppendTokens(var Tokens: TArray<Integer>; const Added: array of Integer);
var
  At: Integer;
begin
  if Length(Added) = 0 then Exit;
  At := Length(Tokens);
  SetLength(Tokens, At + Length(Added));
  Move(Added[0], Tokens[At], Length(Added) * SizeOf(Integer));
end;

constructor TGemma2Cpu.Create(const Folder, RuntimeRoot: string; const Progress: TQwenProgress);
const
  TokenizerFiles: array[0..6] of string = ('tokenizer-tokens.bin', 'tokenizer-tokens.index',
    'tokenizer-merges.bin', 'tokenizer-chars.bin', 'tokenizer-bytes.bin', 'tokenizer-specials.txt',
    'tokenizer.index');
var
  Data: TJSONValue;
  Model, Hashes: TJSONObject;
  EOS: TJSONArray;
  Error: TTVMCallError;
  I: Integer;
  NativeBundle: Boolean;
  Target, LibraryName, PrefillLibrary: string;
begin
  inherited Create;
  Data := ReadLanguageManifest(Folder);
  try
    Model := ModelObject(Data, 'Gemma 2 manifest');
    Target := ModelText(Model.GetValue('target'));
    LibraryName := ModelText(Model.GetValue('library'));
    RequireCpu((ModelText(Model.GetValue('model_type')) = 'gemma2') and
      ((Target = 'cpu') or (Target = 'cuda')) and
      (((Target = 'cpu') and (LibraryName = 'gemma2-cpu.dll')) or
        ((Target = 'cuda') and (LibraryName <> ''))), 'Expected a Gemma 2 TVM bundle');
    FDevice := UpperCase(Target);
    FBackend := 'TVM LLVM CPU FP32 (experimental Gemma 2)';
    FContext := ModelInt(Model.GetValue('context_window_size'), 'context window');
    FChunk := ModelInt(Model.GetValue('prefill_chunk_size'), 'prefill chunk');
    FBos := ModelInt(Model.GetValue('bos_token_id'), 'BOS token');
    RequireCpu((FContext > 1) and (FContext <= 131072) and (FChunk > 0) and (FChunk <= FContext),
      'Invalid Gemma 2 context');
    EOS := ModelArray(Model.GetValue('eos_token_id'), 'EOS tokens');
    SetLength(FStopIDs, EOS.Count);
    for I := 0 to High(FStopIDs) do FStopIDs[I] := ModelInt(EOS.Items[I], 'EOS token');
    NativeBundle := ModelText(Model.GetValue('schema')) = 'crosspascal.gemma2.text.v1';
    if Target = 'cuda' then
    begin
      RequireCpu(NativeBundle and (ModelText(Model.GetValue('backend')) = 'TVM CUDA W8A32'),
        'Invalid Gemma 2 CUDA bundle');
      FCuda := TCudaExecution.Create;
      FCuda.RequireArchitecture(ModelText(Model.GetValue('cuda_arch')));
      FBackend := 'TVM CUDA W8A32';
    end;
    if Assigned(Progress) then Progress('Verifying the Gemma 2 ' + FDevice + ' bundle...');
    QwenCuda.InitializeRuntime(IncludeTrailingPathDelimiter(RuntimeRoot) +
      'crosspascal-tvm\runtime\windows-x64-cuda\', FCuda <> nil);
    if NativeBundle then
    begin
      PrefillLibrary := ModelText(Model.GetValue('prefill_library'));
      if Target = 'cpu' then
        RequireCpu(PrefillLibrary = 'gemma2-prefill-cpu.dll', 'Invalid Gemma 2 CPU prefill library');
      if FCuda <> nil then FCuda.Enter;
      try
        FNative := TTokenGraph.Create(Folder, LibraryName, PrefillLibrary,
          Model, FContext, FChunk, TokenizerFiles, FCuda);
      finally
        if FCuda <> nil then FCuda.Leave;
      end;
    end
    else begin
      Hashes := TJSONObject.Create;
      try
        Hashes.AddPair('gemma2-cpu.dll', ModelText(Model.GetValue('library_sha256')));
        Hashes.AddPair('params.bin', ModelText(Model.GetValue('params_sha256')));
        VerifyArtifacts(Folder, Hashes);
      finally
        Hashes.Free;
      end;
    end;
  finally
    Data.Free;
  end;
  FTokenizer := TGemmaTokenizer.Create;
  RequireTVM(FTokenizer.Load(Folder, @Error), @Error);
  if FNative = nil then
    RequireTVM(FModel.Load(Folder, @Error, 'gemma2-cpu.dll'), @Error)
  else
    RequireCpu(FTokenizer.TokenCount = FNative.Vocabulary,
      'Gemma 2 tokenizer vocabulary differs from the graph');
  if Assigned(Progress) then Progress('Model loaded on ' + FDevice + ' through TVM');
end;

destructor TGemma2Cpu.Destroy;
begin
  FModel.Unload;
  try
    if (FCuda <> nil) and (FNative <> nil) then
    begin
      FCuda.Enter;
      try
        FNative.Free;
      finally
        FCuda.Leave;
      end;
    end
    else
      FNative.Free;
  finally
    FCuda.Free;
    FTokenizer.Free;
    inherited;
  end;
end;

function TGemma2Cpu.Generate(const Prompt: string; const Options: TQwenGenerationOptions;
  const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): string;
begin
  if FCuda <> nil then FCuda.Enter;
  try
    Result := GenerateInContext(Prompt, Options, Progress, OnText, ShouldStop);
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
end;

function TGemma2Cpu.GenerateInContext(const Prompt: string; const Options: TQwenGenerationOptions;
  const Progress, OnText: TQwenProgress; const ShouldStop: TQwenStopRequested): string;
const
  EffortBudget: array[TQwenAnswerEffort] of Integer = (0, 128, 512, 1024);
  EffortName: array[TQwenAnswerEffort] of string = ('full_context', 'brief', 'balanced', 'detailed');
var
  Tokens, Current, Output: TArray<Integer>;
  Seen: TArray<Boolean>;
  Logits: TArray<Single>;
  Embedded, Row: TTVMFFIObjectHandle;
  Caches: TTensorHandles;
  Text, Answer, Reason, Which: string;
  Error: TTVMCallError;
  StartTurn, EndTurn, I, Position, Limit, Token, StopID: Integer;
  Started, First: UInt64;
  State: Cardinal;
  EndOfSequence, Sampled: Boolean;
  Report, Settings: TJSONObject;
  IDs, PromptIDs, Modalities: TJSONArray;
begin
  ValidateQwenGenerationOptions(Options);
  Text := Prompt;
  if Options.SystemPrompt <> '' then Text := Options.SystemPrompt + #10#10 + Prompt;
  RequireCpu(Trim(Text) <> '', 'Enter a non-empty prompt');
  AppendTokens(Tokens, [FBos]);
  if Options.PlainCompletion then
    AppendTokens(Tokens, FTokenizer.Encode(Text))
  else begin
    StartTurn := FTokenizer.SpecialId('<start_of_turn>');
    EndTurn := FTokenizer.SpecialId('<end_of_turn>');
    RequireCpu((StartTurn >= 0) and (EndTurn >= 0), 'This tokenizer has no Gemma chat markers');
    AppendTokens(Tokens, [StartTurn]);
    AppendTokens(Tokens, FTokenizer.Encode('user' + #10 + Text));
    AppendTokens(Tokens, [EndTurn]);
    AppendTokens(Tokens, FTokenizer.Encode(#10));
    AppendTokens(Tokens, [StartTurn]);
    AppendTokens(Tokens, FTokenizer.Encode('model' + #10));
  end;
  RequireCpu(Length(Tokens) < FContext, 'Prompt does not fit the bundle context');
  Limit := FContext - Length(Tokens);
  Reason := 'context_limit';
  if (Options.MaxNewTokens > 0) and (Options.MaxNewTokens < Limit) then
  begin
    Limit := Options.MaxNewTokens;
    Reason := 'manual_limit';
  end
  else if (Options.MaxNewTokens = 0) and (EffortBudget[Options.Effort] > 0) and
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
    State := Cardinal(GetTickCount64 xor UInt64(GetCurrentThreadId));
  SetLength(Seen, FTokenizer.TokenCount);
  for I in Tokens do if (I >= 0) and (I < Length(Seen)) then Seen[I] := True;
  Started := GetTickCount64;
  First := 0;
  Position := 0;
  Answer := '';
  if Assigned(Progress) then Progress('Processing the prompt on ' + FDevice + '...');
  if FNative = nil then
    RequireTVM(FModel.CreateCache(1, FContext, FChunk, 16, @Error), @Error)
  else
    Caches := FNative.NewCache;
  try
    if FNative = nil then RequireTVM(FModel.AddSequence(0, @Error), @Error);
    Current := Copy(Tokens, 0, FChunk);
    while Length(Output) < Limit do
    begin
      if Assigned(ShouldStop) and ShouldStop() then
      begin
        Reason := 'user_stop';
        Break;
      end;
      if FNative <> nil then
      begin
        if Length(Current) <> FChunk then SetLength(Current, 1);
        try
          Logits := FNative.Evaluate(Current, Position, Caches, TCpuGraphStop(ShouldStop));
        except
          on ECpuGraphStopped do
          begin
            Reason := 'user_stop';
            Break;
          end;
        end;
      end
      else begin
        if Position < Length(Tokens) then
          Which := 'prefill'
        else
          Which := 'decode';
        RequireTVM(FModel.Embed(Current, Embedded, @Error), @Error);
        try
          RequireTVM(FModel.Run(Which, Embedded, 0, Length(Current), Row, @Error), @Error);
          try
            if Position + Length(Current) >= Length(Tokens) then Logits := ReadFloatTensor(Row);
          finally
            TTVMRuntime.ObjectRelease(Row);
          end;
        finally
          TTVMRuntime.ObjectRelease(Embedded);
        end;
      end;
      Inc(Position, Length(Current));
      if Position < Length(Tokens) then
      begin
        Current := Copy(Tokens, Position, FChunk);
        Continue;
      end;
      RequireCpu((Length(Logits) > 0) and (Length(Logits) = Length(Seen)), 'Invalid Gemma 2 logits');
      Token := 0;
      for I := 0 to High(Logits) do
      begin
        RequireCpu(not IsNan(Logits[I]) and not IsInfinite(Logits[I]), 'TVM returned non-finite logits');
        if Logits[I] > Logits[Token] then Token := I;
      end;
      if First = 0 then First := GetTickCount64;
      if Sampled then Token := SampleToken(Logits, Seen, Options, State);
      EndOfSequence := False;
      for StopID in FStopIDs do if StopID = Token then EndOfSequence := True;
      if EndOfSequence then
      begin
        Reason := 'eos';
        Break;
      end;
      Seen[Token] := True;
      Output := Output + [Token];
      Answer := FTokenizer.Decode(Output);
      if Assigned(OnText) then OnText(Answer);
      Current := [Token];
    end;
  finally
    if FNative = nil then
      TTVMRuntime.ObjectRelease(FModel.CacheHandle)
    else
      ReleaseTensors(Caches);
  end;
  Report := TJSONObject.Create;
  try
    Report.AddPair('text', Answer);
    Report.AddPair('stop', Reason);
    Report.AddPair('backend', FBackend);
    if FCuda <> nil then
    begin
      Report.AddPair('device_name', FCuda.DeviceName);
      Report.AddPair('compiled_arch', FCuda.Architecture);
    end;
    Report.AddPair('experimental', JSONBoolean(True));
    Report.AddPair('model_reused', JSONBoolean(True));
    Report.AddPair('prompt_tokens', JSONInteger(Length(Tokens)));
    Report.AddPair('generated_tokens', JSONInteger(Length(Output)));
    Report.AddPair('total_seconds', JSONFloat((GetTickCount64 - Started) / 1000.0));
    if First = 0 then
      Report.AddPair('first_token_seconds', TJSONNull.Create)
    else
      Report.AddPair('first_token_seconds', JSONFloat((First - Started) / 1000.0));
    IDs := TJSONArray.Create;
    Report.AddPair('token_ids', IDs);
    for Token in Output do IDs.AddElement(JSONInteger(Token));
    PromptIDs := TJSONArray.Create;
    Report.AddPair('prompt_token_ids', PromptIDs);
    for Token in Tokens do PromptIDs.AddElement(JSONInteger(Token));
    Modalities := TJSONArray.Create;
    Report.AddPair('modalities', Modalities);
    Modalities.AddElement(TJSONString.Create('text'));
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
    Settings.AddPair('effort', EffortName[Options.Effort]);
    Result := Report.ToJSON;
  finally
    Report.Free;
  end;
end;

end.
