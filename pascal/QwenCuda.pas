{ ************************************************************************** }
{                                                                            }
{ QwenCuda                                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit QwenCuda;

{$IFDEF FPC}
{$MODE DELPHIUNICODE}
{$ENDIF}

{$B-}
{$POINTERMATH ON}
{$IFNDEF WIN64}
  {$MESSAGE ERROR 'QwenCuda requires Windows 64-bit'}
{$ENDIF}
{$IFNDEF TVM_LAZY_INIT}
  {$MESSAGE ERROR 'Define TVM_LAZY_INIT in the project before compiling TVM units'}
{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF};

const
  QwenSourceRepository = 'Qwen/Qwen2.5-0.5B-Instruct';
  QwenSourceRevision = '7ae557604adf67be50417f59c2c2f167def9a775';
  QwenSourceTreeSHA256 = '90361a89884fd025cac6718a75d8ede78d2e259d2be5bef2e33d78f08f8585b1';
  QwenParameterBytes: Int64 = 1976131072;
  QwenContextSize = 64;
  QwenDefaultContextSize = 2048;
  QwenDefaultMaxNewTokens = 0;
  QwenMaxTopK = 128;

type
  EQwenCuda = class(Exception);
  EQwenCudaCancelled = class(EQwenCuda);
  TQwenProgress = {$IFDEF FPC}procedure(const MessageText: string) of object{$ELSE}reference to procedure(const MessageText: string){$ENDIF};
  TQwenStopRequested = function: Boolean of object;

  TQwenSamplingMode = (qsDefault, qsSample, qsGreedy);
  TQwenAnswerEffort = (qeFullContext, qeBrief, qeBalanced, qeDetailed);

  TQwenGenerationOptions = record
    MaxNewTokens: Int64;
    Sampling: TQwenSamplingMode;
    Effort: TQwenAnswerEffort;
    Temperature: Double;
    TopP: Double;
    TopK: Integer;
    RepetitionPenalty: Double;
    Seed: Int64;
    SystemPrompt: string;
    PlainCompletion: Boolean;
  end;

  TQwenCudaResult = record
    ProfileID: string;
    SourceRevision: string;
    SourceTreeSHA256: string;
    DeviceName: string;
    ComputeVersion: string;
    FreeDeviceBytes: Int64;
    TotalDeviceBytes: Int64;
    ModelSHA256: string;
    ParametersSHA256: string;
    ModelMetadataSHA256: string;
    Prompt: string;
    PromptTokenIDs: TArray<Integer>;
    GeneratedTokenIDs: TArray<Integer>;
    ContextTokens: Integer;
    AvailableNewTokens: Integer;
    GenerationOptions: TQwenGenerationOptions;
    Sampled: Boolean;
    SamplingSeed: Cardinal;
    ModelReused: Boolean;
    Answer: string;
    StopReason: string;
    ElapsedMilliseconds: UInt64;
  end;

  TQwenCudaSession = class
  private
    FState: TObject;
    function Execute(const BundleDirectory, RuntimeDirectory, Prompt: string;
      const Options: TQwenGenerationOptions; const Progress: TQwenProgress;
      const StopRequested: TQwenStopRequested; const TextUpdated: TQwenProgress;
      const LoadOnly: Boolean): TQwenCudaResult;
    function GetLoaded: Boolean;
  public
    destructor Destroy; override;
    function Load(const BundleDirectory, RuntimeDirectory: string;
      const Progress: TQwenProgress = nil): TQwenCudaResult;
    procedure Unload;
    property Loaded: Boolean read GetLoaded;
    function Run(const BundleDirectory, RuntimeDirectory, Prompt: string;
      const Options: TQwenGenerationOptions; const Progress: TQwenProgress = nil;
      const StopRequested: TQwenStopRequested = nil;
      const TextUpdated: TQwenProgress = nil): TQwenCudaResult;
  end;

function DefaultQwenGenerationOptions: TQwenGenerationOptions;
procedure ValidateQwenGenerationOptions(const Options: TQwenGenerationOptions);
procedure InitializeRuntime(const Directory: string; const NeedCuda: Boolean = True);
function SampleToken(const Logits: TArray<Single>; const Seen: TArray<Boolean>;
  const Options: TQwenGenerationOptions; var State: Cardinal): Integer;
function RunQwenCuda(const BundleDirectory, RuntimeDirectory, Prompt: string;
  const MaxNewTokens: Int64; const Progress: TQwenProgress = nil;
  const StopRequested: TQwenStopRequested = nil): TQwenCudaResult; overload;
function RunQwenCuda(const BundleDirectory, RuntimeDirectory, Prompt: string;
  const Options: TQwenGenerationOptions; const Progress: TQwenProgress = nil;
  const StopRequested: TQwenStopRequested = nil): TQwenCudaResult; overload;
function QwenResultJSON(const Value: TQwenCudaResult): string;

implementation

uses
  {$IFDEF FPC}Windows, Classes, Generics.Collections, Math, {$ELSE}
  Winapi.Windows, System.Classes, System.Generics.Collections, System.Math, {$ENDIF}
  {$IFDEF FPC}CrossPascal.WinMemory,{$ENDIF} CrossPascal.Json, CrossPascal.SHA256,
  DLPack, TVM, QwenModel, QwenTokenizer;

type
  TQwenCudaProfile = record
    ModelKey: string;
    ProfileID: string;
    SourceRepository: string;
    SourceRevision: string;
    SourceTreeSHA256: string;
    HiddenSize: Integer;
    IntermediateSize: Integer;
    LayerCount: Integer;
    AttentionHeads: Integer;
    KVHeads: Integer;
    HeadSize: Integer;
  end;

  TInputLease = class
  private
    FStreams: TObjectList<TFileStream>;
    FNames: TStringList;
  public
    constructor Create;
    destructor Destroy; override;
    function Hold(const Path: string; const ExpectedBytes: Int64 = -1;
      const ExpectedSHA256: string = ''): TFileStream;
  end;

  TQwenSessionState = class
    Bundle: string;
    Runtime: string;
    Profile: TQwenCudaProfile;
    Info: TQwenCudaResult;
    Lease: TInputLease;
    Model: TQwenModel;
    Tokenizer: TQwenTokenizer;
    destructor Destroy; override;
  end;

const
  QwenVocabSize = 151936;
  QwenBOS = 151643;
  QwenEOS = 151645;
  QwenProfiles: array[0..1] of TQwenCudaProfile = (
    (
      ModelKey: 'qwen2.5-0.5b';
      ProfileID: 'qwen2.5-0.5b-cuda-f32-64-v1';
      SourceRepository: QwenSourceRepository;
      SourceRevision: QwenSourceRevision;
      SourceTreeSHA256: QwenSourceTreeSHA256;
      HiddenSize: 896; IntermediateSize: 4864; LayerCount: 24;
      AttentionHeads: 14; KVHeads: 2; HeadSize: 64
    ),
    (
      ModelKey: 'qwen2.5-1.5b';
      ProfileID: 'qwen2.5-1.5b-cuda-f32-64-v1';
      SourceRepository: 'Qwen/Qwen2.5-1.5B-Instruct';
      SourceRevision: '989aa7980e4cf806f80c7fef2b1adb7bc71aa306';
      SourceTreeSHA256: '5f0180b3ad3b8652860444471fe9e2dd2e5ce325e5e5dccd2be7190ef437cf0d';
      HiddenSize: 1536; IntermediateSize: 8960; LayerCount: 28;
      AttentionHeads: 12; KVHeads: 2; HeadSize: 128
    )
  );
  RuntimeNames: array[0..3] of string = (
    'tvm_ffi.dll', 'tvm_runtime.dll', 'tvm_runtime_extra.dll', 'tvm_runtime_cuda.dll');
  RuntimeBytes: array[0..3] of Int64 = (4082176, 2112512, 828416, 365056);
  RuntimeSHA256: array[0..3] of string = (
    '2ed31dce9e331d23d6b4a2fa37e4166aa7843c334a3c17381f92dd3bcf122913',
    '45e3a9b5d76ea6950cde4442e5bd3208c66a8211d02936e2c0e727f52d2c4ac1',
    '4fe689aa56b1a46349def33dd75f5066f70e1a0ba5d156e2e74e8667d3a7beee',
    '4c67e81596267e94b153e855290fda1320ac0f4852faa36f8ac7184be585e87b');
  TokenizerNames: array[0..4] of string = (
    'tokenizer.index', 'tokenizer-merges.bin', 'tokenizer-specials.txt',
    'tokenizer-tokens.bin', 'tokenizer-tokens.index');
  TokenizerBytes: array[0..4] of Int64 = (187, 1816644, 473, 975969, 1213144);
  TokenizerSHA256: array[0..4] of string = (
    'af53370b9a99e25c51ed57e17b09cf7b91e54f624441733afc01a1ee54115c81',
    '774198a3f5cbc298c6446fd4e7133e55a3676ac95ef21fcb9e792cb31e6c4c31',
    'cf142f59d612002b0ba89a58bb1c12d9aa588afe57f0360e0d89f46d0f45262d',
    '97a0a8a1d844e38ef7ca9de4bda3a3388676f44d0321163d383454d049a507c4',
    '154f80686ba52e8dc9647bd274284fb59fc6c0530ec699c80e0d06e5859291f4');
  DLLSearchFlags = $00000100 or $00001000;
  MaxMetadataBytes = 1024 * 1024;
  QwenGenericProfileID1024 = 'qwen2-compatible-cuda-f32-1024-v1';
  QwenGenericProfileID = 'qwen2-compatible-cuda-f32-2048-v1';
  QwenGenericModelKey = 'qwen2-compatible-local';
  QwenGenericSchema = 'crosspascal.qwen2-compatible.v1';
  QwenGenericTokenizerSHA256 = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539';
  GenericMaxPackedBytes: Int64 = Int64(12) * 1024 * 1024 * 1024;

var
  Running: LongInt;
  LoadedRuntimeDirectory: string;
  RuntimeInitializationFailed: Boolean;

procedure Require(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then
    raise EQwenCuda.Create(MessageText);
end;

procedure RequireCall(const Condition: Boolean;
  const ErrorText: {$IFDEF FPC}PAnsiString{$ELSE}PString{$ENDIF});
begin
  if not Condition then
  begin
    if Assigned(ErrorText) and (ErrorText^ <> '') then
      raise EQwenCuda.Create(ErrorText^);
    raise EQwenCuda.Create('The operation failed without an error message.');
  end;
end;

procedure Notify(const Progress: TQwenProgress; const MessageText: string);
begin
  if Assigned(Progress) then
    Progress(MessageText);
end;

function DirectoryPath(const Value: string): string;
var
  Current, Parent: string;
  Attributes: DWORD;
begin
  Require(Trim(Value) <> '', 'Select a directory first.');
  Result := IncludeTrailingPathDelimiter(ExpandFileName(Value));
  Require(DirectoryExists(Result), 'Directory is unavailable: ' + Result);
  Current := ExcludeTrailingPathDelimiter(Result);
  repeat
    Attributes := GetFileAttributesW(PWideChar(Current));
    Require((Attributes <> INVALID_FILE_ATTRIBUTES) and
      ((Attributes and FILE_ATTRIBUTE_REPARSE_POINT) = 0),
      'Select a directory whose path does not contain a link or junction.');
    Parent := ExtractFileDir(Current);
    if SameText(Parent, Current) or (Parent = '') then
      Break;
    Current := Parent;
  until False;
end;

function IsSHA256(const Value: string): Boolean;
var
  C: Char;
begin
  Result := Length(Value) = 64;
  if Result then
    for C in Value do
      if not CharInSet(C, ['0'..'9', 'a'..'f']) then
        Exit(False);
end;

function StreamSHA256(const Stream: TStream): string;
begin
  Stream.Position := 0;
  Result := SHA256Stream(Stream);
  Stream.Position := 0;
end;

constructor TInputLease.Create;
begin
  inherited Create;
  FStreams := TObjectList<TFileStream>.Create(True);
  FNames := TStringList.Create;
  FNames.CaseSensitive := False;
end;

destructor TInputLease.Destroy;
begin
  FStreams.Free;
  FNames.Free;
  inherited Destroy;
end;

function TInputLease.Hold(const Path: string; const ExpectedBytes: Int64;
  const ExpectedSHA256: string): TFileStream;
var
  Attributes: DWORD;
  Index: Integer;
begin
  Index := FNames.IndexOf(Path);
  if Index >= 0 then
    Result := FStreams[Index]
  else begin
    Attributes := GetFileAttributesW(PWideChar(Path));
    Require((Attributes <> INVALID_FILE_ATTRIBUTES) and
      ((Attributes and (FILE_ATTRIBUTE_DIRECTORY or FILE_ATTRIBUTE_REPARSE_POINT)) = 0),
      'A required regular file is unavailable: ' + Path);
    Result := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
    FStreams.Add(Result);
    FNames.Add(Path);
  end;
  Require(Result.Size > 0, 'A required file is empty: ' + Path);
  if ExpectedBytes >= 0 then
    Require(Result.Size = ExpectedBytes, 'Unexpected file length: ' + Path);
  if ExpectedSHA256 <> '' then
  begin
    Require(IsSHA256(ExpectedSHA256), 'Invalid SHA-256 metadata: ' + Path);
    Require(StreamSHA256(Result) = ExpectedSHA256, 'SHA-256 mismatch: ' + Path);
  end;
end;

function ReadText(const Stream: TFileStream): string;
var
  Data: TBytes;
begin
  Require(Stream.Size <= MaxMetadataBytes, 'Metadata exceeds the size limit.');
  SetLength(Data, Integer(Stream.Size));
  Stream.Position := 0;
  Stream.ReadBuffer(Data[0], Length(Data));
  Result := TEncoding.UTF8.GetString(Data);
  if (Result <> '') and (Result[1] = #$FEFF) then
    Delete(Result, 1, 1);
end;

procedure CheckJSONKeys(const Value: TJSONValue; const Depth: Integer);
var
  Obj: TJSONObject;
  Arr: TJSONArray;
  Names: TStringList;
  Index: Integer;
  Name: string;
begin
  Require(Depth <= 16, 'JSON nesting exceeds the supported limit.');
  if Value is TJSONObject then
  begin
    Obj := TJSONObject(Value);
    Names := TStringList.Create;
    try
      Names.CaseSensitive := True;
      for Index := 0 to Obj.Count - 1 do
      begin
        Name := JSONObjectName(Obj, Index);
        Require(Names.IndexOf(Name) < 0, 'Duplicate JSON key: ' + Name);
        Names.Add(Name);
        CheckJSONKeys(JSONObjectValue(Obj, Index), Depth + 1);
      end;
    finally
      Names.Free;
    end;
  end
  else if Value is TJSONArray then
    begin
      Arr := TJSONArray(Value);
      for Index := 0 to Arr.Count - 1 do
        CheckJSONKeys(Arr.Items[Index], Depth + 1);
    end;
end;

function ReadObject(const Stream: TFileStream): TJSONObject;
var
  Value: TJSONValue;
begin
  Value := ParseCrossJSON(ReadText(Stream));
  try
    Require(Value is TJSONObject, 'Expected a JSON object.');
    CheckJSONKeys(Value, 0);
    Result := TJSONObject(Value);
    Value := nil;
  finally
    Value.Free;
  end;
end;

function Field(const Obj: TJSONObject; const Name: string): TJSONValue;
begin
  Result := Obj.GetValue(Name);
  Require(Assigned(Result), 'Missing JSON field: ' + Name);
end;

function TextField(const Obj: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require(Value is TJSONString, 'Expected a JSON string: ' + Name);
  Result := JSONValueText(Value);
end;

function IntegerValue(const Value: TJSONValue; const Name: string): Int64;
begin
  Require((Value is TJSONNumber) and TryStrToInt64(JSONValueText(Value), Result),
    'Expected an exact JSON integer: ' + Name);
end;

function IntegerField(const Obj: TJSONObject; const Name: string): Int64;
begin
  Result := IntegerValue(Field(Obj, Name), Name);
end;

function BoundedInteger(const Obj: TJSONObject; const Name: string;
  const Minimum, Maximum: Int64): Integer;
var
  Measured: Int64;
begin
  Measured := IntegerField(Obj, Name);
  Require((Measured >= Minimum) and (Measured <= Maximum),
    'Unsupported generic Qwen dimension: ' + Name);
  Result := Integer(Measured);
end;

function BooleanField(const Obj: TJSONObject; const Name: string): Boolean;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require(JSONIsBoolean(Value), 'Expected JSON boolean: ' + Name);
  Result := JSONIsTrue(Value);
end;

function FiniteRangeField(const Obj: TJSONObject; const Name: string;
  const Minimum, Maximum: Double): Double;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require((Value is TJSONNumber) and
    TryStrToFloat(JSONValueText(Value), Result, JSONInvariantFormat) and
    not IsNan(Result) and not IsInfinite(Result) and
    (Result >= Minimum) and (Result <= Maximum),
    'Unsupported generic Qwen numeric field: ' + Name);
end;

function ObjectField(const Obj: TJSONObject; const Name: string): TJSONObject;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require(Value is TJSONObject, 'Expected a JSON object: ' + Name);
  Result := TJSONObject(Value);
end;

function ArrayField(const Obj: TJSONObject; const Name: string): TJSONArray;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require(Value is TJSONArray, 'Expected a JSON array: ' + Name);
  Result := TJSONArray(Value);
end;

procedure MatchText(const Obj: TJSONObject; const Name, Expected: string);
begin
  Require(TextField(Obj, Name) = Expected, 'Unsupported bundle field: ' + Name);
end;

procedure MatchInteger(const Obj: TJSONObject; const Name: string; const Expected: Int64);
begin
  Require(IntegerField(Obj, Name) = Expected, 'Unsupported bundle field: ' + Name);
end;

function BundleContextSize(const ProfileID: string): Integer;
begin
  if ProfileID = QwenGenericProfileID then
    Exit(QwenDefaultContextSize);
  if ProfileID = QwenGenericProfileID1024 then
    Exit(1024);
  Result := QwenContextSize;
end;

function CheckParameters(const Model, Params: TJSONObject; const IndexText: string;
  const Profile: TQwenCudaProfile): Int64;
var
  Lines: TStringList;
  Order, Tensors, Shape: TJSONArray;
  Tensor: TJSONObject;
  Words: TQwenWords;
  Row, Layer, TensorCount, AttentionWidth: Integer;
  Offset, LargestTensor, CacheBytes, GPUBytes, HostBytes: Int64;
  Generic: Boolean;

  procedure Expect(const Name: string; const Dims: array of Int64);
  var
    NBytes: Int64;
    D: Integer;
  begin
    NBytes := SizeOf(Single);
    for D := 0 to High(Dims) do
    begin
      Require((Dims[D] > 0) and (NBytes <= High(Int64) div Dims[D]),
        'Qwen tensor byte size overflow: ' + Name);
      NBytes := NBytes * Dims[D];
    end;
    Require(Row < Order.Count, 'Parameter count is too small.');
    Require((Order.Items[Row] is TJSONString) and (JSONValueText(Order.Items[Row]) = Name),
      'Parameter order mismatch: ' + Name);
    Require(Tensors.Items[Row] is TJSONObject, 'Invalid parameter metadata row.');
    Tensor := TJSONObject(Tensors.Items[Row]);
    MatchText(Tensor, 'name', Name);
    MatchText(Tensor, 'dtype', 'float32');
    MatchInteger(Tensor, 'offset', Offset);
    MatchInteger(Tensor, 'nbytes', NBytes);
    Shape := ArrayField(Tensor, 'shape');
    Require(Shape.Count = Length(Dims), 'Parameter rank mismatch: ' + Name);
    for D := 0 to High(Dims) do
      Require(IntegerValue(Shape.Items[D], Name) = Dims[D], 'Parameter shape mismatch: ' + Name);
    Words := SplitWords(Trim(Lines[Row + 2]));
    Require(Length(Words) = 4 + Length(Dims), 'Invalid parameter index row: ' + Name);
    Require((Words[0] = Name) and (Words[1] = IntToStr(Offset)) and
      (Words[2] = IntToStr(NBytes)) and (Words[3] = IntToStr(Length(Dims))),
      'Parameter index mismatch: ' + Name);
    for D := 0 to High(Dims) do
      Require(Words[4 + D] = IntToStr(Dims[D]), 'Parameter index shape mismatch: ' + Name);
    Require(Offset <= High(Int64) - NBytes, 'Qwen parameter offsets overflow');
    Inc(Offset, NBytes);
    if NBytes > LargestTensor then
      LargestTensor := NBytes;
    Inc(Row);
  end;

begin
  Generic := BundleContextSize(Profile.ProfileID) <> QwenContextSize;
  MatchText(Params, 'dtype', 'float32');
  MatchText(Params, 'sha256', TextField(Model, 'params_sha256'));
  Order := ArrayField(Model, 'param_order');
  Tensors := ArrayField(Params, 'tensors');
  TensorCount := Profile.LayerCount * 7 + 2;
  if Generic and not BooleanField(Model, 'tie_word_embeddings') then
    Inc(TensorCount);
  AttentionWidth := (Profile.AttentionHeads + 2 * Profile.KVHeads) * Profile.HeadSize;
  Require((Order.Count = TensorCount) and (Tensors.Count = TensorCount),
    'The parameter tensor count differs from the selected Qwen profile.');
  Lines := TStringList.Create;
  try
    Lines.Text := IndexText;
    Require(Lines.Count = TensorCount + 2, 'Unexpected number of parameter-index lines.');
    Require((Length(Lines[0]) > 0) and (Lines[0][1] = '#'), 'Invalid parameter-index header.');
    Words := SplitWords(Trim(Lines[1]));
    Require((Length(Words) = 3) and (Words[0] = IntToStr(TensorCount)) and
      (Words[1] = 'float32') and (Words[2] = 'cuda'), 'The parameters must be float32 CUDA.');
    Row := 0;
    Offset := 0;
    LargestTensor := 0;
    Expect('model.embed_tokens.weight', [QwenVocabSize, Profile.HiddenSize]);
    for Layer := 0 to Profile.LayerCount - 1 do
    begin
      Expect(Format('model.layers.%d.self_attn.c_attn.weight', [Layer]),
        [AttentionWidth, Profile.HiddenSize]);
      Expect(Format('model.layers.%d.self_attn.c_attn.bias', [Layer]), [AttentionWidth]);
      Expect(Format('model.layers.%d.self_attn.o_proj.weight', [Layer]),
        [Profile.HiddenSize, Profile.AttentionHeads * Profile.HeadSize]);
      Expect(Format('model.layers.%d.mlp.gate_up_proj.weight', [Layer]),
        [2 * Profile.IntermediateSize, Profile.HiddenSize]);
      Expect(Format('model.layers.%d.mlp.down_proj.weight', [Layer]),
        [Profile.HiddenSize, Profile.IntermediateSize]);
      Expect(Format('model.layers.%d.input_layernorm.weight', [Layer]), [Profile.HiddenSize]);
      Expect(Format('model.layers.%d.post_attention_layernorm.weight', [Layer]), [Profile.HiddenSize]);
    end;
    Expect('model.norm.weight', [Profile.HiddenSize]);
    if Generic and not BooleanField(Model, 'tie_word_embeddings') then
      Expect('lm_head.weight', [QwenVocabSize, Profile.HiddenSize]);
    Require(Row = TensorCount, 'Unexpected parameter layout.');
    MatchInteger(Params, 'total_bytes', Offset);
    MatchInteger(Model, 'params_bytes', Offset);
    if Generic then
    begin
      Require(Offset <= GenericMaxPackedBytes,
        'Generic Qwen FP32 parameters exceed the 12 GiB contract');
      MatchInteger(ObjectField(Model, 'qwen_compatibility'), 'packed_parameter_bytes', Offset);
      MatchInteger(ObjectField(Model, 'qwen_compatibility'), 'tensor_count', TensorCount);
      CacheBytes := Int64(2) * Profile.LayerCount * Profile.KVHeads *
        Profile.HeadSize * BundleContextSize(Profile.ProfileID) * SizeOf(Single);
      GPUBytes := Offset + CacheBytes + Int64(512) * 1024 * 1024;
      HostBytes := Int64(3) * LargestTensor + Int64(1024) * 1024 * 1024;
      if HostBytes < Int64(2048) * 1024 * 1024 then
        HostBytes := Int64(2048) * 1024 * 1024;
      MatchInteger(ObjectField(Model, 'qwen_compatibility'), 'kv_cache_bytes', CacheBytes);
      MatchInteger(ObjectField(Model, 'qwen_compatibility'), 'min_free_gpu_bytes', GPUBytes);
      MatchInteger(ObjectField(Model, 'qwen_compatibility'), 'min_free_host_ram_bytes', HostBytes);
      MatchInteger(ObjectField(Model, 'qwen_compatibility'),
        'min_free_output_disk_bytes',
        Int64(2) * Offset + Int64(512) * 1024 * 1024);
    end;
    Result := Offset;
  finally
    Lines.Free;
  end;
end;

function CheckModel(const Model, Passport: TJSONObject): TQwenCudaProfile;
var
  Artifact, TextInfo, TokenizerInfo, Descriptor: TJSONObject;
  Number, NormEpsilon: Double;
  NumberValue: TJSONValue;
  ProfileID: string;
  ProfileIndex, ExpectedContext: Integer;
  Generic: Boolean;
begin
  ProfileID := TextField(Model, 'llm_profile_id');
  ExpectedContext := BundleContextSize(ProfileID);
  Generic := ExpectedContext <> QwenContextSize;
  ProfileIndex := Low(QwenProfiles);
  while ProfileIndex <= High(QwenProfiles) do
  begin
    if QwenProfiles[ProfileIndex].ProfileID = ProfileID then
      Break;
    Inc(ProfileIndex);
  end;
  Require(Generic or (ProfileIndex <= High(QwenProfiles)),
    'Unsupported Qwen profile: ' + ProfileID);
  if Generic then
  begin
    Result := Default(TQwenCudaProfile);
    Result.ModelKey := QwenGenericModelKey;
    Result.ProfileID := ProfileID;
    Descriptor := ObjectField(Model, 'qwen_compatibility');
    MatchText(Descriptor, 'schema', QwenGenericSchema);
    MatchText(Descriptor, 'model_key', QwenGenericModelKey);
    MatchText(Descriptor, 'source_kind', 'local-sha256');
    MatchText(Descriptor, 'operator_contract', 'dense-silu-gqa-rope-normal-v1');
    MatchText(Descriptor, 'tokenizer_contract', 'qwen2-byte-bpe-im-v1');
    MatchText(Descriptor, 'tokenizer_sha256', QwenGenericTokenizerSHA256);
    MatchText(Model, 'source_kind', 'local-sha256');
    Require(not Assigned(Model.GetValue('source_repo')) and
      not Assigned(Model.GetValue('source_revision')),
      'Local Qwen source must not claim an official curated revision.');
    Result.SourceTreeSHA256 := TextField(Model, 'source_tree_sha256');
    Require(IsSHA256(Result.SourceTreeSHA256),
      'Local Qwen source tree requires full lowercase SHA-256.');
    MatchText(Descriptor, 'source_tree_sha256', Result.SourceTreeSHA256);
    Result.HiddenSize := BoundedInteger(Descriptor, 'hidden_size', 128, 8192);
    Result.IntermediateSize := BoundedInteger(Descriptor, 'intermediate_size', 128, 32768);
    Result.LayerCount := BoundedInteger(Descriptor, 'num_hidden_layers', 1, 80);
    Result.AttentionHeads := BoundedInteger(Descriptor, 'num_attention_heads', 1, 128);
    Result.KVHeads := BoundedInteger(Descriptor, 'num_key_value_heads', 1, 128);
    Result.HeadSize := BoundedInteger(Descriptor, 'head_dim', 32, 256);
    Require((Result.HeadSize mod 2 = 0) and
      (Result.AttentionHeads mod Result.KVHeads = 0) and
      (Int64(Result.AttentionHeads) * Result.HeadSize = Result.HiddenSize),
      'Generic Qwen grouping or RoPE head dimension is unsupported.');
    MatchInteger(Descriptor, 'vocab_size', QwenVocabSize);
    MatchInteger(Descriptor, 'bos_token_id', QwenBOS);
    MatchInteger(Descriptor, 'eos_token_id', QwenEOS);
    MatchInteger(Descriptor, 'context_window_size', ExpectedContext);
    MatchInteger(Descriptor, 'prefill_chunk_size', ExpectedContext);
    MatchInteger(Descriptor, 'max_batch_size', 1);
    Require(BooleanField(Model, 'tie_word_embeddings') = BooleanField(Descriptor, 'tie_word_embeddings'),
      'Generic Qwen tied/untied descriptor differs from model.');
    Number := FiniteRangeField(Descriptor, 'rope_theta', 1000.0, 1000000000.0);
    Require(FiniteRangeField(Model, 'rope_theta', 1000.0, 1000000000.0) = Number,
      'Generic Qwen RoPE descriptor differs from model.');
    NormEpsilon := FiniteRangeField(Descriptor, 'rms_norm_eps', 1.0e-9, 1.0e-3);
    Require(FiniteRangeField(Model, 'rms_norm_eps', 1.0e-9, 1.0e-3) = NormEpsilon,
      'Generic Qwen RMSNorm descriptor differs from model.');
  end
  else
    Result := QwenProfiles[ProfileIndex];
  MatchText(Model, 'model_key', Result.ModelKey);
  MatchText(Model, 'library', 'qwen2-cuda.dll');
  MatchText(Model, 'target', 'cuda');
  Require(TextField(Model, 'arch') <> '', 'The model must declare a CUDA architecture.');
  MatchText(Model, 'dtype', 'float32');
  MatchText(Model, 'build_host_cpu_arch', 'x86_64');
  MatchText(Model, 'tokenizer', 'byte_bpe');
  if not Generic then
  begin
    MatchText(Model, 'source_repo', Result.SourceRepository);
    MatchText(Model, 'source_revision', Result.SourceRevision);
    MatchText(Model, 'source_tree_sha256', Result.SourceTreeSHA256);
  end;
  Require(IsSHA256(TextField(Model, 'build_plan_sha256')), 'Invalid BuildPlan SHA-256.');
  Require(IsSHA256(TextField(Model, 'params_sha256')), 'Invalid parameter SHA-256.');
  MatchInteger(Model, 'context_window_size', ExpectedContext);
  MatchInteger(Model, 'prefill_chunk_size', ExpectedContext);
  MatchInteger(Model, 'max_batch_size', 1);
  MatchInteger(Model, 'hidden_size', Result.HiddenSize);
  if Generic or (ProfileIndex <> Low(QwenProfiles)) or
    Assigned(Model.GetValue('intermediate_size')) then
      MatchInteger(Model, 'intermediate_size', Result.IntermediateSize);
  MatchInteger(Model, 'num_hidden_layers', Result.LayerCount);
  MatchInteger(Model, 'num_attention_heads', Result.AttentionHeads);
  MatchInteger(Model, 'num_key_value_heads', Result.KVHeads);
  MatchInteger(Model, 'head_dim', Result.HeadSize);
  MatchInteger(Model, 'vocab_size', QwenVocabSize);
  MatchInteger(Model, 'bos_token_id', QwenBOS);
  MatchInteger(Model, 'eos_token_id', QwenEOS);
  if not Generic then
  begin
    Require(JSONIsTrue(Field(Model, 'tie_word_embeddings')), 'Tied embeddings are required.');
    NumberValue := Field(Model, 'rope_theta');
    Require((NumberValue is TJSONNumber) and
      TryStrToFloat(JSONValueText(NumberValue), Number, JSONInvariantFormat) and
      (Number = 1000000.0), 'Unexpected RoPE theta.');
  end;
  MatchText(Passport, 'task_kind', 'text_generation');
  MatchText(Passport, 'adapter', 'qwen2');
  Artifact := ObjectField(Passport, 'artifact');
  MatchText(Artifact, 'file', 'qwen2-cuda.dll');
  MatchText(Artifact, 'os', 'windows');
  MatchText(Artifact, 'arch', 'x86_64');
  MatchText(Artifact, 'device', 'cuda');
  MatchText(Artifact, 'device_arch', TextField(Model, 'arch'));
  MatchText(Artifact, 'dtype', 'float32');
  MatchText(Artifact, 'format', 'dll');
  Require(IsSHA256(TextField(Artifact, 'hash')), 'Invalid model-library SHA-256.');
  TextInfo := ObjectField(Passport, 'text');
  MatchInteger(TextInfo, 'context_window', ExpectedContext);
  MatchInteger(TextInfo, 'bos', QwenBOS);
  MatchInteger(TextInfo, 'eos', QwenEOS);
  TokenizerInfo := ObjectField(Passport, 'tokenizer');
  MatchText(TokenizerInfo, 'style', 'byte_bpe');
end;

function ModulePath(const Handle: HMODULE): string;
var
  Buffer: array[0..32767] of Char;
  Count: DWORD;
begin
  Count := GetModuleFileNameW(Handle, @Buffer[0], Length(Buffer));
  Require((Count > 0) and (Count < DWORD(Length(Buffer))), 'Cannot identify a loaded runtime DLL.');
  SetString(Result, Buffer, Count);
  Result := ExpandFileName(Result);
end;

procedure InitializeRuntime(const Directory: string; const NeedCuda: Boolean);
var
  Handles: array[0..3] of HMODULE;
  I, Last: Integer;
  Version: TTVMFFIVersion;

  function LoadRuntimeFile(const FileName: string): HMODULE;
  begin
    Require(GetModuleHandleW(PWideChar(FileName)) = 0,
      'A TVM DLL is already loaded outside the explicit runtime loader.');
    Result := LoadLibraryExW(PWideChar(Directory + FileName), 0, DLLSearchFlags);
    Require(Result <> 0, Format('Cannot load %s: %s', [FileName, SysErrorMessage(GetLastError)]));
    if not SameText(ModulePath(Result), Directory + FileName) then
    begin
      FreeLibrary(Result);
      raise EQwenCuda.Create('A TVM DLL resolved outside the selected runtime directory.');
    end;
  end;

begin
  Require(not RuntimeInitializationFailed, 'Runtime initialization failed earlier. Restart the application.');
  if LoadedRuntimeDirectory <> '' then
  begin
    Require(SameText(LoadedRuntimeDirectory, Directory),
      'Restart the application before selecting a different runtime.');
    if NeedCuda and (TTVMRuntime.TModule.CudaRuntimeHandle = 0) then
      TTVMRuntime.TModule.CudaRuntimeHandle := LoadRuntimeFile(RuntimeNames[3]);
    Exit;
  end;
  Require((TTVMRuntime.TModule.Handle = 0) and (TTVMRuntime.TModule.RuntimeHandle = 0),
    'TVM was initialized before the selected runtime was checked.');
  RuntimeInitializationFailed := True;
  FillChar(Handles, SizeOf(Handles), 0);
  Last := 2;
  if NeedCuda then Last := 3;
  try
    for I := 0 to Last do Handles[I] := LoadRuntimeFile(RuntimeNames[I]);
  except
    for I := Last downto 0 do if Handles[I] <> 0 then FreeLibrary(Handles[I]);
    raise;
  end;
  TTVMRuntime.TModule.Handle := Handles[0];
  TTVMRuntime.TModule.RuntimeHandle := Handles[1];
  TTVMRuntime.TModule.ExtraRuntimeHandle := Handles[2];
  TTVMRuntime.TModule.CudaRuntimeHandle := Handles[3];
  TTVMRuntime.TModule.LibraryPath := Directory;
  TTVMRuntime.TModule.LibraryName := Directory + RuntimeNames[1];
  TTVMRuntime.TModule.Initialize(Handles[0], True);
  TTVMBackend.TModule.Handle := Handles[1];
  TTVMBackend.TModule.Initialize(Handles[1], True);
  TTVMRuntime.TModule.ResolveGlobals;
  Require(Assigned(TTVMRuntime.TModule.TVMFFIGetVersion), 'Missing TVM FFI version function.');
  TTVMRuntime.TModule.TVMFFIGetVersion(Version);
  Require((Version.Major = 0) and (Version.Minor = 1) and (Version.Patch = 14),
    Format('Unexpected TVM FFI ABI %d.%d.%d; expected 0.1.14.',
      [Version.Major, Version.Minor, Version.Patch]));
  LoadedRuntimeDirectory := Directory;
  RuntimeInitializationFailed := False;
end;

function DeviceAttribute(const Attribute: Integer): TTVMFFIAny;
var
  ErrorText: {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
begin
  RequireCall(TTVMRuntime.CallGlobal('runtime.GetDeviceAttr',
    [TTVMRuntime.AnyInt(Ord(kDLCUDA)), TTVMRuntime.AnyInt(0),
     TTVMRuntime.AnyInt(Attribute)], Result, @ErrorText), @ErrorText);
end;

function DeviceInteger(const Attribute: Integer): Int64;
var
  Value: TTVMFFIAny;
begin
  Value := DeviceAttribute(Attribute);
  Require(Value.TypeIndex in [kTVMFFIInt, kTVMFFIBool], 'Unexpected CUDA device-attribute type.');
  Result := Value.VInt64;
end;

function DeviceText(const Attribute: Integer): string;
var
  Value: TTVMFFIAny;
  Bytes: PTVMFFIByteArray;
  TextValue: UTF8String;
begin
  Value := DeviceAttribute(Attribute);
  try
    if Value.TypeIndex = kTVMFFISmallStr then
    begin
      Require(Value.Padding <= 7, 'Invalid short FFI string.');
      SetString(TextValue, PAnsiChar(@Value.VBytes[0]), Integer(Value.Padding));
    end
    else begin
      Require((Value.TypeIndex = kTVMFFIStr) and Assigned(Value.VObj),
        'Unexpected CUDA string-attribute type.');
      Bytes := PTVMFFIByteArray(NativeUInt(Value.VObj) + TVMFFIObjectPayloadOffset);
      Require(Bytes.Size <= 4096, 'CUDA device description is too long.');
      SetString(TextValue, Bytes.Data, Integer(Bytes.Size));
    end;
    Result := UTF8ToString(TextValue);
  finally
    if Value.TypeIndex >= kTVMFFIStaticObjectBegin then
      TTVMRuntime.ObjectRelease(Value.VObj);
  end;
end;

procedure CheckTensor(const Handle: TTVMFFIObjectHandle; const Shape: array of Int64);
var
  Tensor: PDLTensor;
  I: Integer;
begin
  Require(Assigned(Handle), 'TVM returned a null tensor.');
  Tensor := TTVMRuntime.TensorDLTensor(Handle);
  Require(Assigned(Tensor) and (Tensor.ndim = Length(Shape)) and
    (Tensor.device.device_type = kDLCUDA) and (Tensor.device.device_id = 0) and
    (Tensor.dtype.code = Ord(kDLFloat)) and (Tensor.dtype.bits = 32) and
    (Tensor.dtype.lanes = 1) and Assigned(Tensor.shape), 'Unexpected CUDA tensor type or rank.');
  for I := 0 to High(Shape) do
    Require(Tensor.shape[I] = Shape[I], 'Unexpected CUDA tensor shape.');
end;

function DefaultQwenGenerationOptions: TQwenGenerationOptions;
begin
  Result.MaxNewTokens := QwenDefaultMaxNewTokens;
  Result.Sampling := qsDefault;
  Result.Effort := qeFullContext;
  Result.Temperature := 0.7;
  Result.TopP := 0.8;
  Result.TopK := 20;
  Result.RepetitionPenalty := 1.1;
  Result.Seed := -1;
  Result.SystemPrompt := 'You are a helpful assistant.';
  Result.PlainCompletion := False;
end;

procedure ValidateQwenGenerationOptions(const Options: TQwenGenerationOptions);
begin
  Require(Options.MaxNewTokens >= 0, 'Max new tokens must be 0 (Auto) or a positive whole number.');
  Require(not IsNan(Options.Temperature) and (Options.Temperature >= 0.01) and
    (Options.Temperature <= 2), 'Temperature must be between 0.01 and 2.');
  Require(not IsNan(Options.TopP) and (Options.TopP > 0) and (Options.TopP <= 1),
    'Top-P must be greater than 0 and at most 1.');
  Require((Options.TopK >= 1) and (Options.TopK <= QwenMaxTopK),
    Format('Top-K must be a whole number between 1 and %d.', [QwenMaxTopK]));
  Require(not IsNan(Options.RepetitionPenalty) and (Options.RepetitionPenalty >= 1) and
    (Options.RepetitionPenalty <= 2), 'Repetition penalty must be between 1 and 2.');
  Require((Options.Seed >= -1) and (Options.Seed <= Int64(High(Cardinal))),
    'Seed must be blank (random) or a whole number between 0 and 4294967295.');
end;

function SampleToken(const Logits: TArray<Single>; const Seen: TArray<Boolean>;
  const Options: TQwenGenerationOptions; var State: Cardinal): Integer;
var
  Scores, Weights: array[0..QwenMaxTopK - 1] of Double;
  Tokens: array[0..QwenMaxTopK - 1] of Integer;
  I, J, Count, Kept, TopK: Integer;
  Score, Total, Mass, Draw: Double;
begin
  TopK := Options.TopK;
  Count := 0;
  for I := 0 to High(Logits) do
  begin
    Score := Logits[I];
    if Seen[I] then
    begin
      if Score < 0 then
        Score := Score * Options.RepetitionPenalty
      else
        Score := Score / Options.RepetitionPenalty;
    end;
    if (Count = TopK) and (Score <= Scores[TopK - 1]) then
      Continue;
    if Count < TopK then
      Inc(Count);
    J := Count - 1;
    while (J > 0) and (Score > Scores[J - 1]) do
    begin
      Scores[J] := Scores[J - 1];
      Tokens[J] := Tokens[J - 1];
      Dec(J);
    end;
    Scores[J] := Score;
    Tokens[J] := I;
  end;
  Total := 0;
  for I := 0 to Count - 1 do
  begin
    Weights[I] := Exp((Scores[I] - Scores[0]) / Options.Temperature);
    Total := Total + Weights[I];
  end;
  Mass := 0;
  Kept := 0;
  repeat
    Mass := Mass + Weights[Kept];
    Inc(Kept);
  until (Kept = Count) or (Mass >= Options.TopP * Total);
  State := Cardinal((UInt64(State) * 1664525 + 1013904223) and $FFFFFFFF);
  Draw := (State / 4294967296.0) * Mass;
  for I := 0 to Kept - 1 do
  begin
    Draw := Draw - Weights[I];
    if Draw < 0 then
      Exit(Tokens[I]);
  end;
  Result := Tokens[Kept - 1];
end;

function RunQwenCuda(const BundleDirectory, RuntimeDirectory, Prompt: string;
  const MaxNewTokens: Int64; const Progress: TQwenProgress;
  const StopRequested: TQwenStopRequested): TQwenCudaResult;
var
  Options: TQwenGenerationOptions;
begin
  Options := DefaultQwenGenerationOptions;
  Options.MaxNewTokens := MaxNewTokens;
  Result := RunQwenCuda(BundleDirectory, RuntimeDirectory, Prompt, Options, Progress, StopRequested);
end;

function RunQwenCuda(const BundleDirectory, RuntimeDirectory, Prompt: string;
  const Options: TQwenGenerationOptions; const Progress: TQwenProgress;
  const StopRequested: TQwenStopRequested): TQwenCudaResult;
var
  Session: TQwenCudaSession;
begin
  Session := TQwenCudaSession.Create;
  try
    Result := Session.Run(BundleDirectory, RuntimeDirectory, Prompt, Options, Progress,
      StopRequested);
  finally
    Session.Free;
  end;
end;

destructor TQwenSessionState.Destroy;
begin
  Model.Unload;
  Tokenizer.Free;
  Lease.Free;
  inherited Destroy;
end;

destructor TQwenCudaSession.Destroy;
begin
  FState.Free;
  inherited Destroy;
end;

function TQwenCudaSession.GetLoaded: Boolean;
begin
  Result := FState <> nil;
end;

procedure TQwenCudaSession.Unload;
begin
  Require(InterlockedCompareExchange(Running, 1, 0) = 0,
    'A Qwen operation is active. Stop it before unloading.');
  try
    FreeAndNil(FState);
  finally
    InterlockedExchange(Running, 0);
  end;
end;

function TQwenCudaSession.Load(const BundleDirectory, RuntimeDirectory: string;
  const Progress: TQwenProgress): TQwenCudaResult;
begin
  Result := Execute(BundleDirectory, RuntimeDirectory, '', DefaultQwenGenerationOptions,
    Progress, nil, nil, True);
end;

function TQwenCudaSession.Run(const BundleDirectory, RuntimeDirectory, Prompt: string;
  const Options: TQwenGenerationOptions; const Progress: TQwenProgress;
  const StopRequested: TQwenStopRequested; const TextUpdated: TQwenProgress): TQwenCudaResult;
begin
  Result := Execute(BundleDirectory, RuntimeDirectory, Prompt, Options, Progress,
    StopRequested, TextUpdated, False);
end;

function TQwenCudaSession.Execute(const BundleDirectory, RuntimeDirectory, Prompt: string;
  const Options: TQwenGenerationOptions; const Progress: TQwenProgress;
  const StopRequested: TQwenStopRequested; const TextUpdated: TQwenProgress;
  const LoadOnly: Boolean): TQwenCudaResult;
const
  EffortBudget: array[TQwenAnswerEffort] of Integer = (0, 128, 512, 1024);
var
  Bundle, Runtime, ChatText, ModelArch, DeviceArch: string;
  ErrorText: {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
  State: TQwenSessionState;
  ModelInfo, Passport, ParamsInfo: TJSONObject;
  ModelStream, IndexStream, ParamsStream: TFileStream;
  Profile: TQwenCudaProfile;
  ParameterBytes, CacheBytes, WorkspaceBytes: Int64;
  EmbedTensor, LogitsTensor: TTVMFFIObjectHandle;
  Values: TArray<Single>;
  StepTokens: TArray<Integer>;
  Seen: TArray<Boolean>;
  I, Step, TokenID, GenerationLimit, AutoBudget: Integer;
  SampleState: Cardinal;
  Started: UInt64;
  Memory: TMemoryStatusEx;
  SequenceAdded, Generic, Reused, Succeeded: Boolean;

  function Stopping: Boolean;
  begin
    Result := Assigned(StopRequested) and StopRequested();
  end;

  procedure CheckStop;
  begin
    if Stopping then
      raise EQwenCudaCancelled.Create('Stopped before generation.');
  end;

begin
  Require(InterlockedCompareExchange(Running, 1, 0) = 0, 'A Qwen run is already active.');
  Started := GetTickCount64;
  State := nil;
  ModelInfo := nil;
  Passport := nil;
  ParamsInfo := nil;
  EmbedTensor := nil;
  LogitsTensor := nil;
  SequenceAdded := False;
  Succeeded := False;
  Result := Default(TQwenCudaResult);
  Result.GenerationOptions := Options;
  try
    CheckStop;
    ValidateQwenGenerationOptions(Options);
    if not LoadOnly then
      Require(Trim(Prompt) <> '', 'Enter a nonempty prompt.');
    Bundle := DirectoryPath(BundleDirectory);
    Runtime := DirectoryPath(RuntimeDirectory);
    State := TQwenSessionState(FState);
    Reused := Assigned(State) and SameText(State.Bundle, Bundle) and
      SameText(State.Runtime, Runtime);
    if Reused then
    begin
      Notify(Progress, 'Reusing the verified model already loaded on CUDA.');
      Profile := State.Profile;
      Result := State.Info;
    end
    else begin
      FreeAndNil(FState);
      State := TQwenSessionState.Create;
      FState := State;
      State.Bundle := Bundle;
      State.Runtime := Runtime;
      State.Lease := TInputLease.Create;
      Notify(Progress, 'Checking the selected model profile and parameter layout...');
      ModelStream := State.Lease.Hold(Bundle + 'model.json');
      ModelInfo := ReadObject(ModelStream);
      Passport := ReadObject(State.Lease.Hold(Bundle + 'manifest.json'));
      ParamsInfo := ReadObject(State.Lease.Hold(Bundle + 'params.json'));
      IndexStream := State.Lease.Hold(Bundle + 'params.index');
      Profile := CheckModel(ModelInfo, Passport);
      State.Profile := Profile;
      Generic := BundleContextSize(Profile.ProfileID) <> QwenContextSize;
      ParameterBytes := CheckParameters(ModelInfo, ParamsInfo, ReadText(IndexStream), Profile);
      ParamsStream := State.Lease.Hold(Bundle + 'params.bin', ParameterBytes);
      Result.ProfileID := Profile.ProfileID;
      ModelArch := TextField(ModelInfo, 'arch');
      Result.SourceRevision := Profile.SourceRevision;
      Result.SourceTreeSHA256 := Profile.SourceTreeSHA256;
      Result.ModelMetadataSHA256 := StreamSHA256(ModelStream);
      Result.ModelSHA256 := TextField(ObjectField(Passport, 'artifact'), 'hash');
      State.Lease.Hold(Bundle + 'qwen2-cuda.dll', -1, Result.ModelSHA256);
      for I := 0 to High(TokenizerNames) do
        State.Lease.Hold(Bundle + TokenizerNames[I], TokenizerBytes[I], TokenizerSHA256[I]);
      for I := 0 to High(RuntimeNames) do
        State.Lease.Hold(Runtime + RuntimeNames[I], RuntimeBytes[I], RuntimeSHA256[I]);
      State.Tokenizer := TQwenTokenizer.Create;
      RequireCall(State.Tokenizer.Load(Bundle, @ErrorText), @ErrorText);
      Require((State.Tokenizer.TokenCount = QwenBOS) and
        (State.Tokenizer.SpecialId('<|im_end|>') = QwenEOS) and
        (State.Tokenizer.SpecialId('<|endoftext|>') = QwenBOS), 'Unexpected tokenizer vocabulary.');
    end;
    Result.GenerationOptions := Options;
    Result.ModelReused := Reused;
    Result.Prompt := Prompt;
    if Options.PlainCompletion then
      ChatText := Prompt
    else begin
      ChatText := '';
      if Options.SystemPrompt <> '' then
        ChatText := '<|im_start|>system'#10 + Options.SystemPrompt + '<|im_end|>'#10;
      ChatText := ChatText + '<|im_start|>user'#10 + Prompt +
        '<|im_end|>'#10'<|im_start|>assistant'#10;
    end;
    Result.PromptTokenIDs := State.Tokenizer.Encode(ChatText);
    Result.ContextTokens := QwenDefaultContextSize;
    Result.AvailableNewTokens := Result.ContextTokens - Length(Result.PromptTokenIDs);
    Require(
      LoadOnly or ((Length(Result.PromptTokenIDs) > 0) and (Result.AvailableNewTokens > 0)),
      Format(
        'The prompt uses %d tokens with the template. It must leave room in the %d-token context.',
        [
          Length(Result.PromptTokenIDs),
          Result.ContextTokens
        ]
      )
    );
    GenerationLimit := Result.AvailableNewTokens;
    Result.StopReason := 'context_limit';
    if (Options.MaxNewTokens > 0) and (Options.MaxNewTokens < GenerationLimit) then
    begin
      GenerationLimit := Integer(Options.MaxNewTokens);
      Result.StopReason := 'max_new_tokens';
    end;
    AutoBudget := EffortBudget[Options.Effort];
    if (Options.MaxNewTokens = 0) and (AutoBudget > 0) and (AutoBudget < GenerationLimit) then
    begin
      GenerationLimit := AutoBudget;
      Result.StopReason := 'effort_limit';
    end;
    CheckStop;
    for TokenID in Result.PromptTokenIDs do
      Require((TokenID >= 0) and (TokenID < QwenVocabSize),
        'A prompt token is outside the model vocabulary.');
    Result.Sampled := (Options.Sampling = qsSample) or ((Options.Sampling = qsDefault) and
      (Options.MaxNewTokens = 0));
    if Result.Sampled then
    begin
      if Options.Seed < 0 then
        SampleState := Cardinal(GetTickCount64 and $FFFFFFFF)
      else
        SampleState := Cardinal(Options.Seed);
      Result.SamplingSeed := SampleState;
      SetLength(Seen, QwenVocabSize);
      for TokenID in Result.PromptTokenIDs do
        Seen[TokenID] := True;
    end;
    if not Reused then
    begin
      Notify(Progress, 'Checking parameter SHA-256 before loading the model...');
      CheckStop;
      Result.ParametersSHA256 := StreamSHA256(ParamsStream);
      Require(Result.ParametersSHA256 = TextField(ModelInfo, 'params_sha256'),
        'Parameter SHA-256 differs from the converter output.');
      CheckStop;
      Notify(Progress, 'Loading the selected CUDA runtime and checking GPU 0...');
      InitializeRuntime(Runtime);
      Require(DeviceInteger(0) <> 0, 'CUDA device 0 is unavailable. No CPU fallback is enabled.');
      Result.ComputeVersion := DeviceText(4);
      Result.DeviceName := DeviceText(5);
      Result.TotalDeviceBytes := DeviceInteger(14);
      Result.FreeDeviceBytes := DeviceInteger(15);
      DeviceArch := 'sm_' + StringReplace(Result.ComputeVersion, '.', '', []);
      Require(
        SameText(
          ModelArch,
          DeviceArch
        ),
        'The bundle targets ' + ModelArch + ', but CUDA device 0 reports ' + DeviceArch + '. Convert the model for this device in the showcase.'
      );
      WorkspaceBytes := Int64(256) * 1024 * 1024;
      if Generic then
        WorkspaceBytes := 2 * WorkspaceBytes;
      CacheBytes := Int64(2) * Profile.LayerCount * Profile.KVHeads *
        Profile.HeadSize * Result.ContextTokens * SizeOf(Single);
      Require(Result.FreeDeviceBytes >= ParameterBytes + CacheBytes + WorkspaceBytes,
        'Not enough free GPU memory for the parameters, context cache and workspace reserve.');
      FillChar(Memory, SizeOf(Memory), 0);
      Memory.dwLength := SizeOf(Memory);
      Require(GlobalMemoryStatusEx(Memory), 'Cannot query available system memory.');
      if Generic then
        Require(Memory.ullAvailPhys >= UInt64(IntegerField(
          ObjectField(ModelInfo, 'qwen_compatibility'), 'min_free_host_ram_bytes')),
          'Insufficient free host RAM for the derived generic Qwen topology.')
      else
        Require(Memory.ullAvailPhys >= UInt64(1024) * 1024 * 1024,
          'At least 1 GiB of free system memory is required before model loading.');
      Notify(Progress, Format('Loading all %d layers on %s...', [Profile.LayerCount, Result.DeviceName]));
      CheckStop;
      RequireCall(State.Model.Load(Bundle, @ErrorText), @ErrorText);
      CheckStop;
      Require((State.Model.DeviceType = Ord(kDLCUDA)) and (State.Model.DataBits = 32) and
        (Length(State.Model.Entries) = ArrayField(ModelInfo, 'param_order').Count),
        'The loaded model is not the checked CUDA profile.');
      RequireCall(State.Model.CreateCache(1, Result.ContextTokens, Result.ContextTokens, 16, @ErrorText), @ErrorText);
      State.Info := Result;
      State.Info.Prompt := '';
      State.Info.PromptTokenIDs := nil;
      State.Info.Sampled := False;
      State.Info.SamplingSeed := 0;
    end;
    if LoadOnly then
    begin
      Result.ElapsedMilliseconds := GetTickCount64 - Started;
      Succeeded := True;
      Exit;
    end;
    RequireCall(State.Model.AddSequence(0, @ErrorText), @ErrorText);
    SequenceAdded := True;
    StepTokens := Copy(Result.PromptTokenIDs);
    for Step := 0 to GenerationLimit - 1 do
    begin
      Notify(Progress, Format('Generating token %d of %d on CUDA (%d available after the prompt).',
        [Step + 1, GenerationLimit, Result.AvailableNewTokens]));
      if Stopping then
      begin
        Result.StopReason := 'cancelled';
        Break;
      end;
      try
        RequireCall(State.Model.Embed(StepTokens, EmbedTensor, @ErrorText), @ErrorText);
        CheckTensor(EmbedTensor, [1, Length(StepTokens), Profile.HiddenSize]);
        if Step = 0 then
          RequireCall(State.Model.Run('prefill', EmbedTensor, 0, Length(StepTokens), LogitsTensor, @ErrorText), @ErrorText)
        else
          RequireCall(State.Model.Run('decode', EmbedTensor, 0, 1, LogitsTensor, @ErrorText), @ErrorText);
        CheckTensor(LogitsTensor, [1, 1, QwenVocabSize]);
        RequireCall(State.Model.ReadFloats(LogitsTensor, Values, @ErrorText), @ErrorText);
        Require(Length(Values) = QwenVocabSize, 'Unexpected number of output logits.');
        for I := 0 to High(Values) do
          Require(not IsNan(Values[I]) and not IsInfinite(Values[I]), 'The model returned nonfinite logits.');
        if Result.Sampled then
        begin
          TokenID := SampleToken(Values, Seen, Options, SampleState);
          Seen[TokenID] := True;
        end
        else
          TokenID := Integer(ArgMaxOf(@Values[0], Length(Values)));
        SetLength(Result.GeneratedTokenIDs, Length(Result.GeneratedTokenIDs) + 1);
        Result.GeneratedTokenIDs[High(Result.GeneratedTokenIDs)] := TokenID;
        if (TokenID = QwenEOS) or (TokenID = QwenBOS) then
        begin
          Result.StopReason := 'eos';
          Break;
        end;
        if Assigned(TextUpdated) then
          TextUpdated(State.Tokenizer.DecodePrefix(Result.GeneratedTokenIDs));
        StepTokens := TArray<Integer>.Create(TokenID);
      finally
        TTVMRuntime.ObjectRelease(LogitsTensor);
        TTVMRuntime.ObjectRelease(EmbedTensor);
      end;
    end;
    StepTokens := Copy(Result.GeneratedTokenIDs);
    if (Length(StepTokens) > 0) and
      ((StepTokens[High(StepTokens)] = QwenEOS) or (StepTokens[High(StepTokens)] = QwenBOS)) then
        SetLength(StepTokens, Length(StepTokens) - 1);
    Result.Answer := State.Tokenizer.Decode(StepTokens);
    RequireCall(State.Model.RemoveSequence(0, @ErrorText), @ErrorText);
    SequenceAdded := False;
    Result.ElapsedMilliseconds := GetTickCount64 - Started;
    Notify(Progress, 'Generation finished.');
    Succeeded := True;
  finally
    TTVMRuntime.ObjectRelease(LogitsTensor);
    TTVMRuntime.ObjectRelease(EmbedTensor);
    if SequenceAdded then
      State.Model.RemoveSequence(0, @ErrorText);
    ParamsInfo.Free;
    Passport.Free;
    ModelInfo.Free;
    try
      if not Succeeded then
        FreeAndNil(FState);
    finally
      InterlockedExchange(Running, 0);
    end;
  end;
end;

function QwenResultJSON(const Value: TQwenCudaResult): string;
const
  EffortName: array[TQwenAnswerEffort] of string = ('full_context', 'brief', 'balanced', 'detailed');
var
  Obj, Generation: TJSONObject;
  PromptIDs, GeneratedIDs: TJSONArray;
  TokenID: Integer;
begin
  Obj := TJSONObject.Create;
  try
    Obj.AddPair('schema', 'crosspascal-qwen-cuda-run-v1');
    if Value.StopReason = 'cancelled' then
      Obj.AddPair('status', 'CUDA_GENERATION_CANCELLED')
    else
      Obj.AddPair('status', 'CUDA_GENERATION_COMPLETED');
    Obj.AddPair('profile_id', Value.ProfileID);
    Obj.AddPair('source_revision', Value.SourceRevision);
    Obj.AddPair('source_tree_sha256', Value.SourceTreeSHA256);
    Obj.AddPair('device', 'cuda:0');
    Obj.AddPair('device_name', Value.DeviceName);
    Obj.AddPair('compute_version', Value.ComputeVersion);
    Obj.AddPair('free_device_bytes_before_load', JSONUnsigned(Value.FreeDeviceBytes));
    Obj.AddPair('total_device_bytes', JSONUnsigned(Value.TotalDeviceBytes));
    Obj.AddPair('model_sha256', Value.ModelSHA256);
    Obj.AddPair('params_sha256', Value.ParametersSHA256);
    Obj.AddPair('model_metadata_sha256', Value.ModelMetadataSHA256);
    Obj.AddPair('prompt', Value.Prompt);
    Obj.AddPair('context_tokens', JSONInteger(Value.ContextTokens));
    Obj.AddPair('available_new_tokens', JSONInteger(Value.AvailableNewTokens));
    Obj.AddPair('model_reused', JSONBoolean(Value.ModelReused));
    Generation := TJSONObject.Create;
    Obj.AddPair('generation', Generation);
    Generation.AddPair('max_new_tokens', JSONInteger(Value.GenerationOptions.MaxNewTokens));
    Generation.AddPair('system_prompt', Value.GenerationOptions.SystemPrompt);
    Generation.AddPair('plain_completion', JSONBoolean(Value.GenerationOptions.PlainCompletion));
    Generation.AddPair('answer_effort', EffortName[Value.GenerationOptions.Effort]);
    if Value.Sampled then
    begin
      Generation.AddPair('sampling', 'sample');
      Generation.AddPair('temperature', JSONFloat(Value.GenerationOptions.Temperature));
      Generation.AddPair('top_p', JSONFloat(Value.GenerationOptions.TopP));
      Generation.AddPair('top_k', JSONInteger(Value.GenerationOptions.TopK));
      Generation.AddPair('repetition_penalty', JSONFloat(Value.GenerationOptions.RepetitionPenalty));
      Generation.AddPair('seed', JSONInteger(Value.SamplingSeed));
    end
    else
      Generation.AddPair('sampling', 'greedy');
    PromptIDs := TJSONArray.Create;
    Obj.AddPair('prompt_token_ids', PromptIDs);
    for TokenID in Value.PromptTokenIDs do
      PromptIDs.Add(TokenID);
    GeneratedIDs := TJSONArray.Create;
    Obj.AddPair('generated_token_ids', GeneratedIDs);
    for TokenID in Value.GeneratedTokenIDs do
      GeneratedIDs.Add(TokenID);
    Obj.AddPair('answer', Value.Answer);
    Obj.AddPair('stop_reason', Value.StopReason);
    Obj.AddPair('elapsed_ms', JSONInteger(Int64(Value.ElapsedMilliseconds)));
    Obj.AddPair('acceptance_claim', JSONBoolean(False));
    Result := Obj.ToJSON;
  finally
    Obj.Free;
  end;
end;

end.
