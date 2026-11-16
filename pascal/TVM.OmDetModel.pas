{ ************************************************************************** }
{                                                                            }
{ TVM.OmDetModel                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.OmDetModel;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  ClipTokenizer, DLPack, TVM, TVM.CpuGraph, TVM.CudaExecution, TVM.WindowsPaths;

const
  OmDetCheckpointSHA256 = '439d1575d7e237ad565ed6969ea2a2dfcedf2086155e9ca3ac96cd6180a48cfd';
  OmDetTextGraphSHA256 = '65c63f2f54cae6d407641038a968cdf485ea97b70514097ac1efbb0665fb8a04';
  OmDetDetectorGraphSHA256 = '016a4ae886e0cec55503692cd75a50c8f9f0337de90885d5165584a61ca3e1e5';
  OmDetTextCacheCapacity = 32;
  OmDetHiddenSize = 512;

type
  TOmDetLabels = TArray<UnicodeString>;
  TOmDetProgress = procedure(const Stage: UnicodeString) of object;
  TOmDetRaw = record
    Boxes: TArray<Single>;
    Logits: TArray<Single>;
    ResolvedTask: UnicodeString;
    ClassTokenCounts: TArray<Integer>;
    TaskTokenCount: Integer;
    Truncated: Boolean;
  end;

  TOmDetModel = class
  private type
    TCloseState = (csOpen, csClosing, csClosed, csFailed, csQuarantined);
    TNativeResources = class
      Graphs: array[0..1] of TCpuGraph;
      Workspaces: array[0..1] of TCudaWorkspace;
      Cuda: TCudaExecution;
      Next: TNativeResources;
    end;
    TCacheEntry = record
      Sequence: TClipSequence;
      Hidden: TArray<Single>;
    end;
  private
    FNative: TNativeResources;
    FCloseState: TCloseState;
    FCloseError: string;
    FContextUncertain: Boolean;
    FDirectory: TWindowsModelDirectory;
    FLeases: array[0..4] of TFileStream;
    FDevice: TDLDeviceType;
    FDeviceName: string;
    FTextSHA256: string;
    FDetectorSHA256: string;
    FStage: string;
    FTokenizer: TClipTokenizer;
    FProjection: TArray<Single>;
    FCache: TArray<TCacheEntry>;
    FCacheNext: Integer;
    FCacheHits: UInt64;
    FCacheMisses: UInt64;
    procedure RememberError(const Operation, Detail: string);
    procedure ContextFailed(const Operation, Detail: string);
    procedure EnterContext;
    procedure LeaveContext;
    procedure FinishContext(const BodySucceeded: Boolean);
    procedure QuarantineNative;
    procedure ReleaseData;
    procedure Cleanup;
    function TextHidden(const Sequence: TClipSequence; const ShouldStop: TCpuGraphStop): TArray<Single>;
    function Execute(const Index: Integer; const Arguments: array of TTVMFFIObjectHandle;
      const ShouldStop: TCpuGraphStop): TTVMFFIAny;
  public
    constructor Create(const Bundle, RuntimeDirectory: string);
    destructor Destroy; override;
    procedure Close;
    function Run(const ImageInput: TArray<Single>; const Labels: TOmDetLabels;
      const Task: UnicodeString = ''; const AllowTruncate: Boolean = False;
      const ShouldStop: TCpuGraphStop = nil; const Progress: TOmDetProgress = nil): TOmDetRaw;
    property DeviceName: string read FDeviceName;
    property TextSHA256: string read FTextSHA256;
    property DetectorSHA256: string read FDetectorSHA256;
    property Stage: string read FStage;
    property CacheHits: UInt64 read FCacheHits;
    property CacheMisses: UInt64 read FCacheMisses;
  end;

function OmDetRequiresProcessRestart: Boolean;
function OmDetCleanupDiagnostic: string;

implementation

uses
  {$IFDEF FPC}Windows, Math{$ELSE}Winapi.Windows, System.Math{$ENDIF},
  CrossPascal.Json, CrossPascal.SHA256, QwenCuda, TVM.OmDet;

var
  CleanupLock: TRTLCriticalSection;
  RestartRequired: LongInt;
  CleanupDiagnostic: string;
  QuarantinedNative: TOmDetModel.TNativeResources;

function OmDetRequiresProcessRestart: Boolean;
begin
  Result := InterlockedCompareExchange(RestartRequired, 0, 0) <> 0;
end;

function OmDetCleanupDiagnostic: string;
begin
  EnterCriticalSection(CleanupLock);
  try
    Result := CleanupDiagnostic;
  finally
    LeaveCriticalSection(CleanupLock);
  end;
end;

function FiniteValue(const Value: Single): Boolean;
var
  Bits: Cardinal;
begin
  Move(Value, Bits, SizeOf(Bits));
  Result := (Bits and $7F800000) <> $7F800000;
end;

procedure CheckShape(const Handle: TTVMFFIObjectHandle; const Expected: array of Int64);
var
  Shape: TArray<Int64>;
  I: Integer;
begin
  Shape := TTVMRuntime.GetTensorShape(TTVMRuntime.TensorDLTensor(Handle));
  RequireCpu(Length(Shape) = Length(Expected), 'OmDet output rank differs');
  for I := 0 to High(Shape) do RequireCpu(Shape[I] = Expected[I], 'OmDet output shape differs');
end;

function CheckedFloats(const Handle: TTVMFFIObjectHandle): TArray<Single>;
var
  I: Integer;
begin
  Result := ReadFloatTensor(Handle);
  for I := 0 to High(Result) do RequireCpu(FiniteValue(Result[I]), 'Non-finite OmDet output');
end;

constructor TOmDetModel.Create(const Bundle, RuntimeDirectory: string);
const
  Roles: array[0..4] of string = ('text', 'detector', 'projection', 'vocab', 'merges');
  DataNames: array[0..2] of string = ('class-projection.f32', 'vocab.json', 'merges.txt');
  DataHashes: array[0..2] of string = (
    'b52b945b89ddbae183f89564e479dbc46d683ade99a3bce151140e67b815275d',
    'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
    '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a');
  DataSizes: array[0..2] of Integer = (1048576, 1059962, 524619);
var
  Document: TJSONValue;
  Root, Artifact, Provenance: TJSONObject;
  Artifacts: TJSONArray;
  Folder, Device, Architecture, FileName, Digest: string;
  TokenizerError: {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
  I: Integer;
  BodySucceeded: Boolean;
begin
  inherited Create;
  FNative := TNativeResources.Create;
  RequireCpu(not OmDetRequiresProcessRestart, 'Restart the application before loading another OmDet model.');
  FStage := 'Validate model';
  FDirectory := TWindowsModelDirectory.Create(Bundle);
  Folder := FDirectory.Path;
  Document := ReadModelJSON(BundleMember(Folder, 'manifest.json'));
  try
    Root := ModelObject(Document, 'OmDet manifest');
    RequireCpu((ModelInt(Root.GetValue('format_version'), 'format version') = 1) and
      (ModelText(Root.GetValue('adapter')) = 'omdet-turbo') and
      (ModelText(Root.GetValue('task_kind')) = 'text_guided_detection') and
      (ModelText(Root.GetValue('profile')) = 'omdet-turbo-tiny16') and
      (ModelText(Root.GetValue('os')) = 'windows') and
      (ModelText(Root.GetValue('arch')) = 'x86_64') and
      (ModelText(Root.GetValue('dtype')) = 'float32') and
      (ModelText(Root.GetValue('cpu_target')) = 'haswell'), 'Unsupported OmDet manifest');
    RequireCpu((ModelInt(Root.GetValue('image_side'), 'image side') = OmDetImageSide) and
      (ModelInt(Root.GetValue('class_capacity'), 'class capacity') = OmDetClassCapacity) and
      (ModelInt(Root.GetValue('token_capacity'), 'token capacity') = ClipSequenceLength) and
      (ModelInt(Root.GetValue('hidden_size'), 'hidden size') = OmDetHiddenSize) and
      (ModelInt(Root.GetValue('query_count'), 'query count') = OmDetQueryCount) and
      (ModelText(Root.GetValue('preprocessing')) = 'rgb-pillow-bilinear-omdet') and
      (ModelText(Root.GetValue('postprocessing')) = 'sigmoid-valid-prefix-topk900-class-nms-clip'),
      'Unsupported OmDet input/output contract');
    Provenance := ModelObject(Root.GetValue('provenance'), 'OmDet provenance');
    RequireCpu(
      SameText(ModelText(Provenance.GetValue('checkpoint_sha256')), OmDetCheckpointSHA256) and SameText(ModelText(Provenance.GetValue('text_onnx_sha256')), OmDetTextGraphSHA256) and SameText(ModelText(Provenance.GetValue('detector_onnx_sha256')), OmDetDetectorGraphSHA256),
      'OmDet trained source differs'
    );
    Device := ModelText(Root.GetValue('device'));
    RequireCpu((Device = 'cpu') or (Device = 'cuda'), 'Unsupported OmDet device');
    Architecture := ModelText(Root.GetValue('cuda_arch'));
    FDevice := kDLCPU;
    FDeviceName := 'CPU';
    if Device = 'cuda' then
      FDevice := kDLCUDA
    else
      RequireCpu(Architecture = '', 'CPU OmDet bundle contains a CUDA target');
    Artifacts := ModelArray(Root.GetValue('artifacts'), 'OmDet artifacts');
    RequireCpu(Artifacts.Count = Length(FLeases), 'OmDet artifact count differs');
    for I := 0 to High(FLeases) do
    begin
      Artifact := ModelObject(Artifacts.Items[I], 'OmDet artifact');
      RequireCpu(ModelText(Artifact.GetValue('role')) = Roles[I], 'OmDet artifact order differs');
      if I < 2 then
        FileName := Roles[I] + '.' + Device + '.dll'
      else
        FileName := DataNames[I - 2];
      RequireCpu(ModelText(Artifact.GetValue('file')) = FileName, 'OmDet artifact filename differs');
      FLeases[I] := TFileStream.Create(BundleMember(Folder, FileName), fmOpenRead or fmShareDenyWrite);
      RequireCpu(
        (FLeases[I].Size > 0) and (FLeases[I].Size <= 1024 * 1024 * 1024) and (FLeases[I].Size = ModelInt(Artifact.GetValue('bytes'), 'artifact size')),
        'OmDet artifact size differs'
      );
      Digest := SHA256Stream(FLeases[I]);
      RequireCpu(SameText(Digest, ModelText(Artifact.GetValue('sha256'))), 'OmDet artifact SHA-256 differs');
      if I < 2 then
      begin
        if I = 0 then
          FTextSHA256 := Digest
        else
          FDetectorSHA256 := Digest;
      end
      else
        RequireCpu((FLeases[I].Size = DataSizes[I - 2]) and SameText(Digest, DataHashes[I - 2]),
        'OmDet trained tokenizer/projection differs');
    end;
    FStage := 'Load tokenizer and projection';
    FTokenizer := TClipTokenizer.Create;
    if not FTokenizer.Load(Folder, @TokenizerError) then
      raise ECpuGraph.Create('OmDet tokenizer: ' + string(TokenizerError));
    SetLength(FProjection, OmDetHiddenSize * OmDetHiddenSize);
    FLeases[2].Position := 0;
    FLeases[2].ReadBuffer(FProjection[0], Length(FProjection) * SizeOf(Single));
    for I := 0 to High(FProjection) do
      RequireCpu(FiniteValue(FProjection[I]), 'Non-finite OmDet projection');
    InitializeRuntime(ModelRuntimeDirectory(RuntimeDirectory), FDevice = kDLCUDA);
    if FDevice = kDLCUDA then
    begin
      FNative.Cuda := TCudaExecution.Create;
      FNative.Cuda.RequireArchitecture(Architecture);
      FDeviceName := FNative.Cuda.DeviceName + ' (' + Architecture + ')';
    end;
    EnterContext;
    BodySucceeded := False;
    try
      if FNative.Cuda <> nil then
        RequireCpu(FNative.Cuda.FreeBytes >= UInt64(2) * 1024 * 1024 * 1024,
          'OmDet requires at least 2 GiB of free CUDA memory');
      FStage := 'Load native graphs';
      for I := 0 to 1 do
      begin
        FileName := BundleMember(Folder, Roles[I] + '.' + Device + '.dll');
        FNative.Graphs[I] := TCpuGraph.Create(FileName, FDevice);
        if FNative.Cuda <> nil then
          FNative.Workspaces[I] := TCudaWorkspace.Create(FNative.Cuda, FileName, UInt64(1024) * 1024 * 1024);
      end;
      BodySucceeded := True;
    finally
      FinishContext(BodySucceeded);
    end;
    FStage := 'Ready';
  finally
    Document.Free;
  end;
end;

procedure TOmDetModel.ReleaseData;
var
  I: Integer;
begin
  FCache := nil;
  FProjection := nil;
  FreeAndNil(FTokenizer);
  for I := High(FLeases) downto Low(FLeases) do
  begin
    try
      FreeAndNil(FLeases[I]);
    except
      on E: Exception do RememberError('Model file cleanup', E.Message);
    end;
  end;
end;

procedure TOmDetModel.RememberError(const Operation, Detail: string);
begin
  if FCloseError <> '' then FCloseError := FCloseError + ' ';
  FCloseError := FCloseError + Operation + ': ' + Detail;
end;

procedure TOmDetModel.ContextFailed(const Operation, Detail: string);
begin
  FContextUncertain := True;
  InterlockedExchange(RestartRequired, 1);
  RememberError(Operation, Detail);
  EnterCriticalSection(CleanupLock);
  try
    if CleanupDiagnostic = '' then
      CleanupDiagnostic := FCloseError
    else if Pos(Operation + ': ' + Detail, CleanupDiagnostic) = 0 then
        CleanupDiagnostic := CleanupDiagnostic + ' ' + Operation + ': ' + Detail;
  finally
    LeaveCriticalSection(CleanupLock);
  end;
end;

procedure TOmDetModel.EnterContext;
begin
  if FNative.Cuda = nil then Exit;
  try
    FNative.Cuda.Enter;
  except
    on E: Exception do
    begin
      ContextFailed('CUDA context entry', E.Message);
      raise;
    end;
  end;
end;

procedure TOmDetModel.LeaveContext;
begin
  if FNative.Cuda = nil then Exit;
  try
    FNative.Cuda.Leave;
  except
    on E: Exception do
    begin
      ContextFailed('CUDA context restoration', E.Message);
      raise;
    end;
  end;
end;

procedure TOmDetModel.FinishContext(const BodySucceeded: Boolean);
begin
  if BodySucceeded then
    LeaveContext
  else begin
    try
      LeaveContext;
    except
      // The original constructor or inference exception remains primary.
    end;
  end;
end;

procedure TOmDetModel.QuarantineNative;
begin
  if FNative = nil then Exit;
  InterlockedExchange(RestartRequired, 1);
  EnterCriticalSection(CleanupLock);
  try
    FNative.Next := QuarantinedNative;
    QuarantinedNative := FNative;
    FNative := nil;
  finally
    LeaveCriticalSection(CleanupLock);
  end;
end;

procedure TOmDetModel.Cleanup;
var
  Entered, Quarantined: Boolean;
  Workspace: TCudaWorkspace;
  Graph: TCpuGraph;
  I: Integer;
begin
  if FCloseState <> csOpen then Exit;
  FCloseState := csClosing;
  Entered := False;
  Quarantined := False;
  try
    try
      if FNative <> nil then
      begin
        if (FNative.Cuda <> nil) and OmDetRequiresProcessRestart then
          FContextUncertain := True;
        if not FContextUncertain then
        begin
          try
            EnterContext;
            Entered := FNative.Cuda <> nil;
          except
            // EnterContext records the failure. Never guess whether push occurred.
          end;
          if not FContextUncertain then
          begin
            for I := High(FNative.Graphs) downto Low(FNative.Graphs) do
              if not FContextUncertain then
              begin
                Workspace := FNative.Workspaces[I];
                FNative.Workspaces[I] := nil;
                try
                  Workspace.Free;
                except
                  on E: Exception do ContextFailed('CUDA workspace cleanup', E.Message);
                end;
                if not FContextUncertain then
                begin
                  Graph := FNative.Graphs[I];
                  FNative.Graphs[I] := nil;
                  try
                    Graph.Free;
                  except
                    on E: Exception do ContextFailed('Model graph cleanup', E.Message);
                  end;
                end;
              end;
          end;
          if Entered then
          begin
            try
              LeaveContext;
            except
              // Exactly one pop for the successful push; no retry after an error.
            end;
          end;
        end;
        if FContextUncertain then
        begin
          if FCloseError = '' then RememberError('CUDA cleanup', 'Context state is uncertain.');
          QuarantineNative;
          Quarantined := True;
        end
        else begin
          FreeAndNil(FNative.Cuda);
          FreeAndNil(FNative);
        end;
      end;
    except
      on E: Exception do
      begin
        ContextFailed('Native cleanup', E.Message);
        QuarantineNative;
        Quarantined := True;
      end;
    end;
  finally
    try
      ReleaseData;
    except
      on E: Exception do RememberError('Model file cleanup', E.Message);
    end;
    try
      FreeAndNil(FDirectory);
    except
      on E: Exception do RememberError('Model path cleanup', E.Message);
    end;
    if Quarantined then
      FCloseState := csQuarantined
    else if (FCloseError <> '') or (FNative <> nil) or FContextUncertain then
        FCloseState := csFailed
      else
        FCloseState := csClosed;
  end;
end;

procedure TOmDetModel.Close;
begin
  Cleanup;
  RequireCpu(FCloseState = csClosed,
    'OmDet cleanup did not complete. ' + FCloseError +
    ' Restart the application before running another model.');
end;

destructor TOmDetModel.Destroy;
begin
  try
    Cleanup;
  except
    // Destruction must preserve an exception already being unwound.
    try
      QuarantineNative;
    except
    end;
    try
      ReleaseData;
    except
    end;
    try
      FreeAndNil(FDirectory);
    except
    end;
  end;
  inherited;
end;

function TOmDetModel.Execute(const Index: Integer; const Arguments: array of TTVMFFIObjectHandle;
  const ShouldStop: TCpuGraphStop): TTVMFFIAny;
begin
  if FNative.Workspaces[Index] <> nil then
    Result := FNative.Workspaces[Index].Run(FNative.Graphs[Index], Arguments, ShouldStop)
  else
    Result := FNative.Graphs[Index].Run(Arguments, ShouldStop);
end;

function TOmDetModel.TextHidden(const Sequence: TClipSequence;
  const ShouldStop: TCpuGraphStop): TArray<Single>;
var
  Argument, Returned, Output: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
  I, Slot: Integer;
begin
  for I := 0 to High(FCache) do
    if CompareMem(@FCache[I].Sequence.Ids[0], @Sequence.Ids[0], SizeOf(Sequence.Ids)) then
    begin
      Inc(FCacheHits);
      Exit(FCache[I].Hidden);
    end;
  for I := Low(Sequence.Ids) to High(Sequence.Ids) do
    RequireCpu((Sequence.Ids[I] >= 0) and (Sequence.Ids[I] < ClipVocabularySize),
      'OmDet token ID is out of range');
  Inc(FCacheMisses);
  Argument := nil;
  Returned := nil;
  Output := nil;
  try
    Argument := DeviceTensor([1, ClipSequenceLength], kDLInt, 64, @Sequence.Ids[0], FDevice);
    Value := Execute(0, [Argument], ShouldStop);
    Returned := Value.VObj;
    if Value.TypeIndex = kTVMFFITensor then
    begin
      CheckShape(Returned, [1, ClipSequenceLength, OmDetHiddenSize]);
      Result := CheckedFloats(Returned);
    end
    else begin
      Output := TensorItem(Returned, 0);
      CheckShape(Output, [1, ClipSequenceLength, OmDetHiddenSize]);
      Result := CheckedFloats(Output);
    end;
  finally
    TTVMRuntime.ObjectRelease(Output);
    TTVMRuntime.ObjectRelease(Returned);
    TTVMRuntime.ObjectRelease(Argument);
  end;
  Slot := Length(FCache);
  if Slot < OmDetTextCacheCapacity then
    SetLength(FCache, Slot + 1)
  else begin
    Slot := FCacheNext;
    FCacheNext := (FCacheNext + 1) mod OmDetTextCacheCapacity;
  end;
  FCache[Slot].Sequence := Sequence;
  FCache[Slot].Hidden := Result;
end;

function TOmDetModel.Run(const ImageInput: TArray<Single>; const Labels: TOmDetLabels;
  const Task: UnicodeString; const AllowTruncate: Boolean; const ShouldStop: TCpuGraphStop;
  const Progress: TOmDetProgress): TOmDetRaw;
var
  Sequences: TArray<TClipSequence>;
  TaskSequence: TClipSequence;
  ClassFeatures, TaskFeatures, Hidden: TArray<Single>;
  Arguments: array[0..3] of TTVMFFIObjectHandle;
  Returned, Boxes, Scores: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
  LabelIndex, I, J, K, EOS: Integer;
  Total, Term: Double;
  BodySucceeded: Boolean;

  procedure Stage(const MessageText: string);
  begin
    FStage := MessageText;
    if Assigned(ShouldStop) and ShouldStop() then raise ECpuGraphStopped.Create('OmDet stopped');
    if Assigned(Progress) then Progress(MessageText);
  end;

begin
  RequireCpu((FCloseState = csOpen) and not OmDetRequiresProcessRestart,
    'Restart the application before running another OmDet model.');
  Result := Default(TOmDetRaw);
  Stage('Validate query and image input');
  RequireCpu((Length(Labels) >= 1) and (Length(Labels) <= OmDetClassCapacity),
    'OmDet accepts 1 to 16 text classes');
  RequireCpu(Length(ImageInput) = 3 * OmDetImageSide * OmDetImageSide, 'OmDet image input size differs');
  for I := 0 to High(ImageInput) do
    RequireCpu(FiniteValue(ImageInput[I]), 'Non-finite OmDet image input');
  Result.ResolvedTask := Task;
  if Task = '' then
  begin
    Result.ResolvedTask := 'Detect ';
    for I := 0 to High(Labels) do
    begin
      if I > 0 then Result.ResolvedTask := Result.ResolvedTask + ', ';
      Result.ResolvedTask := Result.ResolvedTask + Labels[I];
    end;
    Result.ResolvedTask := Result.ResolvedTask + '.';
  end;
  Stage('Tokenize classes and task');
  SetLength(Sequences, Length(Labels));
  SetLength(Result.ClassTokenCounts, Length(Labels));
  for I := 0 to High(Labels) do
  begin
    RequireCpu(Trim(Labels[I]) <> '', 'OmDet class names must not be empty');
    Sequences[I] := FTokenizer.EncodeSequence(Labels[I], AllowTruncate);
    Result.ClassTokenCounts[I] := Sequences[I].TokenCount;
    Result.Truncated := Result.Truncated or Sequences[I].Truncated;
  end;
  TaskSequence := FTokenizer.EncodeSequence(Result.ResolvedTask, AllowTruncate);
  Result.TaskTokenCount := TaskSequence.TokenCount;
  Result.Truncated := Result.Truncated or TaskSequence.Truncated;
  FillChar(Arguments, SizeOf(Arguments), 0);
  Returned := nil;
  Boxes := nil;
  Scores := nil;
  EnterContext;
  BodySucceeded := False;
  try
    SetLength(ClassFeatures, OmDetClassCapacity * OmDetHiddenSize);
    for LabelIndex := 0 to High(Labels) do
    begin
      Stage(Format('Encode class %d of %d', [LabelIndex + 1, Length(Labels)]));
      Hidden := TextHidden(Sequences[LabelIndex], ShouldStop);
      EOS := 0;
      for I := 1 to ClipSequenceLength - 1 do
        if Sequences[LabelIndex].Ids[I] > Sequences[LabelIndex].Ids[EOS] then EOS := I;
      for J := 0 to OmDetHiddenSize - 1 do
      begin
        Total := 0;
        for K := 0 to OmDetHiddenSize - 1 do
        begin
          Term := Hidden[EOS * OmDetHiddenSize + K];
          Term := Term * FProjection[K * OmDetHiddenSize + J];
          Total := Total + Term;
        end;
        ClassFeatures[LabelIndex * OmDetHiddenSize + J] := Total;
      end;
    end;
    for I := Length(Labels) to OmDetClassCapacity - 1 do
      Move(ClassFeatures[(Length(Labels) - 1) * OmDetHiddenSize],
        ClassFeatures[I * OmDetHiddenSize], OmDetHiddenSize * SizeOf(Single));
    Stage('Encode task');
    Hidden := TextHidden(TaskSequence, ShouldStop);
    SetLength(TaskFeatures, ClipSequenceLength * OmDetHiddenSize);
    for I := 0 to ClipSequenceLength - 1 do
      if TaskSequence.AttentionMask[I] <> 0 then
        Move(Hidden[I * OmDetHiddenSize], TaskFeatures[I * OmDetHiddenSize],
          OmDetHiddenSize * SizeOf(Single));
    Stage('Detect objects');
    Arguments[0] := DeviceTensor([1, 3, OmDetImageSide, OmDetImageSide], kDLFloat, 32,
      Pointer(ImageInput), FDevice);
    Arguments[1] := DeviceTensor([OmDetClassCapacity, 1, OmDetHiddenSize], kDLFloat, 32,
      Pointer(ClassFeatures), FDevice);
    Arguments[2] := DeviceTensor([ClipSequenceLength, 1, OmDetHiddenSize], kDLFloat, 32,
      Pointer(TaskFeatures), FDevice);
    Arguments[3] := DeviceTensor([1, ClipSequenceLength], kDLInt, 64,
      @TaskSequence.AttentionMask[0], FDevice);
    Value := Execute(1, Arguments, ShouldStop);
    Returned := Value.VObj;
    Boxes := TensorItem(Returned, 0);
    Scores := TensorItem(Returned, 1);
    CheckShape(Boxes, [1, OmDetQueryCount, 4]);
    CheckShape(Scores, [1, OmDetQueryCount, OmDetClassCapacity]);
    Result.Boxes := CheckedFloats(Boxes);
    Result.Logits := CheckedFloats(Scores);
    Stage('Inference complete');
    BodySucceeded := True;
  finally
    TTVMRuntime.ObjectRelease(Scores);
    TTVMRuntime.ObjectRelease(Boxes);
    TTVMRuntime.ObjectRelease(Returned);
    for I := High(Arguments) downto Low(Arguments) do TTVMRuntime.ObjectRelease(Arguments[I]);
    FinishContext(BodySucceeded);
  end;
end;

initialization
  InitializeCriticalSection(CleanupLock);

finalization
  // Quarantined native groups stay untouched until process termination.
  DeleteCriticalSection(CleanupLock);

end.
