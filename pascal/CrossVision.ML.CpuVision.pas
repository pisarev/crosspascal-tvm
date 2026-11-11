{ ************************************************************************** }
{                                                                            }
{ CrossVision.ML.CpuVision                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossVision.ML.CpuVision;

{$IFDEF FPC}
{$MODE DELPHIUNICODE}
{$ENDIF}

{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes,{$ELSE}System.SysUtils, System.Classes,{$ENDIF}
  CrossPascal.Json, CrossVision.Mat, CrossVision.ML;

type
  TCpuVisionProfile = record
    ID: string;
    Title: string;
    TaskKind: string;
    Adapter: string;
    InputName: string;
    InputSize: Integer;
    ClassCount: Integer;
    SourceHashes: TArray<string>;
    CatalogSHA256: string;
    Mean: TArray<Single>;
    Std: TArray<Single>;
    ScoreThreshold: Single;
    NMSThreshold: Single;
  end;

  TCpuVisionResult = record
    ProfileID: string;
    Device: string;
    DeviceName: string;
    CudaArchitecture: string;
    SourceSHA256: string;
    ModelSHA256: string;
    InputSHA256: string;
    ManifestSHA256: string;
    InputName: string;
    RuntimeDirectory: string;
    Width: Integer;
    Height: Integer;
    ElapsedMilliseconds: Int64;
    Classifications: TClassificationArray;
    Detections: TDetectionArray;
    AuditJSONText: string;
    JSONText: string;
  end;

  TCpuVisionProgress = {$IFDEF FPC}procedure(const MessageText: string) of object{$ELSE}reference to procedure(const MessageText: string){$ENDIF};

function LoadCpuVisionProfiles(const CatalogPath: string): TArray<TCpuVisionProfile>;
function RunCpuVision(const Profile: TCpuVisionProfile;
  const BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256: string; const Image: IMat;
  const Progress: TCpuVisionProgress = nil; const AuditDirectory: string = ''): TCpuVisionResult;
function RunVision(const Profile: TCpuVisionProfile;
  const BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256: string; const Image: IMat;
  const Progress: TCpuVisionProgress = nil): TCpuVisionResult;
procedure WriteCpuVisionReport(const Path, JSONText: string);

implementation

uses
  {$IFDEF FPC}Windows, Math, Types, Generics.Collections, Generics.Defaults, {$ELSE}
  Winapi.Windows, System.Math, System.Types, System.Generics.Collections,
  System.Generics.Defaults, {$ENDIF}
  {$IFDEF FPC}CrossPascal.WinMemory,{$ENDIF} CrossPascal.SHA256, DLPack, TVM, TVM.Yolo11,
  TVM.RtDetr, CrossVision.Geometry.Types, QwenCuda, TVM.CudaExecution;

const
  RuntimeNames: array[0..3] of string = ('tvm_ffi.dll', 'tvm_runtime.dll', 'tvm_runtime_extra.dll', 'tvm_runtime_cuda.dll');
  RuntimeHashes: array[0..3] of string = (
    '2ed31dce9e331d23d6b4a2fa37e4166aa7843c334a3c17381f92dd3bcf122913',
    '45e3a9b5d76ea6950cde4442e5bd3208c66a8211d02936e2c0e727f52d2c4ac1',
    '4fe689aa56b1a46349def33dd75f5066f70e1a0ba5d156e2e74e8667d3a7beee',
    '4c67e81596267e94b153e855290fda1320ac0f4852faa36f8ac7184be585e87b');
  LightSqueezeNetSHA256 = '770b0f3c8623e18bf58b53754d710051b4c268248422142980a132bbe6dfe908';

type
  PCpuVisionAudit = ^TCpuVisionAudit;
  TCpuVisionAudit = record
    Profile: TCpuVisionProfile;
    Binding: TCpuVisionResult;
    Directory: string;
    Captured: Boolean;
    JSONText: string;
  end;

var
  VisionRuntimeLock: TRTLCriticalSection;
  LoadedRuntimeDirectory: string;
  RuntimeLeases: array[0..3] of TFileStream;

procedure Require(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then
    raise Exception.Create(MessageText);
end;

function IsSHA256(const Value: string): Boolean;
var
  C: Char;
begin
  Result := Length(Value) = 64;
  if not Result then Exit;
  for C in LowerCase(Value) do
    if not CharInSet(C, ['0'..'9', 'a'..'f']) then
      Exit(False);
end;

function Field(const Obj: TJSONObject; const Name: string): TJSONValue;
begin
  Result := Obj.GetValue(Name);
  Require(Result <> nil, 'Missing JSON field: ' + Name);
end;

function TextField(const Obj: TJSONObject; const Name: string): string;
var
  V: TJSONValue;
begin
  V := Field(Obj, Name);
  Require(V is TJSONString, 'Expected JSON string: ' + Name);
  Result := JSONValueText(V);
end;

function IntegerField(const Obj: TJSONObject; const Name: string): Integer;
var
  V: TJSONValue;
begin
  V := Field(Obj, Name);
  Require((V is TJSONNumber) and TryStrToInt(JSONValueText(V), Result), 'Expected JSON integer: ' + Name);
end;

function ObjectField(const Obj: TJSONObject; const Name: string): TJSONObject;
var
  V: TJSONValue;
begin
  V := Field(Obj, Name);
  Require(V is TJSONObject, 'Expected JSON object: ' + Name);
  Result := TJSONObject(V);
end;

function ArrayField(const Obj: TJSONObject; const Name: string): TJSONArray;
var
  V: TJSONValue;
begin
  V := Field(Obj, Name);
  Require(V is TJSONArray, 'Expected JSON array: ' + Name);
  Result := TJSONArray(V);
end;

function NumberValue(const V: TJSONValue; const Name: string): Double;
begin
  Require(V is TJSONNumber, 'Expected JSON number: ' + Name);
  Require(TryStrToFloat(JSONValueText(V), Result, JSONInvariantFormat),
    'Invalid JSON number: ' + Name);
  Require(not IsNan(Result) and not IsInfinite(Result), 'Non-finite value: ' + Name);
end;

function FloatArray(const Obj: TJSONObject; const Name: string): TArray<Single>;
var
  A: TJSONArray;
  I: Integer;
begin
  A := ArrayField(Obj, Name);
  Require((A.Count = 0) or (A.Count = 3), 'Expected zero or three RGB values: ' + Name);
  SetLength(Result, A.Count);
  for I := 0 to A.Count - 1 do
    Result[I] := NumberValue(A.Items[I], Name);
end;

function HashStream(const Stream: TStream): string;
begin
  Stream.Position := 0;
  try
    Result := SHA256Stream(Stream);
  finally
    Stream.Position := 0;
  end;
end;

function ReadTextStream(const Stream: TStream; const Maximum: Int64): string;
var
  Bytes: TBytes;
begin
  Require((Stream.Size > 0) and (Stream.Size <= Maximum), 'Invalid metadata file size.');
  SetLength(Bytes, Integer(Stream.Size));
  Stream.Position := 0;
  Stream.ReadBuffer(Bytes[0], Length(Bytes));
  Result := TEncoding.UTF8.GetString(Bytes);
  if (Result <> '') and (Result[1] = #$FEFF) then Delete(Result, 1, 1);
  Stream.Position := 0;
end;

function ParseObject(const Text: string): TJSONObject;
var
  Value: TJSONValue;
begin
  Value := ParseCrossJSON(Text);
  if not (Value is TJSONObject) then
  begin
    Value.Free;
    raise Exception.Create('Expected a JSON object.');
  end;
  Result := TJSONObject(Value);
end;

function SafeFile(const Directory, Name: string): string;
begin
  Require((Name <> '') and (Name = ExtractFileName(Name)) and
    (Pos(':', Name) = 0) and (Pos('/', Name) = 0) and (Pos('\', Name) = 0) and
    (Name <> '.') and (Name <> '..'), 'A bundle file must be a plain filename.');
  Result := IncludeTrailingPathDelimiter(ExpandFileName(Directory)) + Name;
end;

function ContainsHash(const Hashes: TArray<string>; const Value: string): Boolean;
var
  Hash: string;
begin
  for Hash in Hashes do
    if SameText(Hash, Value) then Exit(True);
  Result := False;
end;

function LoadCpuVisionProfiles(const CatalogPath: string): TArray<TCpuVisionProfile>;
var
  Stream: TFileStream;
  Root, Item: TJSONObject;
  Rows, Hashes: TJSONArray;
  I, J, K: Integer;
  CatalogHash: string;
begin
  Stream := TFileStream.Create(CatalogPath, fmOpenRead or fmShareDenyWrite);
  try
    CatalogHash := HashStream(Stream);
    Root := ParseObject(ReadTextStream(Stream, 1024 * 1024));
    try
      Require(TextField(Root, 'schema') = 'crosspascal.cpu-vision-profiles.v1',
        'Unsupported CPU vision catalog.');
      Rows := ArrayField(Root, 'models');
      Require(Rows.Count = 5, 'The supplied catalog must contain the five named profiles.');
      SetLength(Result, Rows.Count);
      for I := 0 to Rows.Count - 1 do
      begin
        Require(Rows.Items[I] is TJSONObject, 'Invalid profile entry.');
        Item := TJSONObject(Rows.Items[I]);
        Result[I].ID := TextField(Item, 'id');
        Result[I].CatalogSHA256 := CatalogHash;
        Require((Result[I].ID = 'squeezenet') or (Result[I].ID = 'resnet50') or
          (Result[I].ID = 'yolo11n-cls') or (Result[I].ID = 'yolo11n') or
          (Result[I].ID = 'rtdetr-r18vd'), 'Unknown profile ID.');
        for J := 0 to I - 1 do
          Require(Result[J].ID <> Result[I].ID, 'Duplicate profile ID.');
        Result[I].Title := TextField(Item, 'title');
        Result[I].TaskKind := TextField(Item, 'task_kind');
        Result[I].Adapter := TextField(Item, 'adapter');
        Result[I].InputName := TextField(Item, 'input_name');
        Result[I].InputSize := IntegerField(Item, 'input_size');
        Result[I].ClassCount := IntegerField(Item, 'class_count');
        Result[I].Mean := FloatArray(Item, 'mean');
        Result[I].Std := FloatArray(Item, 'std');
        Require(Length(Result[I].Mean) = Length(Result[I].Std),
          'Mean and standard deviation must have matching lengths.');
        for K := 0 to High(Result[I].Std) do
          Require(Result[I].Std[K] > 0, 'Standard deviations must be positive.');
        Result[I].ScoreThreshold := NumberValue(Field(Item, 'score_threshold'), 'score_threshold');
        Result[I].NMSThreshold := NumberValue(Field(Item, 'nms_threshold'), 'nms_threshold');
        Require((Result[I].ScoreThreshold >= 0) and (Result[I].ScoreThreshold <= 1) and
          (Result[I].NMSThreshold >= 0) and (Result[I].NMSThreshold <= 1),
          'Thresholds must be within [0,1].');
        if Result[I].TaskKind = 'classification' then
          Require((Result[I].InputSize = 224) and (Result[I].ClassCount = 1000) and
            (Result[I].Adapter = 'yolo11'), 'Invalid classifier profile.')
        else
          Require((Result[I].TaskKind = 'object_detection') and
            (Result[I].InputSize = 640) and (Result[I].ClassCount = 80) and
            ((Result[I].Adapter = 'yolo11') or (Result[I].Adapter = 'rtdetr')),
            'Invalid detector profile.');
        if Result[I].TaskKind = 'object_detection' then
          Require(Length(Result[I].Mean) = 0,
            'Detector adapters support RGB/255 without additional normalization.');
        Hashes := ArrayField(Item, 'accepted_source_sha256');
        SetLength(Result[I].SourceHashes, Hashes.Count);
        for J := 0 to Hashes.Count - 1 do
        begin
          Require(Hashes.Items[J] is TJSONString, 'Invalid source hash entry.');
          Result[I].SourceHashes[J] := LowerCase(JSONValueText(Hashes.Items[J]));
          Require(IsSHA256(Result[I].SourceHashes[J]), 'Invalid source SHA-256.');
          Require(Result[I].SourceHashes[J] <> LightSqueezeNetSHA256,
            'The ONNX light test fixture is not a trained SqueezeNet classifier.');
        end;
      end;
    finally
      Root.Free;
    end;
  finally
    Stream.Free;
  end;
end;

procedure InitializeVisionRuntime(const RuntimeDirectory: string; const NeedCuda: Boolean);
var
  Directory: string;
  Lease: TFileStream;
  I, Last: Integer;
begin
  Directory := IncludeTrailingPathDelimiter(ExpandFileName(RuntimeDirectory));
  EnterCriticalSection(VisionRuntimeLock);
  try
    if LoadedRuntimeDirectory <> '' then
      Require(SameText(LoadedRuntimeDirectory, Directory),
        'Restart the application before selecting a different TVM runtime.');
    LoadedRuntimeDirectory := Directory;
    Last := 2;
    if NeedCuda then Last := 3;
    for I := 0 to Last do
      if RuntimeLeases[I] = nil then
      begin
        Lease := TFileStream.Create(Directory + RuntimeNames[I], fmOpenRead or fmShareDenyWrite);
        try
          Require(SameText(HashStream(Lease), RuntimeHashes[I]),
            'Runtime SHA-256 does not match the supplied family: ' + RuntimeNames[I]);
          RuntimeLeases[I] := Lease;
          Lease := nil;
        finally
          Lease.Free;
        end;
      end;
    QwenCuda.InitializeRuntime(Directory, NeedCuda);
  finally
    LeaveCriticalSection(VisionRuntimeLock);
  end;
end;

procedure ValidateManifest(const Root: TJSONObject; const Profile: TCpuVisionProfile;
  const AllowCuda: Boolean; out ModelName, ModelHash, SourceHash, InputName, Device, CudaArch: string);
var
  Artifact, IO, Provenance, Vision: TJSONObject;
  Shape: TJSONArray;
  Mean, Std: TArray<Single>;
  I, Dimension: Integer;
begin
  Require(TextField(Root, 'task_kind') = Profile.TaskKind,
    'Bundle task does not match the selected profile.');
  Require(TextField(Root, 'adapter') = Profile.Adapter,
    'Bundle adapter does not match the selected profile.');
  Require(IntegerField(Root, 'adapter_api_version') = 1, 'Unsupported adapter API version.');
  Artifact := ObjectField(Root, 'artifact');
  Device := TextField(Artifact, 'device');
  Require((TextField(Artifact, 'os') = 'windows') and (TextField(Artifact, 'arch') = 'x86_64') and
    (TextField(Artifact, 'dtype') = 'float32'), 'Expected a Windows x64 float32 bundle.');
  Require((Device = 'cpu') or (AllowCuda and (Device = 'cuda')), 'The bundle device is not supported here.');
  CudaArch := '';
  if Device = 'cuda' then CudaArch := TextField(Artifact, 'cuda_arch');
  Require(TextField(Artifact, 'format') = 'so', 'Expected a TVM shared-library artifact.');
  Require(TextField(Artifact, 'entry') = 'main', 'Expected the main model entry.');
  ModelName := TextField(Artifact, 'file');
  Require(SameText(ExtractFileExt(ModelName), '.dll'), 'The model artifact must be a DLL.');
  ModelHash := LowerCase(TextField(Artifact, 'hash'));
  Require(IsSHA256(ModelHash), 'Missing or invalid model-library hash.');
  IO := ObjectField(Root, 'io');
  InputName := TextField(IO, 'name');
  Require((InputName <> '') and (Pos(#0, InputName) = 0) and
    ((Profile.InputName = '') or (InputName = Profile.InputName)) and
    (TextField(IO, 'layout') = 'NCHW') and (TextField(IO, 'dtype') = 'float32'),
    'The input name, layout or dtype does not match the profile.');
  Shape := ArrayField(IO, 'shape');
  Require(Shape.Count = 4, 'Expected a four-dimensional input.');
  for I := 0 to 3 do
  begin
    Require((Shape.Items[I] is TJSONNumber) and TryStrToInt(JSONValueText(Shape.Items[I]), Dimension),
      'Expected fixed integer input dimensions.');
    case I of
      0: Require(Dimension = 1, 'Only batch size 1 is supported.');
      1: Require(Dimension = 3, 'Only three-channel RGB inputs are supported.');
    else
      Require(Dimension = Profile.InputSize, 'Input size does not match the selected profile.');
    end;
  end;
  Provenance := ObjectField(Root, 'provenance');
  SourceHash := LowerCase(TextField(Provenance, 'source_hash'));
  Require(IsSHA256(SourceHash), 'Missing source-model hash.');
  Require(SourceHash <> LightSqueezeNetSHA256,
    'The ONNX light test fixture is not a trained SqueezeNet classifier.');
  Require(Length(Profile.SourceHashes) > 0,
    'This profile still needs a pinned trained model conversion. See the example README.');
  Require(ContainsHash(Profile.SourceHashes, SourceHash),
    'The source model is not one of the pinned sources for this profile.');
  if Root.GetValue('vision') <> nil then
  begin
    Vision := ObjectField(Root, 'vision');
    Mean := FloatArray(Vision, 'mean');
    Std := FloatArray(Vision, 'std');
    Require((Length(Mean) = Length(Profile.Mean)) and (Length(Std) = Length(Profile.Std)),
      'Bundle normalization differs from the selected profile.');
    for I := 0 to High(Mean) do
      Require((Abs(Mean[I] - Profile.Mean[I]) < 1e-6) and
        (Abs(Std[I] - Profile.Std[I]) < 1e-6), 'Bundle normalization differs from the profile.');
  end;
end;

function SaveAuditTensor(const Tensor: PDLTensor; const ExpectedShape: array of Int64;
  const Directory, FileName, Role: string; out Elements: Int64): TJSONObject;
const
  MaximumBytes = 16 * 1024 * 1024;
var
  Shape: TArray<Int64>;
  ShapeJSON, StridesJSON: TJSONArray;
  I, ByteCount: Integer;
  ExpectedStride: Int64;
  Bytes: TBytes;
  SHA256: string;
  Handle: THandle;
  Stream: THandleStream;
begin
  Require(Assigned(Tensor), 'Audit tensor is nil.');
  Require(Assigned(Tensor.data) and Assigned(Tensor.shape), 'Audit tensor data or shape is nil.');
  Require((Tensor.device.device_type = kDLCPU) and (Tensor.device.device_id = 0),
    'Audit snapshots require CPU device 0.');
  Require(TTVMRuntime.CheckTensor(Tensor, [kDLFloat], 32), 'Audit snapshots require scalar Float32.');
  Require((Tensor.ndim = Length(ExpectedShape)) and (Tensor.ndim > 0), 'Audit tensor rank changed.');
  Require(Tensor.byte_offset = 0, 'Audit snapshots require zero byte offset.');
  Shape := TTVMRuntime.GetTensorShape(Tensor);
  Elements := 1;
  for I := 0 to High(Shape) do
  begin
    Require((Shape[I] > 0) and (Shape[I] = ExpectedShape[I]), 'Audit tensor shape changed.');
    Require(Elements <= (MaximumBytes div SizeOf(Single)) div Shape[I],
      'Audit tensor exceeds the 16 MiB snapshot limit.');
    Elements := Elements * Shape[I];
  end;
  ExpectedStride := 1;
  for I := High(Shape) downto 0 do
  begin
    if Assigned(Tensor.strides) then
      Require(Tensor.strides[I] = ExpectedStride, 'Audit tensor is not compact row-major.');
    ExpectedStride := ExpectedStride * Shape[I];
  end;
  ByteCount := Integer(Elements * SizeOf(Single));
  Require(UInt64(ByteCount - 1) <= UInt64(High(NativeUInt) - NativeUInt(Tensor.data)),
    'Audit tensor byte span overflows address space.');
  SetLength(Bytes, ByteCount);
  System.Move(Tensor.data^, Bytes[0], ByteCount);
  SHA256 := SHA256Bytes(Bytes);
  Handle := CreateFileW(PWideChar(SafeFile(Directory, FileName)), GENERIC_WRITE, 0, nil,
    CREATE_NEW, FILE_ATTRIBUTE_NORMAL, 0);
  if Handle = INVALID_HANDLE_VALUE then RaiseLastOSError;
  try
    Stream := THandleStream.Create(Handle);
    try
      Stream.WriteBuffer(Bytes[0], Length(Bytes));
    finally
      Stream.Free;
    end;
  finally
    CloseHandle(Handle);
  end;
  Result := TJSONObject.Create;
  try
    Result.AddPair('role', Role);
    Result.AddPair('file', FileName);
    Result.AddPair('format', 'raw-contiguous-f32-le');
    Result.AddPair('dtype', 'float32');
    Result.AddPair('device', 'cpu');
    Result.AddPair('device_id', JSONInteger(0));
    Result.AddPair('lanes', JSONInteger(Tensor.dtype.lanes));
    Result.AddPair('byte_offset', JSONInteger(0));
    Result.AddPair('bytes', JSONInteger(ByteCount));
    Result.AddPair('elements', JSONInteger(Elements));
    Result.AddPair('sha256', SHA256);
    ShapeJSON := TJSONArray.Create;
    Result.AddPair('shape', ShapeJSON);
    for I := 0 to High(Shape) do ShapeJSON.Add(Shape[I]);
    if Assigned(Tensor.strides) then
    begin
      StridesJSON := TJSONArray.Create;
      Result.AddPair('strides', StridesJSON);
      for I := 0 to High(Shape) do StridesJSON.Add(Tensor.strides[I]);
    end
    else
      Result.AddPair('strides', TJSONNull.Create);
  except
    Result.Free;
    raise;
  end;
end;

function CaptureCpuVisionAudit(const Context: Pointer; const InputTensor: PDLTensor;
  const OutputTensors: TArray<PDLTensor>; const Examined: Int64; const Error: PString): Boolean;
var
  State: PCpuVisionAudit;
  Root, Row: TJSONObject;
  Outputs: TJSONArray;
  Elements, OutputElements: Int64;
  I: Integer;
  ExpectedShape: TArray<Int64>;
  Role: string;
begin
  Result := False;
  try
    Require(Assigned(Context), 'Missing CPU audit context.');
    State := PCpuVisionAudit(Context);
    Require(not State.Captured, 'CPU audit observer was invoked more than once.');
    Require((Length(OutputTensors) = 1) or
      ((State.Profile.Adapter = 'rtdetr') and (Length(OutputTensors) = 2)),
      'CPU audit output count changed.');
    if State.Profile.Adapter = 'rtdetr' then
      Require(Length(OutputTensors) = 2, 'The audited RT-DETR profile requires two raw outputs.');
    Root := TJSONObject.Create;
    try
      Root.AddPair('schema', 'crosspascal.cpu-vision-audit.v1');
      Root.AddPair('status', 'RAW_SNAPSHOT_CAPTURED');
      Root.AddPair('profile', State.Profile.ID);
      Root.AddPair('directory', State.Directory);
      Root.AddPair('source_sha256', State.Binding.SourceSHA256);
      Root.AddPair('model_sha256', State.Binding.ModelSHA256);
      Root.AddPair('manifest_sha256', State.Binding.ManifestSHA256);
      Root.AddPair('profile_catalog_sha256', State.Profile.CatalogSHA256);
      Root.AddPair('image_sha256', State.Binding.InputSHA256);
      Root.AddPair('input_name', State.Binding.InputName);
      Root.AddPair('input', SaveAuditTensor(InputTensor,
        [1, 3, State.Profile.InputSize, State.Profile.InputSize],
        State.Directory, 'input.f32', 'input', Elements));
      Outputs := TJSONArray.Create;
      Root.AddPair('outputs', Outputs);
      OutputElements := 0;
      for I := 0 to High(OutputTensors) do
      begin
        if State.Profile.TaskKind = 'classification' then
        begin
          ExpectedShape := TArray<Int64>.Create(1, 1000);
          Role := 'scores';
        end
        else if State.Profile.Adapter = 'rtdetr' then
          begin
            if I = 0 then
            begin
              ExpectedShape := TArray<Int64>.Create(1, 300, 80);
              Role := 'logits';
            end
            else begin
              ExpectedShape := TArray<Int64>.Create(1, 300, 4);
              Role := 'pred_boxes';
            end;
          end
          else begin
            ExpectedShape := TArray<Int64>.Create(1, 84, 8400);
            Role := 'dense_detections';
          end;
        Row := SaveAuditTensor(OutputTensors[I], ExpectedShape, State.Directory,
          'output-' + IntToStr(I) + '.f32', Role, Elements);
        Outputs.AddElement(Row);
        Row.AddPair('output_index', JSONInteger(I));
        Inc(OutputElements, Elements);
      end;
      Require(Examined = OutputElements,
        'The existing runtime finite scan did not examine every audited output scalar.');
      Root.AddPair('output_finiteness_method', 'TTVMRuntime.TensorsAreFinite before this observer');
      Root.AddPair('output_finite_elements_examined', JSONInteger(Examined));
      Root.AddPair('output_elements', JSONInteger(OutputElements));
      Root.AddPair('input_finiteness', 'Not separately scanned by the runtime observer.');
      Root.AddPair('inference_calls_observed', JSONInteger(1));
      State.JSONText := Root.ToJSON;
      WriteCpuVisionReport(SafeFile(State.Directory, 'audit.json'), State.JSONText);
      State.Captured := True;
    finally
      Root.Free;
    end;
    Result := True;
  except
    on E: Exception do
      if Assigned(Error) then Error^ := 'CPU audit capture failed: ' + E.Message;
  end;
end;

function ResultJSON(const Value: TCpuVisionResult; const Profile: TCpuVisionProfile): string;
var
  Root, Row: TJSONObject;
  Classes, Detections, Box, RuntimeFiles: TJSONArray;
  Item: TClassification;
  DetectionItem: TDetection;
  I, Last: Integer;
begin
  Root := TJSONObject.Create;
  try
    Root.AddPair('schema', 'crosspascal.' + Value.Device + '-vision-result.v1');
    Root.AddPair('status', UpperCase(Value.Device) + '_INFERENCE_COMPLETED');
    Root.AddPair('release_acceptance', JSONBoolean(False));
    Root.AddPair('profile', Profile.ID);
    Root.AddPair('profile_catalog_sha256', Profile.CatalogSHA256);
    Root.AddPair('task_kind', Profile.TaskKind);
    Root.AddPair('device', Value.Device);
    Root.AddPair('device_id', JSONInteger(0));
    if Value.Device = 'cuda' then
    begin
      Root.AddPair('device_name', Value.DeviceName);
      Root.AddPair('cuda_arch', Value.CudaArchitecture);
    end;
    Root.AddPair('source_sha256', Value.SourceSHA256);
    Root.AddPair('model_sha256', Value.ModelSHA256);
    Root.AddPair('manifest_sha256', Value.ManifestSHA256);
    Root.AddPair('input_name', Value.InputName);
    Root.AddPair('image_sha256', Value.InputSHA256);
    Root.AddPair('width', JSONInteger(Value.Width));
    Root.AddPair('height', JSONInteger(Value.Height));
    Root.AddPair('elapsed_ms', JSONUnsigned(Value.ElapsedMilliseconds));
    Root.AddPair('runtime_directory', Value.RuntimeDirectory);
    Root.AddPair('preprocessing',
      'Direct resize: area if either dimension shrinks, cubic otherwise; BGR to RGB, divide by 255; profile RGB normalization.');
    Root.AddPair('classification_scores', 'Unmodified model outputs; not automatically probabilities.');
    if Value.AuditJSONText <> '' then
      Root.AddPair('inference_audit', ParseObject(Value.AuditJSONText));
    RuntimeFiles := TJSONArray.Create;
    Root.AddPair('runtime_files', RuntimeFiles);
    Last := 2;
    if Value.Device = 'cuda' then Last := 3;
    for I := 0 to Last do
    begin
      Row := TJSONObject.Create;
      Row.AddPair('file', RuntimeNames[I]);
      Row.AddPair('sha256', RuntimeHashes[I]);
      RuntimeFiles.AddElement(Row);
    end;
    Classes := TJSONArray.Create;
    Root.AddPair('classifications', Classes);
    for Item in Value.Classifications do
    begin
      Row := TJSONObject.Create;
      Row.AddPair('class_id', JSONInteger(Item.ClassId));
      Row.AddPair('class_name', Item.ClassName);
      Row.AddPair('score', JSONFloat(Item.Score));
      Classes.AddElement(Row);
    end;
    Detections := TJSONArray.Create;
    Root.AddPair('detections', Detections);
    for DetectionItem in Value.Detections do
    begin
      Row := TJSONObject.Create;
      Row.AddPair('class_id', JSONInteger(DetectionItem.ClassId));
      Row.AddPair('class_name', DetectionItem.ClassName);
      Row.AddPair('score', JSONFloat(DetectionItem.Score));
      Box := TJSONArray.Create;
      Box.Add(DetectionItem.Box.Left);
      Box.Add(DetectionItem.Box.Top);
      Box.Add(DetectionItem.Box.Right);
      Box.Add(DetectionItem.Box.Bottom);
      Row.AddPair('xyxy_pixels', Box);
      Detections.AddElement(Row);
    end;
    Result := Root.ToJSON;
  finally
    Root.Free;
  end;
end;

function CompareClassifications({$IFDEF FPC}constref{$ELSE}const{$ENDIF} A, B: TClassification): Integer;
begin
  if A.Score > B.Score then
    Result := -1
  else if A.Score < B.Score then
      Result := 1
    else
      Result := A.ClassId - B.ClassId;
end;

function RunVisionBundle(const Profile: TCpuVisionProfile;
  const BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256: string; const Image: IMat;
  const Progress: TCpuVisionProgress; const AuditDirectory: string; const AllowCuda: Boolean): TCpuVisionResult;
const
  MiB = 1024 * 1024;
var
  ManifestLease, ModelLease, SidecarLease, ImageLease: TFileStream;
  Root: TJSONObject;
  Descriptor: TModelDescriptor;
  ModelName, ModelHash, SourceHash, ModelPath, Sidecar, InputName, DeviceName: string;
  ErrorText: {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
  Loaded, Entered: Boolean;
  Device: TDLDeviceType;
  Cuda: TCudaExecution;
  Workspace: TCudaWorkspace;
  FreeBytes, WorkspaceBytes: UInt64;
  Scores: TArray<Single>;
  Labels: TArray<Integer>;
  Places: TArray<TRect>;
  Shape: TArray<Int64>;
  Started: UInt64;
  MemoryStatus: TMemoryStatusEx;
  I: Integer;
  Audit: TCpuVisionAudit;
  AuditID: TGUID;

  procedure Say(const Text: string);
  begin
    if Assigned(Progress) then Progress(Text);
  end;

begin
  Result := Default(TCpuVisionResult);
  Audit := Default(TCpuVisionAudit);
  Require((Image <> nil) and not Image.IsEmpty and (Image.Channels = 3),
    'Open a non-empty color image before inference.');
  Require((Image.Width > 0) and (Image.Height > 0) and
    (Int64(Image.Width) * Image.Height <= 40000000), 'The image exceeds the 40 megapixel limit.');
  Require(IsSHA256(ImageSHA256), 'The decoded image must have a recorded source SHA-256.');
  MemoryStatus := Default(TMemoryStatusEx);
  MemoryStatus.dwLength := SizeOf(MemoryStatus);
  Require(GlobalMemoryStatusEx(MemoryStatus), 'Cannot read available physical memory.');
  Require(MemoryStatus.ullAvailPhys >= UInt64(1024) * 1024 * 1024,
    'At least 1 GiB of free physical memory is required before loading a CPU model.');
  ManifestLease := nil;
  ModelLease := nil;
  SidecarLease := nil;
  ImageLease := nil;
  Root := nil;
  Loaded := False;
  Entered := False;
  Cuda := nil;
  Workspace := nil;
  try
    Say('Checking the selected bundle and source identity...');
    ManifestLease := TFileStream.Create(SafeFile(BundleDirectory, 'manifest.json'),
      fmOpenRead or fmShareDenyWrite);
    Result.ManifestSHA256 := HashStream(ManifestLease);
    Root := ParseObject(ReadTextStream(ManifestLease, 1024 * 1024));
    ValidateManifest(Root, Profile, AllowCuda, ModelName, ModelHash, SourceHash, InputName,
      Result.Device, Result.CudaArchitecture);
    Result.InputName := InputName;
    ModelPath := SafeFile(BundleDirectory, ModelName);
    ModelLease := TFileStream.Create(ModelPath, fmOpenRead or fmShareDenyWrite);
    Require((ModelLease.Size > 0) and (ModelLease.Size <= Int64(512) * 1024 * 1024),
      'The model DLL must be between 1 byte and 512 MiB.');
    Require(SameText(HashStream(ModelLease), ModelHash), 'Model-library SHA-256 mismatch.');
    SidecarLease := TFileStream.Create(ModelPath + '.inputs', fmOpenRead or fmShareDenyWrite);
    Sidecar := Trim(ReadTextStream(SidecarLease, 4096));
    Require(Sidecar = '0'#9 + InputName + #9 + '1,3,' +
      IntToStr(Profile.InputSize) + ',' + IntToStr(Profile.InputSize),
      'The model must have exactly the selected single input.');
    ImageLease := TFileStream.Create(ImagePath, fmOpenRead or fmShareDenyWrite);
    Result.InputSHA256 := HashStream(ImageLease);
    Require(SameText(Result.InputSHA256, ImageSHA256),
      'The image file changed after it was decoded. Open it again before inference.');
    Result.ProfileID := Profile.ID;
    Result.SourceSHA256 := SourceHash;
    Result.ModelSHA256 := ModelHash;
    Result.RuntimeDirectory := ExpandFileName(RuntimeDirectory);
    Result.Width := Image.Width;
    Result.Height := Image.Height;
    if AuditDirectory <> '' then
    begin
      Require(DirectoryExists(AuditDirectory), 'The explicit audit parent directory must already exist.');
      Require(CreateGUID(AuditID) = 0, 'Cannot create a unique CPU audit directory name.');
      Audit.Directory := IncludeTrailingPathDelimiter(ExpandFileName(AuditDirectory)) +
        Profile.ID + '-' + GUIDToString(AuditID);
      Require(CreateDir(Audit.Directory), 'Cannot create a new CPU audit directory.');
      Audit.Profile := Profile;
      Audit.Binding := Result;
    end;
    Device := kDLCPU;
    DeviceName := 'CPU';
    if Result.Device = 'cuda' then
    begin
      Device := kDLCUDA;
      Cuda := TCudaExecution.Create;
      Result.DeviceName := Cuda.DeviceName;
      DeviceName := 'CUDA (' + Result.DeviceName + ')';
      Cuda.RequireArchitecture(Result.CudaArchitecture);
      Cuda.Enter;
      Entered := True;
      FreeBytes := Cuda.FreeBytes;
      Require(FreeBytes >= UInt64(ModelLease.Size) + 256 * MiB,
        'Insufficient free CUDA memory to load the vision bundle.');
      WorkspaceBytes := FreeBytes - UInt64(ModelLease.Size) - 128 * MiB;
      if WorkspaceBytes > 1024 * MiB then WorkspaceBytes := 1024 * MiB;
    end;
    Say('Loading the pinned TVM runtime on ' + DeviceName + '...');
    InitializeVisionRuntime(RuntimeDirectory, Device = kDLCUDA);
    ErrorText := '';
    Loaded := TTVMRuntime.LoadModel(ModelPath, 'so', Descriptor, Device, 0, 'main', @ErrorText);
    Require(Loaded, 'Cannot load the model on ' + DeviceName + ': ' + ErrorText);
    Require(Descriptor.Device.device_type = Device, 'The model did not select the requested device.');
    if Cuda <> nil then Workspace := TCudaWorkspace.Create(Cuda, ModelPath, WorkspaceBytes);
    if AuditDirectory <> '' then
    begin
      Descriptor.InferenceObserver := CaptureCpuVisionAudit;
      Descriptor.InferenceObserverContext := @Audit;
    end;
    Shape := TArray<Int64>.Create(1, 3, Profile.InputSize, Profile.InputSize);
    TTVMRuntime.SetInputSpec(Descriptor, Shape, kDLFloat, 32, 1);
    Say('Running ' + Profile.Title + ' on ' + DeviceName + '...');
    Started := GetTickCount64;
    if Workspace <> nil then Workspace.BeginRun;
    try
    if Profile.TaskKind = 'classification' then
    begin
      if not TYolo11.Categorize(Descriptor, Image, Ord(kDLCPU), 0, Scores,
        InputName, @ErrorText, Profile.Mean, Profile.Std) then
          raise Exception.Create('Classification failed: ' + ErrorText);
      Require(Length(Scores) = Profile.ClassCount, 'Classifier output class count changed.');
      SetLength(Result.Classifications, Length(Scores));
      for I := 0 to High(Scores) do
      begin
        Require(not IsNan(Scores[I]) and not IsInfinite(Scores[I]), 'Non-finite classifier output.');
        Result.Classifications[I] := Classification(I, '', Scores[I]);
      end;
      {$IFDEF FPC}
      TArrayHelper<TClassification>.Sort(Result.Classifications,
        TComparer<TClassification>.Construct(CompareClassifications));
      {$ELSE}
      TArray.Sort<TClassification>(Result.Classifications,
        TComparer<TClassification>.Construct(CompareClassifications));
      {$ENDIF}
      if Length(Result.Classifications) > 5 then SetLength(Result.Classifications, 5);
    end
    else begin
      if Profile.Adapter = 'rtdetr' then
      begin
        if not TRtDetr.Detect(Descriptor, Image, Ord(kDLCPU), 0, Profile.ScoreThreshold,
          Scores, Labels, Places, InputName, @ErrorText) then
            raise Exception.Create('RT-DETR inference failed: ' + ErrorText);
      end
      else if not TYolo11.Detect(Descriptor, Image, Ord(kDLCPU), 0,
        Profile.ScoreThreshold, Profile.NMSThreshold, Scores, Labels, Places,
        InputName, @ErrorText, ylDense) then
          raise Exception.Create('YOLO inference failed: ' + ErrorText);
      Require((Length(Scores) = Length(Labels)) and (Length(Scores) = Length(Places)),
        'Detector output arrays have different lengths.');
      SetLength(Result.Detections, Length(Scores));
      for I := 0 to High(Scores) do
      begin
        Require(not IsNan(Scores[I]) and not IsInfinite(Scores[I]) and
          (Scores[I] >= 0) and (Scores[I] <= 1), 'Invalid detector score.');
        Require((Labels[I] >= 0) and (Labels[I] < Profile.ClassCount), 'Invalid detector class.');
        Result.Detections[I] := Detection(TRectI.Create(Places[I].Left, Places[I].Top,
          Places[I].Right, Places[I].Bottom), Scores[I], Labels[I], '');
      end;
    end;
    finally
      if Workspace <> nil then
      begin
        Workspace.EndRun;
        Workspace.CheckError;
      end;
    end;
    if AuditDirectory <> '' then
    begin
      Require(Audit.Captured, 'The requested CPU audit snapshot was not observed.');
      Result.AuditJSONText := Audit.JSONText;
    end;
    Result.ElapsedMilliseconds := GetTickCount64 - Started;
  finally
    if Loaded then
    begin
      Descriptor.InferenceObserver := nil;
      Descriptor.InferenceObserverContext := nil;
      TTVMRuntime.FreeModel(Descriptor);
    end;
    Workspace.Free;
    if Entered then Cuda.Leave;
    Cuda.Free;
    Root.Free;
    ImageLease.Free;
    SidecarLease.Free;
    ModelLease.Free;
    ManifestLease.Free;
  end;
  Result.JSONText := ResultJSON(Result, Profile);
  Say('Inference on ' + DeviceName + ' completed. The model has been released.');
end;

function RunCpuVision(const Profile: TCpuVisionProfile;
  const BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256: string; const Image: IMat;
  const Progress: TCpuVisionProgress; const AuditDirectory: string): TCpuVisionResult;
begin
  Result := RunVisionBundle(Profile, BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256,
    Image, Progress, AuditDirectory, False);
end;

function RunVision(const Profile: TCpuVisionProfile;
  const BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256: string; const Image: IMat;
  const Progress: TCpuVisionProgress): TCpuVisionResult;
begin
  Result := RunVisionBundle(Profile, BundleDirectory, RuntimeDirectory, ImagePath, ImageSHA256,
    Image, Progress, '', True);
end;

procedure WriteCpuVisionReport(const Path, JSONText: string);
var
  Handle: THandle;
  Stream: THandleStream;
  Bytes: TBytes;
begin
  Handle := CreateFileW(PWideChar(ExpandFileName(Path)), GENERIC_WRITE, 0, nil,
    CREATE_NEW, FILE_ATTRIBUTE_NORMAL, 0);
  if Handle = INVALID_HANDLE_VALUE then RaiseLastOSError;
  try
    Stream := THandleStream.Create(Handle);
    try
      Bytes := TEncoding.UTF8.GetBytes(JSONText + sLineBreak);
      if Length(Bytes) > 0 then Stream.WriteBuffer(Bytes[0], Length(Bytes));
    finally
      Stream.Free;
    end;
  finally
    CloseHandle(Handle);
  end;
end;

procedure ReleaseRuntimeLeases;
var
  I: Integer;
begin
  for I := 0 to High(RuntimeLeases) do FreeAndNil(RuntimeLeases[I]);
end;

initialization
  InitializeCriticalSection(VisionRuntimeLock);

finalization
  ReleaseRuntimeLeases;
  DeleteCriticalSection(VisionRuntimeLock);

end.
