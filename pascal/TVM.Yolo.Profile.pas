{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Profile                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Profile;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.Json,
  TVM.Yolo.Detections, TVM.Yolo.Classification, TVM.Yolo.Masks,
  TVM.Yolo.Transform, TVM.Yolo.PixelPreprocess;

type
  EYoloProfile = class(EArgumentException);
  TYoloTask = (ytClassification, ytDetection, ytInstances, ytPose, ytOriented, ytSemantic, ytDepth);
  TYoloPreprocessing = (ypPillowClassification, ypClassicClassification, ypLetterbox, ypScaleFill);
  TYoloTensorType = (yttFloat32, yttUInt8);
  TYoloNmsKind = (ynGreedyIoU, ynProbIoUFast, ynSelectedInGraph);

  TYoloOutputProfile = record
    Name: string;
    Shape: TArray<Int64>;
    DataType: TYoloTensorType;
    function Elements: Int64;
    function ByteCount: Int64;
  end;

  { A model contract, independent of GPU marketing names and model sizes.
    Validation is repeated by a model loader before allocating tensors.
    A bundle must additionally bind this contract to its artifact and source
    digests; parsing a catalog alone does not authenticate a model binary. }
  TYoloProfile = record
    ID: string;
    Family: string;
    InputName: string;
    SourceSHA256: string;
    RecipeSHA256: string;
    RecipeFile: string;
    Task: TYoloTask;
    Preprocessing: TYoloPreprocessing;
    InputWidth: Integer;
    InputHeight: Integer;
    ClassNames: TArray<string>;
    Mean: TArray<Single>;
    Std: TArray<Single>;
    Outputs: TArray<TYoloOutputProfile>;
    ScaleUp: Boolean;
    Padding: Byte;
    Detections: TYoloDetectionOptions;
    NmsKind: TYoloNmsKind;
    BoxEncoding: string;
    CoordinateMode: TYoloCoordinateMode;
    ClassScoreKind: TYoloClassScoreKind;
    TopK: Integer;
    InstanceMaskKind: TYoloInstanceMaskKind;
    PrototypeStride: Integer;
    KeypointCount: Integer;
    KeypointDimensions: Integer;
    ClassOffset: Single;
    DepthUnits: string;
    DepthAlignCorners: Boolean;
    procedure Validate;
    function ClassCount: Integer;
    function BoxCount: Integer;
    function FeatureCount: Integer;
    function PixelPlan(const OriginalWidth, OriginalHeight: Integer): TYoloPixelPlan;
    function ImageTransform(const OriginalWidth, OriginalHeight: Integer): TYoloImageTransform;
  end;
  TYoloProfiles = TArray<TYoloProfile>;

function ReadYoloProfile(const Value: TJSONValue): TYoloProfile;
function ParseYoloProfiles(const JSONText: string): TYoloProfiles;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF};

procedure Require(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise EYoloProfile.Create(MessageText);
end;

function Field(const Obj: TJSONObject; const Name: string): TJSONValue;
begin
  Result := Obj.GetValue(Name);
  Require(Result <> nil, 'Missing YOLO profile field: ' + Name);
end;

function AsObject(const Value: TJSONValue): TJSONObject;
begin
  Require(Value is TJSONObject, 'Expected a YOLO profile object');
  Result := TJSONObject(Value);
end;

function AsArray(const Value: TJSONValue): TJSONArray;
begin
  Require(Value is TJSONArray, 'Expected a YOLO profile array');
  Result := TJSONArray(Value);
end;

function Text(const Obj: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require(Value is TJSONString, 'Expected a YOLO profile string: ' + Name);
  Result := JSONValueText(Value);
end;

function Number(const Value: TJSONValue): Single;
var
  Parsed: Double;
begin
  Require((Value is TJSONNumber) and TryStrToFloat(JSONValueText(Value), Parsed, JSONInvariantFormat),
    'Expected a YOLO profile number');
  Require(not IsNan(Parsed) and not IsInfinite(Parsed) and (Abs(Parsed) <= MaxSingle),
    'Nonfinite or oversized YOLO profile number');
  Result := Parsed;
end;

function IntegerValue(const Value: TJSONValue): Integer;
begin
  Require((Value is TJSONNumber) and TryStrToInt(JSONValueText(Value), Result),
    'Expected a YOLO profile integer');
end;

function BooleanValue(const Obj: TJSONObject; const Name: string): Boolean;
var
  Value: TJSONValue;
begin
  Value := Field(Obj, Name);
  Require(JSONIsBoolean(Value), 'Expected a YOLO profile Boolean: ' + Name);
  Result := JSONIsTrue(Value);
end;

procedure UniqueKeys(const Value: TJSONValue);
var
  Obj: TJSONObject;
  Rows: TJSONArray;
  I, J: Integer;
begin
  if Value is TJSONObject then
  begin
    Obj := TJSONObject(Value);
    for I := 0 to Obj.Count - 1 do
    begin
      for J := 0 to I - 1 do
        Require(JSONObjectName(Obj, I) <> JSONObjectName(Obj, J), 'Duplicate YOLO profile field');
      UniqueKeys(JSONObjectValue(Obj, I));
    end;
  end
  else if Value is TJSONArray then
  begin
    Rows := TJSONArray(Value);
    for I := 0 to Rows.Count - 1 do UniqueKeys(Rows.Items[I]);
  end;
end;

function Shape(const Value: TJSONValue): TArray<Int64>;
var
  Rows: TJSONArray;
  I: Integer;
begin
  Rows := AsArray(Value);
  Require(Rows.Count in [1..4], 'Unsupported YOLO tensor rank');
  SetLength(Result, Rows.Count);
  for I := 0 to High(Result) do
  begin
    Result[I] := IntegerValue(Rows.Items[I]);
    Require(Result[I] > 0, 'YOLO tensor dimensions must be positive and static');
  end;
end;

function Singles(const Value: TJSONValue): TArray<Single>;
var
  Rows: TJSONArray;
  I: Integer;
begin
  Rows := AsArray(Value);
  Require(Rows.Count = 3, 'YOLO RGB normalization requires three channels');
  SetLength(Result, 3);
  for I := 0 to 2 do Result[I] := Number(Rows.Items[I]);
end;

procedure RequireName(const Name: string);
begin
  Require((Name <> '') and (Pos(#0, Name) = 0) and (Pos(#10, Name) = 0) and
    (Pos(#13, Name) = 0), 'Invalid YOLO profile name');
end;

procedure RequireSHA256(const Value: string);
var
  C: Char;
begin
  Require(Length(Value) = 64, 'Invalid YOLO profile SHA256');
  for C in Value do Require(CharInSet(C, ['0'..'9', 'a'..'f']), 'Invalid YOLO profile SHA256');
end;

procedure RequireRecipeFile(const Value: string);
var
  Normalized, Part: string;
  I, Start: Integer;
begin
  RequireName(Value);
  Normalized := StringReplace(Value, '\', '/', [rfReplaceAll]);
  Require((Normalized[1] <> '/') and (Pos(':', Normalized) = 0), 'YOLO recipe must be relative');
  Start := 1;
  for I := 1 to Length(Normalized) + 1 do
    if (I > Length(Normalized)) or (Normalized[I] = '/') then
    begin
      Part := Copy(Normalized, Start, I - Start);
      Require((Part <> '') and (Part <> '.') and (Part <> '..'), 'Invalid YOLO recipe path segment');
      Start := I + 1;
    end;
end;

function TYoloOutputProfile.Elements: Int64;
var
  Dimension: Int64;
begin
  Require((Length(Shape) >= 1) and (Length(Shape) <= 4), 'Invalid YOLO output rank');
  Result := 1;
  for Dimension in Shape do
  begin
    Require((Dimension > 0) and (Dimension <= High(Integer)) and
      (Result <= High(Integer) div Dimension), 'YOLO output dimensions exceed addressable storage');
    Result := Result * Dimension;
  end;
end;

function TYoloOutputProfile.ByteCount: Int64;
var
  Width: Integer;
begin
  Require(DataType in [yttFloat32, yttUInt8], 'Invalid YOLO output data type');
  Width := 1;
  if DataType = yttFloat32 then Width := SizeOf(Single);
  Result := Elements;
  Require(Result <= High(Integer) div Width, 'YOLO output byte count exceeds address range');
  Result := Result * Width;
end;

function TYoloProfile.ClassCount: Integer;
begin
  Result := Length(ClassNames);
end;

function TYoloProfile.BoxCount: Integer;
begin
  Require(Task in [ytDetection, ytInstances, ytPose, ytOriented], 'YOLO task has no box rows');
  Require((Length(Outputs) > 0) and (Length(Outputs[0].Shape) = 3), 'Invalid YOLO box tensor');
  if Detections.Layout = yblChannelsFirst then
    Result := Outputs[0].Shape[2]
  else
    Result := Outputs[0].Shape[1];
end;

function TYoloProfile.FeatureCount: Integer;
begin
  Require(Task in [ytDetection, ytInstances, ytPose, ytOriented], 'YOLO task has no box features');
  Require((Length(Outputs) > 0) and (Length(Outputs[0].Shape) = 3), 'Invalid YOLO feature tensor');
  if Detections.Layout = yblChannelsFirst then
    Result := Outputs[0].Shape[1]
  else
    Result := Outputs[0].Shape[2];
end;

function TYoloProfile.PixelPlan(const OriginalWidth, OriginalHeight: Integer): TYoloPixelPlan;
var
  Mode: TYoloPixelMode;
begin
  Validate;
  case Preprocessing of
    ypLetterbox: Mode := ypmLetterbox;
    ypScaleFill: Mode := ypmScaleFill;
    ypClassicClassification: Mode := ypmClassicClassification;
  else
    raise EYoloProfile.Create('This YOLO profile requires the Pillow classification transform');
  end;
  Result := TYoloPixelPlan.Create(Mode, OriginalWidth, OriginalHeight,
    InputWidth, InputHeight, ScaleUp, Padding);
end;

function TYoloProfile.ImageTransform(const OriginalWidth, OriginalHeight: Integer): TYoloImageTransform;
begin
  Validate;
  Require(Preprocessing = ypLetterbox, 'Only a letterboxed YOLO profile has this coordinate transform');
  Result := TYoloImageTransform.Letterbox(OriginalWidth, OriginalHeight,
    InputWidth, InputHeight, ScaleUp, CoordinateMode);
end;

procedure TYoloProfile.Validate;
var
  I, J: Integer;
  Features, TotalBytes: Int64;
begin
  RequireName(ID); RequireName(Family); RequireName(InputName);
  RequireSHA256(SourceSHA256); RequireSHA256(RecipeSHA256); RequireRecipeFile(RecipeFile);
  Require(Task in [ytClassification, ytDetection, ytInstances, ytPose, ytOriented, ytSemantic, ytDepth],
    'Invalid YOLO task');
  Require(Preprocessing in [ypPillowClassification, ypClassicClassification, ypLetterbox, ypScaleFill],
    'Invalid YOLO preprocessing');
  Require((InputWidth > 0) and (InputWidth <= 4096) and (InputHeight > 0) and (InputHeight <= 4096),
    'Invalid YOLO input dimensions');
  Require((ClassCount > 0) and (Length(Mean) = 3) and (Length(Std) = 3),
    'Invalid YOLO classes or normalization');
  for I := 0 to High(ClassNames) do RequireName(ClassNames[I]);
  for I := 0 to 2 do
    Require(not IsNan(Mean[I]) and not IsInfinite(Mean[I]) and not IsNan(Std[I]) and
      not IsInfinite(Std[I]) and (Std[I] > 0), 'Invalid YOLO normalization');
  Require(Length(Outputs) in [1, 2], 'Invalid YOLO output count');
  TotalBytes := 0;
  for I := 0 to High(Outputs) do
  begin
    RequireName(Outputs[I].Name);
    Inc(TotalBytes, Outputs[I].ByteCount);
    Require(TotalBytes <= High(Integer), 'YOLO outputs exceed addressable storage');
    Require(Outputs[I].Shape[0] = 1, 'YOLO outputs require batch one');
    for J := 0 to I - 1 do
      Require(Outputs[I].Name <> Outputs[J].Name, 'Duplicate YOLO output name');
    if Task = ytSemantic then
      Require(Outputs[I].DataType in [yttUInt8, yttFloat32], 'Expected a YOLO class map or Float32 logits')
    else
      Require(Outputs[I].DataType = yttFloat32, 'Expected Float32 YOLO outputs');
  end;
  if Task = ytClassification then
  begin
    Require((Preprocessing in [ypPillowClassification, ypClassicClassification]) and
      (Length(Outputs) = 1) and (Length(Outputs[0].Shape) = 2) and
      (Outputs[0].Shape[1] = ClassCount) and (TopK > 0) and (TopK <= ClassCount) and
      (ClassScoreKind in [ycsProbabilities, ycsLogits]), 'Invalid YOLO classification contract');
    Exit;
  end;
  if Task in [ytSemantic, ytDepth] then
  begin
    Require(Length(Outputs) = 1, 'YOLO map task requires one output');
    if Task = ytSemantic then
    begin
      Require((Preprocessing = ypLetterbox) and (ClassCount <= 256), 'Invalid YOLO semantic preprocessing');
      if Outputs[0].DataType = yttUInt8 then
        Require(Length(Outputs[0].Shape) = 3, 'Invalid YOLO semantic class map contract')
      else
        Require((ClassCount > 1) and (Length(Outputs[0].Shape) = 4) and
          (Outputs[0].Shape[1] = ClassCount), 'Invalid YOLO semantic logit contract');
    end
    else
      Require(
        (Preprocessing = ypScaleFill) and (ClassCount = 1) and (DepthUnits = 'meters') and DepthAlignCorners and (Length(Outputs[0].Shape) = 4) and (Outputs[0].Shape[1] = 1),
        'Invalid YOLO depth contract'
      );
    Exit;
  end;
  Require((Preprocessing = ypLetterbox) and
    (CoordinateMode in [ycmRoundedResizeAxes, ycmLegacySymmetricGain, ycmLegacyRoundedGain]),
    'Invalid YOLO box preprocessing or coordinate policy');
  Require((Detections.Layout in [yblChannelsFirst, yblObjectnessRows, yblEndToEndRows]) and
    (Detections.ClassCount = ClassCount) and (Detections.ExtraCount >= 0) and
    not IsNan(Detections.Confidence) and not IsInfinite(Detections.Confidence) and
    (Detections.Confidence >= 0) and (Detections.Confidence <= 1) and
    not IsNan(Detections.NmsIoU) and not IsInfinite(Detections.NmsIoU) and
    (Detections.NmsIoU >= 0) and (Detections.NmsIoU <= 1) and
    (Detections.MaxDetections > 0) and (Detections.MaxNmsCandidates >= Detections.MaxDetections) and
    (not Detections.ApplyObjectnessToExtra or (Detections.Layout = yblObjectnessRows)),
    'Invalid YOLO detection contract');
  Require(Length(Outputs[0].Shape) = 3, 'Invalid YOLO box output rank');
  if Detections.Layout = yblEndToEndRows then
    Features := 6 + Int64(Detections.ExtraCount)
  else
    Features := 4 + Int64(Ord(Detections.Layout = yblObjectnessRows)) + ClassCount +
      Detections.ExtraCount;
  Require((Features = FeatureCount) and (BoxCount > 0), 'YOLO box output differs from its declared columns');
  if Detections.Layout = yblEndToEndRows then
    Require((NmsKind = ynSelectedInGraph) and (Detections.MaxDetections <= BoxCount),
      'End-to-end YOLO output must not apply NMS again')
  else if Task = ytOriented then
    Require(NmsKind = ynProbIoUFast, 'Expected YOLO oriented fast ProbIoU NMS')
  else
    Require(NmsKind = ynGreedyIoU, 'Expected YOLO class-aware IoU NMS');
  if Task = ytOriented then
    Require(BoxEncoding = 'xywh_angle', 'Expected YOLO XYWH and angle')
  else if Detections.Layout = yblEndToEndRows then
    Require(BoxEncoding = 'xyxy', 'Expected YOLO XYXY columns')
  else
    Require(BoxEncoding = 'xywh', 'Expected YOLO XYWH columns');
  if Task = ytInstances then
  begin
    Require(
      (Length(Outputs) = 2) and (Detections.ExtraCount > 0) and (Length(Outputs[1].Shape) = 4) and (Outputs[1].Shape[1] = Detections.ExtraCount) and (PrototypeStride > 0) and ((PrototypeStride and (PrototypeStride - 1)) = 0) and (Outputs[1].Shape[2] * PrototypeStride = InputHeight) and (Outputs[1].Shape[3] * PrototypeStride = InputWidth) and (InstanceMaskKind in [yimClassicV5, yimModernNative]),
      'Invalid YOLO instance prototype contract'
    );
    Require(not Detections.ApplyObjectnessToExtra or (InstanceMaskKind = yimClassicV5),
      'Objectness-scaled mask columns require the classic mask processor');
  end
  else begin
    Require(Length(Outputs) = 1, 'This YOLO task requires one prediction tensor');
    case Task of
      ytDetection:
        Require(Detections.ExtraCount = 0, 'Unexpected columns in YOLO detection output');
      ytPose:
        Require(
          (KeypointCount > 0) and (KeypointDimensions in [2, 3]) and (Int64(KeypointCount) * KeypointDimensions = Detections.ExtraCount),
          'Invalid YOLO keypoint shape'
        );
      ytOriented:
        Require(
          (Detections.ExtraCount = 1) and (Detections.Layout in [yblChannelsFirst, yblEndToEndRows]) and not IsNan(ClassOffset) and not IsInfinite(ClassOffset) and (ClassOffset > 0),
          'Invalid YOLO oriented box contract'
        );
    end;
  end;
end;

function ReadYoloProfile(const Value: TJSONValue): TYoloProfile;
var
  Obj, Item, Recipe: TJSONObject;
  Rows: TJSONArray;
  InputShape: TArray<Int64>;
  I, PaddingValue: Integer;
  Kind: string;
begin
  Result := Default(TYoloProfile);
  Obj := AsObject(Value); UniqueKeys(Obj);
  Result.ID := Text(Obj, 'id');
  Result.Family := Text(Obj, 'family');
  Result.InputName := Text(Obj, 'input_name');
  Result.SourceSHA256 := Text(Obj, 'source_sha256');
  Recipe := AsObject(Field(Obj, 'graph_recipe'));
  Result.RecipeSHA256 := Text(Recipe, 'sha256');
  Result.RecipeFile := Text(Recipe, 'file');
  Kind := Text(Obj, 'task');
  if Kind = 'classification' then
    Result.Task := ytClassification
  else if Kind = 'object_detection' then
    Result.Task := ytDetection
  else if Kind = 'instance_segmentation' then
    Result.Task := ytInstances
  else if Kind = 'pose' then
    Result.Task := ytPose
  else if Kind = 'oriented_detection' then
    Result.Task := ytOriented
  else if Kind = 'semantic_segmentation' then
    Result.Task := ytSemantic
  else if Kind = 'depth_estimation' then
    Result.Task := ytDepth
  else
    raise EYoloProfile.Create('Unsupported YOLO task: ' + Kind);
  Kind := Text(Obj, 'preprocessing');
  if Kind = 'pillow_shortest_resize_center_crop' then
    Result.Preprocessing := ypPillowClassification
  else if Kind = 'opencv_center_crop_then_resize' then
    Result.Preprocessing := ypClassicClassification
  else if Kind = 'opencv_letterbox' then
    Result.Preprocessing := ypLetterbox
  else if Kind = 'opencv_scale_fill' then
    Result.Preprocessing := ypScaleFill
  else
    raise EYoloProfile.Create('Unsupported YOLO preprocessing: ' + Kind);
  InputShape := Shape(Field(Obj, 'input_shape'));
  Require((Length(InputShape) = 4) and (InputShape[0] = 1) and (InputShape[1] = 3),
    'Expected one RGB NCHW input');
  Result.InputHeight := InputShape[2];
  Result.InputWidth := InputShape[3];
  Result.Mean := Singles(Field(Obj, 'mean'));
  Result.Std := Singles(Field(Obj, 'std'));
  Result.ScaleUp := BooleanValue(Obj, 'scale_up');
  PaddingValue := IntegerValue(Field(Obj, 'padding')); Require(PaddingValue in [0..255], 'Invalid YOLO border value');
  Result.Padding := PaddingValue;
  Rows := AsArray(Field(Obj, 'class_names')); SetLength(Result.ClassNames, Rows.Count);
  for I := 0 to Rows.Count - 1 do
  begin
    Require(Rows.Items[I] is TJSONString, 'Expected a YOLO class name');
    Result.ClassNames[I] := JSONValueText(Rows.Items[I]);
  end;
  Require(IntegerValue(Field(Obj, 'class_count')) = Result.ClassCount, 'YOLO class labels and count differ');
  Rows := AsArray(Field(Obj, 'outputs')); SetLength(Result.Outputs, Rows.Count);
  for I := 0 to Rows.Count - 1 do
  begin
    Item := AsObject(Rows.Items[I]);
    Result.Outputs[I].Name := Text(Item, 'name');
    Result.Outputs[I].Shape := Shape(Field(Item, 'shape'));
    Kind := Text(Item, 'dtype');
    if Kind = 'float32' then
      Result.Outputs[I].DataType := yttFloat32
    else if Kind = 'uint8' then
      Result.Outputs[I].DataType := yttUInt8
    else
      raise EYoloProfile.Create('Unsupported YOLO output data type: ' + Kind);
  end;
  if Result.Task = ytClassification then
  begin
    Kind := Text(Obj, 'score_kind');
    if Kind = 'probabilities' then
      Result.ClassScoreKind := ycsProbabilities
    else if Kind = 'logits' then
      Result.ClassScoreKind := ycsLogits
    else
      raise EYoloProfile.Create('Unsupported YOLO classification scores');
    Result.TopK := IntegerValue(Field(Obj, 'top_k'));
  end
  else if Result.Task in [ytSemantic, ytDepth] then
  begin
    if Result.Task = ytSemantic then
    begin
      Kind := Text(Obj, 'layout');
      Require(((Kind = 'class_map') and (Result.Outputs[0].DataType = yttUInt8)) or
        ((Kind = 'semantic_logits') and (Result.Outputs[0].DataType = yttFloat32)),
        'Unsupported YOLO semantic output representation');
    end;
    if Result.Task = ytDepth then
    begin
      Result.DepthUnits := Text(Obj, 'depth_units');
      Result.DepthAlignCorners := BooleanValue(Obj, 'resize_align_corners');
    end;
  end
  else begin
    Kind := Text(Obj, 'layout');
    if Kind = 'channels_first' then
      Result.Detections.Layout := yblChannelsFirst
    else if Kind = 'objectness_rows' then
      Result.Detections.Layout := yblObjectnessRows
    else if Kind = 'end_to_end_rows' then
      Result.Detections.Layout := yblEndToEndRows
    else
      raise EYoloProfile.Create('Unsupported YOLO box layout');
    Kind := Text(Obj, 'coordinate_mode');
    if Kind = 'rounded_resize_axes' then
      Result.CoordinateMode := ycmRoundedResizeAxes
    else if Kind = 'legacy_symmetric_gain' then
      Result.CoordinateMode := ycmLegacySymmetricGain
    else if Kind = 'legacy_rounded_gain' then
      Result.CoordinateMode := ycmLegacyRoundedGain
    else
      raise EYoloProfile.Create('Unsupported YOLO coordinate policy');
    Result.Detections.ClassCount := Result.ClassCount;
    Result.Detections.ExtraCount := IntegerValue(Field(Obj, 'extra_count'));
    Result.Detections.Confidence := Number(Field(Obj, 'confidence'));
    Result.Detections.NmsIoU := Number(Field(Obj, 'nms_iou'));
    Result.Detections.MaxDetections := IntegerValue(Field(Obj, 'max_detections'));
    Result.Detections.MaxNmsCandidates := IntegerValue(Field(Obj, 'max_nms_candidates'));
    Result.Detections.ApplyObjectnessToExtra := BooleanValue(Obj, 'apply_objectness_to_extra');
    Kind := Text(Obj, 'nms_kind');
    if Kind = 'greedy_iou' then
      Result.NmsKind := ynGreedyIoU
    else if Kind = 'prob_iou_fast' then
      Result.NmsKind := ynProbIoUFast
    else if Kind = 'none_selected_in_graph' then
      Result.NmsKind := ynSelectedInGraph
    else
      raise EYoloProfile.Create('Unsupported YOLO suppression policy');
    Result.BoxEncoding := Text(Obj, 'box_encoding');
    if Result.Task = ytInstances then
    begin
      Kind := Text(Obj, 'mask_kind');
      if Kind = 'classic_v5' then
        Result.InstanceMaskKind := yimClassicV5
      else if Kind = 'modern_native' then
        Result.InstanceMaskKind := yimModernNative
      else
        raise EYoloProfile.Create('Unsupported YOLO instance mask processor');
      Result.PrototypeStride := IntegerValue(Field(Obj, 'prototype_stride'));
    end;
    if Result.Task = ytPose then
    begin
      Rows := AsArray(Field(Obj, 'keypoint_shape')); Require(Rows.Count = 2, 'Invalid YOLO keypoint rank');
      Result.KeypointCount := IntegerValue(Rows.Items[0]);
      Result.KeypointDimensions := IntegerValue(Rows.Items[1]);
    end;
    if Result.Task = ytOriented then Result.ClassOffset := Number(Field(Obj, 'class_offset'));
  end;
  Result.Validate;
end;

function ParseYoloProfiles(const JSONText: string): TYoloProfiles;
var
  Value: TJSONValue;
  Root: TJSONObject;
  Rows: TJSONArray;
  I, J: Integer;
begin
  Value := ParseCrossJSON(JSONText);
  try
    Root := AsObject(Value); UniqueKeys(Root);
    Require(Text(Root, 'schema') = 'crosspascal.yolo-profiles.v1', 'Unsupported YOLO profile catalog schema');
    Rows := AsArray(Field(Root, 'profiles')); Require(Rows.Count > 0, 'Empty YOLO profile catalog');
    SetLength(Result, Rows.Count);
    for I := 0 to Rows.Count - 1 do
    begin
      Result[I] := ReadYoloProfile(Rows.Items[I]);
      for J := 0 to I - 1 do
        Require(Result[I].ID <> Result[J].ID, 'Duplicate YOLO profile identifier');
    end;
  finally
    Value.Free;
  end;
end;

end.
