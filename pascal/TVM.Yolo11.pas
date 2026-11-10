{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo11                                                                 }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo11;

{$IFDEF FPC}
  {$MODE DELPHI}
{$ENDIF}
{$POINTERMATH ON}
{$B-}

interface

uses
  {$IFDEF FPC}
  {$IFDEF MSWINDOWS}Windows,{$ENDIF}Classes, Math, SysUtils, Types,
  Generics.Collections, Generics.Defaults, DLPack, OpenCV, OpenCV.Core, OpenCV.ImgProc,
  OpenCV.ImgCodecs, OpenCV.ObjDetect, OpenCV.Custom, OpenCVTypes, OpenCVConsts,
  CrossVision.Float16, CrossVision.Mat, TVM;
  {$ELSE}
  WinApi.Windows, System.SysUtils, System.Classes, System.Math, System.Types, System.Generics.Collections,
  System.Generics.Defaults, DLPack, OpenCV, OpenCV.Core, OpenCV.ImgProc, OpenCV.ImgCodecs,
  OpenCV.ObjDetect, OpenCV.Custom, OpenCVTypes, OpenCVConsts, CrossVision.Float16,
  CrossVision.Mat, TVM;
  {$ENDIF}

type
  PCandidateDetection = ^TCandidateDetection;
  TCandidateDetection = record
  public
    Place: TRect;
    PlaceExact: TRectF;
    &Class: Integer;
    Score: Single;
    OriginalIndex: Int64;
    class function Make(const Place: TRect; const PlaceExact: TRectF; const &Class: Integer;
      const Score: Single; const OriginalIndex: Int64): TCandidateDetection; static;
    procedure Reset;
  end;

  PYolo11IntermediateDetection = ^TYolo11IntermediateDetection;
  TYolo11IntermediateDetection = record
  public
    Place: TRect;
    PlaceExact: TRectF;
    &Class: Integer;
    Score: Single;
    TrailingColumns: TArray<Single>;
    OriginalIndex: Int64;
    class function Make(const Place: TRect; const &Class: Integer; const Score: Single;
      const TrailingColumns: TArray<Single>;
      const OriginalIndex: Int64): TYolo11IntermediateDetection; static;
    procedure Reset;
  end;
  TYolo11IntermediateDetections = TArray<TYolo11IntermediateDetection>;
  PYolo11IntermediateDetections = ^TYolo11IntermediateDetections;

  TYoloLayout = (ylAuto, ylDense, ylDenseObjectness, ylNmsFree);

  PYolo11Detection = ^TYolo11Detection;
  TYolo11Detection = record
    Place: TRect;
    &Class: Integer;
    Score: Single;
    Mask: IMat;
    class function Make(const Place: TRect; const &Class: Integer; const Score: Single;
      const Mask: IMat): TYolo11Detection; static;
    procedure Reset;
  end;
  TYolo11Detections = TArray<TYolo11Detection>;

  TYolo11DetectionsHelper = record helper for TYolo11Detections
  public
    function Empty: Boolean;
    function Lo: Integer;
    function Hi: Integer;
    function Size: Integer;
    function At(const Index: Integer): PYolo11Detection;
  end;

  TYolo11 = record
  public
    class function Detect(var Descriptor: TModelDescriptor; const Image: IMat;
      var OutputTensors: TArray<PDLTensor>; const Threshold, NMSThreshold: Single; out Scores: TArray<Single>;
      out Labels: TArray<Integer>; out Places: TArray<TRect>; const InputTensorName: string = 'images';
      const Error: PString = nil; const Layout: TYoloLayout = ylAuto; const NumTrailing: Integer = 0;
      const Found: PYolo11IntermediateDetections = nil): Boolean; overload; static;
    class function Detect(var Descriptor: TModelDescriptor; const Image: IMat;
      const OutputDeviceType, OutputDeviceID: Integer; const Threshold, NMSThreshold: Single;
      out Scores: TArray<Single>; out Labels: TArray<Integer>; out Places: TArray<TRect>;
      const InputTensorName: string = 'images'; const Error: PString = nil;
      const Layout: TYoloLayout = ylAuto): Boolean; overload; static;
    class function RunAndDecode(var Descriptor: TModelDescriptor; const Image: IMat;
      const OutputDeviceType, OutputDeviceID: Integer; const Threshold, NMSThreshold: Single;
      const NumTrailing: Integer; out Found: TYolo11IntermediateDetections;
      const InputTensorName: string = 'images'; const Error: PString = nil;
      const Layout: TYoloLayout = ylAuto): Boolean; static;
    class function Categorize(var Descriptor: TModelDescriptor; const Image: IMat;
      var OutputTensors: TArray<PDLTensor>; out A: TArray<Single>; const InputTensorName: string = 'images';
      const Error: PString = nil; const Mean: TArray<Single> = nil;
      const Std: TArray<Single> = nil): Boolean; overload; static;
    class function Categorize(var Descriptor: TModelDescriptor; const Image: IMat;
      const OutputDeviceType, OutputDeviceID: Integer; out A: TArray<Single>;
      const InputTensorName: string = 'images'; const Error: PString = nil; const Mean: TArray<Single> = nil;
      const Std: TArray<Single> = nil): Boolean; overload; static;
    class function DecodeAndFilterDetections(const DetectTensor: PDLTensor;
      const ConfidenceThreshold, NMSThreshold: Single; const ModelInputSize, OriginalImageSize: TSize;
      const NumMaskCoeffs: Integer; out FilteredDetections: TYolo11IntermediateDetections;
      const Error: PString; const Layout: TYoloLayout = ylAuto): Boolean; static;
    class function IsNmsFreeLayout(const DetectTensor: PDLTensor;
      const NumMaskCoeffs: Integer): Boolean; static;
    class function DecodeNmsFreeDetections(const DetectTensor: PDLTensor; const ConfidenceThreshold: Single;
      const ModelInputSize, OriginalImageSize: TSize; const NumMaskCoeffs: Integer;
      out FilteredDetections: TYolo11IntermediateDetections; const Error: PString): Boolean; static;
    class function PrepareProtoData(const PrototypeTensor: PDLTensor; out PrototypeDataPtr: Pointer;
      out NumPrototypes: Integer; out MaskPrototypeSize: TSize; const Error: PString = nil): Boolean; static;
    class function GenerateMasks(const FilteredDetections: TYolo11IntermediateDetections;
      const PrototypeDataPtr: Pointer; const NumPrototypes: Integer;
      const BoxSpaceSize, MaskPrototypeSize: TSize; out Detections: TYolo11Detections;
      const Error: PString = nil): Boolean; static;
    class function Segment(var Descriptor: TModelDescriptor; const Image: IMat;
      var OutputTensors: TArray<PDLTensor>; const Threshold, NMSThreshold: Single;
      out Detections: TYolo11Detections; const InputTensorName: string = 'images'; const Error: PString = nil;
      const Layout: TYoloLayout = ylAuto): Boolean; overload; static;
    class function Segment(var Descriptor: TModelDescriptor; const Image: IMat;
      const OutputDeviceType: Integer; const OutputDeviceID: Integer; const Threshold, NMSThreshold: Single;
      out Detections: TYolo11Detections; const InputTensorName: string = 'images'; const Error: PString = nil;
      const Layout: TYoloLayout = ylAuto): Boolean; overload; static;
    class function BuildClassInstanceMaps(const Detections: TYolo11Detections;
      const Size: TSize): TArray<IMat>; static;
    class function ComposeMasksFromProto(const Prototype, Detection: PDLTensor;
      const Indices: TArray<Integer>; const NumFeatures: Int64; const CoefStartIndex: Integer;
      out Masks: TArray<IMat>; const Threshold: Single; const Error: PString = nil): Boolean; static;
  end;

function CalcIoU(const A, B: TRect): Single; inline;
function GetDataElementPtr(const BasePtr: Pointer; const BoxIndex, FeatureIndex, NumBoxes, NumFeatures: Int64;
  const NCHWLayout: Boolean; const ElementSize: UInt64): Pointer; inline;
function MultiplyMaskCoeffsProtos(const Coeffs, Prototypes, Output: PSingle;
  const NumPrototypes, PrototypeElementCount: Integer): Boolean; inline;

implementation

type
  TYolo11Tensors = record
  public
    Input: PDLTensor;
    Image: PDLTensor;
    Detection: PDLTensor;
    Prototype: PDLTensor;
    CpuDetection: PDLTensor;
    CpuPrototype: PDLTensor;
  end;

procedure ApplyChannelNormalization(const Image: IMat; const Mean, Std: TArray<Single>);
var
  X, Y, C, Channels: Integer;
  Row: PSingle;
  Value: Double;
begin
  if not Assigned(Image) or (Length(Mean) = 0) or (Length(Std) = 0) then
    Exit;
  Channels := Image.Channels;
  if (Length(Mean) < Channels) or (Length(Std) < Channels) then
    Exit;
  for Y := 0 to Image.Height - 1 do
  begin
    Row := PSingle(PByte(Image.DataPtr) + Y * Image.Step);
    for X := 0 to Image.Width - 1 do
      for C := 0 to Channels - 1 do
      begin
        if Std[C] <> 0 then
        begin
          Value := Row^;
          Value := (Value - Mean[C]) / Std[C];
          Row^ := Value;
        end;
        Inc(Row);
      end;
  end;
end;

class function TCandidateDetection.Make(const Place: TRect; const PlaceExact: TRectF;
  const &Class: Integer; const Score: Single; const OriginalIndex: Int64): TCandidateDetection;
begin
  Result.Reset;
  Result.Place := Place;
  Result.PlaceExact := PlaceExact;
  Result.&Class := &Class;
  Result.Score := Score;
  Result.OriginalIndex := OriginalIndex;
end;

procedure TCandidateDetection.Reset;
begin
  FillChar(Self, SizeOf(TCandidateDetection), 0);
end;

class function TYolo11IntermediateDetection.Make(const Place: TRect; const &Class: Integer;
  const Score: Single; const TrailingColumns: TArray<Single>;
  const OriginalIndex: Int64): TYolo11IntermediateDetection;
begin
  Result.Place := Place;
  FillChar(Result.PlaceExact, SizeOf(Result.PlaceExact), 0);
  Result.&Class := &Class;
  Result.Score := Score;
  Result.TrailingColumns := TrailingColumns;
  Result.OriginalIndex := OriginalIndex;
end;

procedure TYolo11IntermediateDetection.Reset;
begin
  TrailingColumns := nil;
  FillChar(Self, SizeOf(TYolo11IntermediateDetection), 0);
end;

class function TYolo11Detection.Make(const Place: TRect; const &Class: Integer; const Score: Single;
  const Mask: IMat): TYolo11Detection;
begin
  Result.Reset;
  Result.Place := Place;
  Result.&Class := &Class;
  Result.Score := Score;
  Result.Mask := Mask;
end;

procedure TYolo11Detection.Reset;
begin
  Mask := nil;
  Place := TRect.Create(0, 0, 0, 0);
  &Class := 0;
  Score := 0;
end;

function TYolo11DetectionsHelper.At(const Index: Integer): PYolo11Detection;
begin
  Result := @Self[Index];
end;

function TYolo11DetectionsHelper.Empty: Boolean;
begin
  Result := not Assigned(Self);
end;

function TYolo11DetectionsHelper.Hi: Integer;
begin
  Result := High(Self);
end;

function TYolo11DetectionsHelper.Lo: Integer;
begin
  Result := Low(Self);
end;

function TYolo11DetectionsHelper.Size: Integer;
begin
  Result := Length(Self);
end;

function CalcIoU(const A, B: TRect): Single;
var
  IntersectionRect: TRect;
  IntersectionSize: TSize;
  IntersectionArea, UnionArea: Single;
begin
  IntersectionRect := TRect.Create(Max(A.Left, B.Left), Max(A.Top, B.Top), Min(A.Right, B.Right),
    Min(A.Bottom, B.Bottom));
  IntersectionSize := TSize.Create(
    IntersectionRect.Right - IntersectionRect.Left,
    IntersectionRect.Bottom - IntersectionRect.Top
  );
  if (IntersectionSize.cx < 1) or (IntersectionSize.cy < 1) then Exit(0);
  IntersectionArea := IntersectionSize.cx * IntersectionSize.cy;
  UnionArea := (A.Right - A.Left) * (A.Bottom - A.Top) + (B.Right - B.Left) * (B.Bottom - B.Top) -
    IntersectionArea;
  if UnionArea > 0 then
    Result := IntersectionArea / UnionArea
  else
    Result := 0;
end;

function GetDataElementPtr(const BasePtr: Pointer; const BoxIndex, FeatureIndex, NumBoxes, NumFeatures: Int64;
  const NCHWLayout: Boolean; const ElementSize: UInt64): Pointer;
var
  Offset: UInt64;
begin
  if NCHWLayout then
    Offset := UInt64(FeatureIndex * NumBoxes + BoxIndex) * ElementSize
  else
    Offset := UInt64(BoxIndex * NumFeatures + FeatureIndex) * ElementSize;
  Result := Pointer(UInt64(BasePtr) + Offset);
end;

function MultiplyMaskCoeffsProtos(const Coeffs, Prototypes, Output: PSingle;
  const NumPrototypes, PrototypeElementCount: Integer): Boolean;
var
  I, J: Int64;
  Sum: Extended;
begin
  if not Assigned(Coeffs) or not Assigned(Prototypes) or not Assigned(Output) or
    (NumPrototypes <= 0) or (PrototypeElementCount <= 0) then
      Exit(False);
  for I := 0 to PrototypeElementCount - 1 do
  begin
    Sum := 0;
    for J := 0 to NumPrototypes - 1 do
      Sum := Sum + Coeffs[J] * Prototypes[J * PrototypeElementCount + I];
    Output[I] := Sum;
  end;
  Result := True;
end;

class function TYolo11.ComposeMasksFromProto(const Prototype, Detection: PDLTensor;
  const Indices: TArray<Integer>; const NumFeatures: Int64; const CoefStartIndex: Integer;
  out Masks: TArray<IMat>; const Threshold: Single; const Error: PString): Boolean;
type
  TPtr = record
    Base: PSingle;
    Prototype: PSingle;
    Mask: PSingle;
    Coef: PSingle;
    &End: PSingle;
  end;
var
  I, J, W, H, N, Index, Count, NumPrototypes, NumMaskCoeffs: Integer;
  Ptr: TPtr;
  CoefBase: PSingle;
  CoefStep: NativeInt;
  Sigm: Single;
begin
  Masks := nil;
  if not Assigned(Prototype) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.ComposeMasksFromProto: Prototype tensor is nil';
    Exit(False);
  end;
  if not Assigned(Detection) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.ComposeMasksFromProto: Detection tensor is nil';
    Exit(False);
  end;
  if not Assigned(Prototype.data) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.ComposeMasksFromProto: Prototype.data is nil';
    Exit(False);
  end;
  if not Assigned(Detection.data) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.ComposeMasksFromProto: Detection.data is nil';
    Exit(False);
  end;
  if (TDLDataTypeCode(Prototype.dtype.code) <> kDLFloat) or (Prototype.dtype.bits <> 32) then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.ComposeMasksFromProto: Prototype tensor must be Float32, got code %d, bits %d',
        [Prototype.dtype.code, Prototype.dtype.bits]);
    Exit(False);
  end;
  if (Detection.ndim < 2) or (TDLDataTypeCode(Detection.dtype.code) <> kDLFloat) or
    (Detection.dtype.bits <> 32) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.ComposeMasksFromProto: Detection tensor must be at least 2D and Float32, got ndim %d, code %d, bits %d',
          [Detection.ndim, Detection.dtype.code, Detection.dtype.bits]);
      Exit(False);
    end;
  case Prototype.ndim of
    4:
      begin
        W := Prototype.shape[3];
        H := Prototype.shape[2];
      end
  else
    W := Prototype.shape[2];
    H := Prototype.shape[1];
  end;
  if Prototype.ndim = 4 then
    NumPrototypes := Prototype.shape[1]
  else
    NumPrototypes := Prototype.shape[0];
  if NumPrototypes < 1 then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.ComposeMasksFromProto: Number of prototypes (%d) derived from tensor shape is less than 1',
        [NumPrototypes]);
    Exit(False);
  end;
  NumMaskCoeffs := NumFeatures - CoefStartIndex;
  if NumMaskCoeffs < 1 then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.ComposeMasksFromProto: CoefStartIndex=%d is out of bounds for NumFeatures=%d',
        [CoefStartIndex, NumFeatures]);
    Exit(False);
  end;
  if NumMaskCoeffs > NumPrototypes then
    NumMaskCoeffs := NumPrototypes;
  case Detection.ndim of
    3: N := Detection.shape[2];
    2: N := Detection.shape[0];
  else
    if Assigned(Error) then
      Error^ := Format('TYolo11.ComposeMasksFromProto: Detection tensor has unsupported ndim: %d (expected 2 or 3)',
        [Detection.ndim]);
    Exit(False);
  end;
  FillChar(Ptr, SizeOf(TPtr), 0);
  if Length(Indices) > 0 then
    Count := Length(Indices)
  else
    Count := N;
  SetLength(Masks, Count);
  for I := 0 to Count - 1 do
  begin
    if Length(Indices) > 0 then
      Index := Indices[I]
    else
      Index := I;
    Masks[I] := NewMat(W, H, CV_MAKETYPE(CV_32F, 1));
    Ptr.Mask := PSingle(Masks[I].DataPtr);
    Ptr.Base := Ptr.Mask;
    FillChar(Ptr.Mask^, W * H * SizeOf(Single), 0);
    if (Detection.ndim = 3) and (Detection.shape[1] = NumFeatures) then
    begin
      CoefStep := N;
      CoefBase := PSingle(Detection.data) + Index + CoefStartIndex * N;
    end
    else if Detection.shape[1] = NumFeatures then
      begin
        CoefStep := 1;
        CoefBase := PSingle(Detection.data) + Index * NumFeatures + CoefStartIndex;
      end
      else if Detection.shape[0] = NumFeatures then
        begin
          CoefStep := N;
          CoefBase := PSingle(Detection.data) + CoefStartIndex * N + Index;
        end
        else begin
          if Assigned(Error) then
          begin
            if Detection.ndim > 1 then
              J := Detection.shape[2]
            else
              J := -1;
            Error^ := Format(
              'TYolo11.ComposeMasksFromProto: Cannot determine coefficient layout from Detection tensor shape (ndim: %d, shape[0]: %d, shape[1]: %d, shape[2]: %d) and NumFeatures: %d for index %d',
              [
                Detection.ndim,
                Detection.shape[0],
                Detection.shape[1],
                J,
                NumFeatures,
                I
              ]
            );
          end;
          Exit(False);
        end;
    for J := 0 to NumMaskCoeffs - 1 do
    begin
      Ptr.Prototype := PSingle(Prototype.data) + J * W * H;
      Ptr.Mask := Ptr.Base;
      while Ptr.Mask < Ptr.Base + W * H do
      begin
        Ptr.Mask^ := Ptr.Mask^ + CoefBase[J * CoefStep] * Ptr.Prototype^;
        Inc(Ptr.Mask);
        Inc(Ptr.Prototype);
      end;
    end;
    Ptr.Mask := Ptr.Base;
    Ptr.&End := Ptr.Base + W * H;
    while Ptr.Mask < Ptr.&End do
    begin
      Sigm := 1 / (1 + Exp(-Ptr.Mask^));
      if Sigm >= Threshold then
        Ptr.Mask^ := Sigm
      else
        Ptr.Mask^ := 0;
      Inc(Ptr.Mask);
    end;
  end;
  Result := True;
end;

class function TYolo11.BuildClassInstanceMaps(const Detections: TYolo11Detections;
  const Size: TSize): TArray<IMat>;
var
  I: Integer;
  Mask: IMat;
  R: TRect;
  RoiX, RoiY, RoiW, RoiH: Integer;
  RoiView: IMat;
begin
  SetLength(Result, Length(Detections));
  for I := Low(Detections) to High(Detections) do
  begin
    Result[I] := NewMat(Size.cx, Size.cy, CV_MAKETYPE(CV_8U, 1)).Fill(0);
    Mask := Detections[I].Mask;
    if Assigned(Mask) then
    begin
      R := Detections[I].Place;
      if (R.Left < Size.Width) and (R.Top < Size.Height) and (R.Width > 0) and (R.Height > 0) then
      begin
        RoiX := R.Left;
        RoiY := R.Top;
        RoiW := R.Width;
        RoiH := R.Height;
        if RoiX + RoiW > Result[I].Width then
          RoiW := Result[I].Width - RoiX;
        if RoiY + RoiH > Result[I].Height then
          RoiH := Result[I].Height - RoiY;
        if RoiW > Mask.Width then
          RoiW := Mask.Width;
        if RoiH > Mask.Height then
          RoiH := Mask.Height;
        if (RoiW > 0) and (RoiH > 0) then
        begin
          RoiView := RegionView(Result[I], RoiX, RoiY, RoiW, RoiH);
          if Assigned(RoiView) then
            CopyRegionInto(Mask, 0, 0, RoiW, RoiH, RoiView);
        end;
      end;
    end;
  end;
end;

{$IFDEF FPC}

function RectFOf(const L, T, R, B: Single): TRectF;
begin
  Result.Left := L;
  Result.Top := T;
  Result.Right := R;
  Result.Bottom := B;
end;

type
  TIndexComparer<T> = class(TInterfacedObject, IComparer<T>)
  private
    FList: TList<TCandidateDetection>;
  public
    constructor Create(AList: TList<TCandidateDetection>);
    function Compare(constref ALeft, ARight: T): Integer;
  end;

constructor TIndexComparer<T>.Create(AList: TList<TCandidateDetection>);
begin
  inherited Create;
  FList := AList;
end;

function TIndexComparer<T>.Compare(constref ALeft, ARight: T): Integer;
begin
  if FList[Integer(ALeft)].Score > FList[Integer(ARight)].Score then
    Result := -1
  else if FList[Integer(ARight)].Score > FList[Integer(ALeft)].Score then
      Result := 1
    else
      Result := 0;
end;
{$ENDIF}

class function TYolo11.Categorize(var Descriptor: TModelDescriptor; const Image: IMat;
  const OutputDeviceType, OutputDeviceID: Integer; out A: TArray<Single>; const InputTensorName: string;
  const Error: PString; const Mean, Std: TArray<Single>): Boolean;
var
  OutputTensors: TArray<PDLTensor>;
  S: string;
begin
  OutputTensors := nil;
  try
    if not TTVMRuntime.InitializeOutputTensors(Descriptor, OutputTensors, OutputDeviceType,
      OutputDeviceID, @S) then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.Categorize: Failed to initialize output tensors. Details: %s', [S]);
        Exit(False);
      end;
    Result := Categorize(Descriptor, Image, OutputTensors, A, InputTensorName, Error, Mean, Std);
  finally
    OutputTensors := nil;
  end;
end;

class function TYolo11.Categorize(var Descriptor: TModelDescriptor; const Image: IMat;
  var OutputTensors: TArray<PDLTensor>; out A: TArray<Single>; const InputTensorName: string;
  const Error: PString; const Mean, Std: TArray<Single>): Boolean;
type
  TTensors = record
    Input: PDLTensor;
    Image: PDLTensor;
    Scores: PDLTensor;
    CpuScores: PDLTensor;
  end;
var
  Tensors: TTensors;
  S: string;
  TensorIndex: Int64;
  Scale: Single;
  Img: IMat;
  Shape: TArray<Int64>;
  NumClasses, I: Integer;
  ExpectedStride: Int64;
  ScoreBytes: UInt64;
begin
  A := nil;
  FillChar(Tensors, SizeOf(TTensors), 0);
  try
    if not TTVMRuntime.GetInputTensorIndex(Descriptor, InputTensorName, TensorIndex, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Categorize: Failed to get input tensor index for "%s". Details: %s',
          [InputTensorName, S]);
      Exit(False);
    end;
    if not TTVMRuntime.GetInputTensor(Descriptor, TensorIndex, Tensors.Input, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Categorize: Failed to get input tensor schema (index %d). Details: %s',
          [TensorIndex, S]);
      Exit(False);
    end;
    if not Assigned(Tensors.Input) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Categorize: GetInputTensor succeeded but returned a nil InputSchema';
      Exit(False);
    end;
    case TDLDataTypeCode(Tensors.Input.dtype.code) of
      kDLFloat: Scale := 1.0 / 255.0;
      kDLUInt, kDLInt: Scale := 1.0;
    else begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.Categorize: Unsupported input tensor data type code: %d. Cannot determine image scale',
            [Tensors.Input.dtype.code]);
        Exit(False);
      end;
    end;
    Img := nil;
    try
      if not TTVMRuntime.ConvertImageToTensorFormat(Image, Img, Tensors.Input, Yolo11Tensor, Scale,
        INTER_CUBIC, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Categorize: Image preprocessing (ConvertImageToTensorFormat) failed. Details: %s', [S]);
          Exit(False);
        end;
      if not Assigned(Img) then
      begin
        if Assigned(Error) then
          Error^ := 'TYolo11.Categorize: ConvertImageToTensorFormat reported success but ProcessedImg is nil';
        Exit(False);
      end;
      Img := ConvertBgrToRgb(Img);
      ApplyChannelNormalization(Img, Mean, Std);
      if not TTVMRuntime.ConvertImageToDLTensor(Img, Descriptor.Device, Yolo11Tensor, Tensors.Image,
        TDLDataTypeCode(Tensors.Input.dtype.code), Tensors.Input.dtype.bits,
        Tensors.Input.dtype.lanes, 1, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Categorize: Failed to build input tensor (ConvertImageToDLTensor). Details: %s', [S]);
          Exit(False);
        end;
      if not Assigned(Tensors.Image) then
      begin
        if Assigned(Error) then
          Error^ := 'TYolo11.Categorize: ConvertImageToDLTensor reported success but InputImage is nil';
        Exit(False);
      end;
    finally
      if Assigned(Img) then Img := nil;
    end;
    if not TTVMRuntime.RunInference(Descriptor, InputTensorName, Tensors.Image, OutputTensors, Error) then
      Exit(False);
    if Length(OutputTensors) <> 1 then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Categorize: classifier must have exactly one output tensor, got %d',
          [Length(OutputTensors)]);
      Exit(False);
    end;
    if not Assigned(OutputTensors[0]) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Categorize: OutputTensors[0] is nil after RunInference';
      Exit(False);
    end;
    Tensors.Scores := TTVMRuntime.GetOrCreateCpuTensor(OutputTensors[0], Tensors.CpuScores,
      'Scores', 'TYolo11.Categorize', Error);
    if not Assigned(Tensors.Scores) then Exit(False);
    if not TTVMRuntime.CheckTensor(Tensors.Scores, [kDLFloat], 32) then
    begin
      if Assigned(Error) then
        Error^ := Format(
          'TYolo11.Categorize: Scores tensor must be Float32. Got code %d, bits %d. Shape: %s',
          [
            Tensors.Scores.dtype.code,
            Tensors.Scores.dtype.bits,
            ShapeToString(TTVMRuntime.GetTensorShape(Tensors.Scores))
          ]
        );
      Exit(False);
    end;
    if not Assigned(Tensors.Scores.data) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Categorize: Scores tensor data is nil after ensuring CPU';
      Exit(False);
    end;
    if not Assigned(Tensors.Scores.shape) or (Tensors.Scores.ndim < 1) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Categorize: Scores tensor must have a nonempty shape after ensuring CPU';
      Exit(False);
    end;
    Shape := TTVMRuntime.GetTensorShape(Tensors.Scores);
    if (Length(Shape) = 0) or ((Length(Shape) > 1) and (Shape[0] <> 1)) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Categorize: expected a class vector with batch size 1. Shape: %s',
          [ShapeToString(Shape)]);
      Exit(False);
    end;
    NumClasses := 1;
    for I := 0 to High(Shape) do
    begin
      if (Shape[I] < 1) or (Shape[I] > High(Integer) div SizeOf(Single)) then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.Categorize: invalid class-vector dimension. Shape: %s',
            [ShapeToString(Shape)]);
        Exit(False);
      end;
      if Shape[I] > 1 then
      begin
        if NumClasses <> 1 then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Categorize: only one class dimension may be non-singleton. Shape: %s',
              [ShapeToString(Shape)]);
          Exit(False);
        end;
        NumClasses := Integer(Shape[I]);
      end;
    end;
    if NumClasses < 1 then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Categorize: Invalid number of classes derived from scores tensor shape (%d). Shape: %s',
          [NumClasses, ShapeToString(Shape)]);
      Exit(False);
    end;
    if Assigned(Tensors.Scores.strides) then
    begin
      ExpectedStride := 1;
      for I := High(Shape) downto 0 do
      begin
        if Tensors.Scores.strides[I] <> ExpectedStride then
        begin
          if Assigned(Error) then
            Error^ := 'TYolo11.Categorize: Scores tensor must have contiguous row-major strides';
          Exit(False);
        end;
        ExpectedStride := ExpectedStride * Shape[I];
      end;
    end;
    if Tensors.Scores.byte_offset <> 0 then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Categorize: nonzero tensor byte offsets are not supported';
      Exit(False);
    end;
    ScoreBytes := UInt64(NumClasses) * SizeOf(Single);
    if ScoreBytes - UInt64(1) > UInt64(High(NativeUInt) - NativeUInt(Tensors.Scores.data)) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Categorize: Scores tensor byte span overflows address space';
      Exit(False);
    end;
    SetLength(A, NumClasses);
    System.Move(Tensors.Scores.data^, A[0], NumClasses * SizeOf(Single));
  finally
    if Assigned(Tensors.Image) then
      FreeMem(Tensors.Image);
    if Assigned(Tensors.CpuScores) then
      FreeMem(Tensors.CpuScores);
  end;
  Result := True;
end;

function ReadRowMajorSingle(const BasePtr: Pointer; const Row, Col, NumRows, NumCols: Int64;
  const Float16Flag: Boolean; const ElementSize: UInt64): Single;
var
  P: Pointer;
  Half: TFloat16;
begin
  P := GetDataElementPtr(BasePtr, Row, Col, NumRows, NumCols, False, ElementSize);
  if Float16Flag then
  begin
    Half.Value := PWord(P)^;
    Result := Half.ToSingle;
  end
  else
    Result := PSingle(P)^;
end;

class function TYolo11.IsNmsFreeLayout(const DetectTensor: PDLTensor; const NumMaskCoeffs: Integer): Boolean;
const
  MaxSaneClass = 10000;
  ScoreSlack = 1e-4;
var
  Shape: TArray<Int64>;
  NumRows, I: Int64;
  ElementSize: UInt64;
  Float16Flag: Boolean;
  Score, PrevScore, ClassValue, Left, Top, Right, Bottom: Single;
  RowWidth: Int64;
begin
  Result := False;
  if not Assigned(DetectTensor) or not Assigned(DetectTensor.data) then
    Exit;
  if DetectTensor.device.device_type <> kDLCPU then
    Exit;
  if DetectTensor.dtype.code <> UInt8(kDLFloat) then
    Exit;
  case DetectTensor.dtype.bits of
    16:
      begin
        Float16Flag := True;
        ElementSize := SizeOf(TFloat16);
      end;
    32:
      begin
        Float16Flag := False;
        ElementSize := SizeOf(Single);
      end;
  else
    Exit;
  end;
  Shape := TTVMRuntime.GetTensorShape(DetectTensor);
  if (Length(Shape) <> 3) or (Shape[0] > 1) or (Shape[2] <> 6 + NumMaskCoeffs) then
    Exit;
  NumRows := Shape[1];
  if NumRows < 1 then
    Exit;
  RowWidth := 6 + NumMaskCoeffs;
  PrevScore := 1 + ScoreSlack;
  for I := 0 to NumRows - 1 do
  begin
    Score := ReadRowMajorSingle(DetectTensor.data, I, 4, NumRows, RowWidth, Float16Flag, ElementSize);
    if (Score < -ScoreSlack) or (Score > 1 + ScoreSlack) then
      Exit;
    if Score > PrevScore + ScoreSlack then
      Exit;
    PrevScore := Score;
    ClassValue := ReadRowMajorSingle(DetectTensor.data, I, 5, NumRows, RowWidth, Float16Flag, ElementSize);
    if (ClassValue < 0) or (ClassValue >= MaxSaneClass) then
      Exit;
    if Abs(ClassValue - Round(ClassValue)) > 1e-3 then
      Exit;
    Left := ReadRowMajorSingle(DetectTensor.data, I, 0, NumRows, RowWidth, Float16Flag, ElementSize);
    Top := ReadRowMajorSingle(DetectTensor.data, I, 1, NumRows, RowWidth, Float16Flag, ElementSize);
    Right := ReadRowMajorSingle(DetectTensor.data, I, 2, NumRows, RowWidth, Float16Flag, ElementSize);
    Bottom := ReadRowMajorSingle(DetectTensor.data, I, 3, NumRows, RowWidth, Float16Flag, ElementSize);
    if (Right < Left) or (Bottom < Top) then
      Exit;
  end;
  Result := True;
end;

class function TYolo11.DecodeNmsFreeDetections(const DetectTensor: PDLTensor;
  const ConfidenceThreshold: Single; const ModelInputSize, OriginalImageSize: TSize;
  const NumMaskCoeffs: Integer; out FilteredDetections: TYolo11IntermediateDetections;
  const Error: PString): Boolean;
var
  Shape: TArray<Int64>;
  NumRows, I: Int64;
  ElementSize: UInt64;
  Float16Flag: Boolean;
  XFactor, YFactor: Extended;
  Score, ClassValue: Single;
  OriginRect: TRectF;
  DetectionRect: TRect;
  Found: TList<TCandidateDetection>;
  K, J: Integer;
  RowWidth: Int64;
begin
  FilteredDetections := nil;
  Shape := TTVMRuntime.GetTensorShape(DetectTensor);
  NumRows := Shape[1];
  RowWidth := 6 + NumMaskCoeffs;
  Float16Flag := DetectTensor.dtype.bits = 16;
  if Float16Flag then
    ElementSize := SizeOf(TFloat16)
  else
    ElementSize := SizeOf(Single);
  if (ModelInputSize.cx <= 0) or (ModelInputSize.cy <= 0) then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.DecodeNmsFreeDetections: Model input size must be positive, got %dx%d',
        [ModelInputSize.cx, ModelInputSize.cy]);
    Exit(False);
  end;
  XFactor := OriginalImageSize.cx / ModelInputSize.cx;
  YFactor := OriginalImageSize.cy / ModelInputSize.cy;
  Found := TList<TCandidateDetection>.Create;
  try
    for I := 0 to NumRows - 1 do
    begin
      Score := ReadRowMajorSingle(DetectTensor.data, I, 4, NumRows, RowWidth, Float16Flag, ElementSize);
      if Score < ConfidenceThreshold then
        Continue;
      ClassValue := ReadRowMajorSingle(DetectTensor.data, I, 5, NumRows, RowWidth, Float16Flag, ElementSize);
      OriginRect := {$IFDEF FPC}RectFOf{$ELSE}TRectF.Create{$ENDIF}(
        ReadRowMajorSingle(DetectTensor.data, I, 0, NumRows, RowWidth, Float16Flag, ElementSize) * XFactor,
        ReadRowMajorSingle(DetectTensor.data, I, 1, NumRows, RowWidth, Float16Flag, ElementSize) * YFactor,
        ReadRowMajorSingle(DetectTensor.data, I, 2, NumRows, RowWidth, Float16Flag, ElementSize) * XFactor,
        ReadRowMajorSingle(DetectTensor.data, I, 3, NumRows, RowWidth, Float16Flag, ElementSize) * YFactor);
      DetectionRect := TRect.Create(
        Max(
          0,
          Min(
            OriginalImageSize.cx - 1,
            Round(OriginRect.TopLeft.X)
          )
        ),
        Max(
          0,
          Min(
            OriginalImageSize.cy - 1,
            Round(OriginRect.TopLeft.Y)
          )
        ),
        Max(
          0,
          Min(
            OriginalImageSize.cx - 1,
            Round(OriginRect.BottomRight.X)
          )
        ),
        Max(
          0,
          Min(
            OriginalImageSize.cy - 1,
            Round(OriginRect.BottomRight.Y)
          )
        )
      );
      if (DetectionRect.Right > DetectionRect.Left) and
        (DetectionRect.Bottom > DetectionRect.Top) then
          Found.Add(TCandidateDetection.Make(DetectionRect, OriginRect, Round(ClassValue), Score, I));
    end;
    SetLength(FilteredDetections, Found.Count);
    for K := 0 to Found.Count - 1 do
    begin
      FilteredDetections[K] := TYolo11IntermediateDetection.Make(
        Found[K].Place,
        Found[K].&Class,
        Found[K].Score,
        nil,
        Found[K].OriginalIndex
      );
      FilteredDetections[K].PlaceExact := Found[K].PlaceExact;
      SetLength(FilteredDetections[K].TrailingColumns, NumMaskCoeffs);
      for J := 0 to NumMaskCoeffs - 1 do
        FilteredDetections[K].TrailingColumns[J] := ReadRowMajorSingle(
          DetectTensor.data,
          FilteredDetections[K].OriginalIndex,
          6 + J,
          NumRows,
          RowWidth,
          Float16Flag,
          ElementSize
        );
    end;
  finally
    Found.Free;
  end;
  Result := True;
end;

class function TYolo11.DecodeAndFilterDetections(const DetectTensor: PDLTensor;
  const ConfidenceThreshold, NMSThreshold: Single; const ModelInputSize, OriginalImageSize: TSize;
  const NumMaskCoeffs: Integer; out FilteredDetections: TYolo11IntermediateDetections; const Error: PString;
  const Layout: TYoloLayout): Boolean;
type
  TShapeInterpretation = record
    FeaturesDim1: Int64;
    BoxesDim1: Int64;
    NumClasses1: Int64;
    FeaturesDim2: Int64;
    BoxesDim2: Int64;
    NumClasses2: Int64;
  end;
const
  MAX_REASONABLE_CLASSES = 10000;
var
  Shape: TArray<Int64>;
  Interpretation: TShapeInterpretation;
  NumBoxes, ActualNumFeatures, ElementSize, CoeffFeatureIndex: Int64;
  BoxColumns: Integer;
  HasObjectness: Boolean;
  Objectness: Single;
  I, J, K, NumClasses, BestClass, CandidateIndex: Integer;
  NCHWLayout, Float16Flag: Boolean;
  OutputDataPtr, P: Pointer;
  CandidateList: TList<TCandidateDetection>;
  XFactor, YFactor: Extended;
  Float16: TFloat16;
  PlaceData: array[0..3] of Single;
  X, Y, W, H, ClassScore, MaxClassScore: Single;
  Rect, OriginRect: TRectF;
  DetectionRect: TRect;
  NMSResultIndices, SortedIndices: TList<Integer>;
  Flags: TArray<Boolean>;
  TensorBytes: UInt64;
begin
  FilteredDetections := nil;
  if not Assigned(DetectTensor) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: DetectTensor is nil';
    Exit(False);
  end;
  if DetectTensor.device.device_type <> kDLCPU then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: DetectTensor must be on CPU';
    Exit(False);
  end;
  if not Assigned(DetectTensor.shape) or (DetectTensor.ndim <> 3) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: DetectTensor must have a nonnil 3-D shape';
    Exit(False);
  end;
  Shape := TTVMRuntime.GetTensorShape(DetectTensor);
  if (Length(Shape) <> 3) or (Shape[0] > 1) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: Unexpected DetectTensor shape (expected 3D, batch 1 or dynamic; multi-image batch not supported): ' +
        ShapeToString(Shape);
    Exit(False);
  end;
  if (Layout = ylNmsFree) or ((Layout = ylAuto) and IsNmsFreeLayout(DetectTensor, NumMaskCoeffs)) then
    Exit(DecodeNmsFreeDetections(DetectTensor, ConfidenceThreshold, ModelInputSize,
      OriginalImageSize, NumMaskCoeffs, FilteredDetections, Error));
  if (Shape[0] <> 1) or (Shape[1] < 1) or (Shape[2] < 1) or
    (Shape[1] > High(Integer)) or (Shape[2] > High(Integer)) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.DecodeAndFilterDetections: dense tensor must have actual batch size 1 and positive dimensions that fit decoder indexing';
      Exit(False);
    end;
  HasObjectness := Layout = ylDenseObjectness;
  if HasObjectness then
    BoxColumns := 5
  else
    BoxColumns := 4;
  FillChar(Interpretation, SizeOf(TShapeInterpretation), 0);
  Interpretation.FeaturesDim1 := Shape[1];
  Interpretation.BoxesDim1 := Shape[2];
  Interpretation.NumClasses1 := Interpretation.FeaturesDim1 - BoxColumns - NumMaskCoeffs;
  Interpretation.BoxesDim2 := Shape[1];
  Interpretation.FeaturesDim2 := Shape[2];
  Interpretation.NumClasses2 := Interpretation.FeaturesDim2 - BoxColumns - NumMaskCoeffs;
  repeat
    if (Interpretation.NumClasses1 > 0) and (Interpretation.NumClasses2 <= 0) then
    begin
      ActualNumFeatures := Interpretation.FeaturesDim1;
      NumBoxes := Interpretation.BoxesDim1;
      NumClasses := Interpretation.NumClasses1;
      NCHWLayout := True;
      Break;
    end;
    if (Interpretation.NumClasses2 > 0) and (Interpretation.NumClasses1 <= 0) then
    begin
      ActualNumFeatures := Interpretation.FeaturesDim2;
      NumBoxes := Interpretation.BoxesDim2;
      NumClasses := Interpretation.NumClasses2;
      NCHWLayout := False;
      Break;
    end;
    if (Interpretation.NumClasses1 > 0) and (Interpretation.NumClasses2 > 0) then
    begin
      if (Interpretation.NumClasses1 > MAX_REASONABLE_CLASSES) and
        (Interpretation.NumClasses2 <= MAX_REASONABLE_CLASSES) then
        begin
          ActualNumFeatures := Interpretation.FeaturesDim2;
          NumBoxes := Interpretation.BoxesDim2;
          NumClasses := Interpretation.NumClasses2;
          NCHWLayout := False;
          Break;
        end;
      if (Interpretation.NumClasses2 > MAX_REASONABLE_CLASSES) and
        (Interpretation.NumClasses1 <= MAX_REASONABLE_CLASSES) then
        begin
          ActualNumFeatures := Interpretation.FeaturesDim1;
          NumBoxes := Interpretation.BoxesDim1;
          NumClasses := Interpretation.NumClasses1;
          NCHWLayout := True;
          Break;
        end;
      ActualNumFeatures := Interpretation.FeaturesDim1;
      NumBoxes := Interpretation.BoxesDim1;
      NumClasses := Interpretation.NumClasses1;
      NCHWLayout := True;
      Break;
    end;
    if Assigned(Error) then
      Error^ := Format(
        'TYolo11.DecodeAndFilterDetections: Calculated NumClasses is invalid for both layouts. NCHW=%d, NHWC=%d. Check tensor shape or NumMaskCoeffs parameter',
        [
          Interpretation.NumClasses1,
          Interpretation.NumClasses2
        ]
      );
    Exit(False);
  until True;
  if DetectTensor.dtype.code <> UInt8(kDLFloat) then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.DecodeAndFilterDetections: DetectTensor data type must be Float, got code = %d',
        [DetectTensor.dtype.code]);
    Exit(False);
  end;
  case DetectTensor.dtype.bits of
    16:
      begin
        Float16Flag := True;
        ElementSize := SizeOf(TFloat16);
      end;
    32:
      begin
        Float16Flag := False;
        ElementSize := SizeOf(Single);
      end;
  else
    if Assigned(Error) then
      Error^ := Format('TYolo11.DecodeAndFilterDetections: DetectTensor float type must be 16 or 32 bits, got %d bits',
        [DetectTensor.dtype.bits]);
    Exit(False);
  end;
  if DetectTensor.dtype.lanes <> 1 then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: dense tensor must have one scalar lane';
    Exit(False);
  end;
  if Assigned(DetectTensor.strides) and
    ((DetectTensor.strides[0] <> Shape[1] * Shape[2]) or
    (DetectTensor.strides[1] <> Shape[2]) or (DetectTensor.strides[2] <> 1)) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.DecodeAndFilterDetections: dense tensor must have contiguous row-major strides';
      Exit(False);
    end;
  if DetectTensor.byte_offset <> 0 then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: nonzero tensor byte offsets are not supported';
    Exit(False);
  end;
  OutputDataPtr := DetectTensor.data;
  if not Assigned(OutputDataPtr) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.DecodeAndFilterDetections: DetectTensor data pointer is nil';
    Exit(False);
  end;
  TensorBytes := UInt64(Shape[1]) * UInt64(Shape[2]) * UInt64(ElementSize);
  if (TensorBytes > UInt64(High(NativeInt))) or
    (TensorBytes - 1 > UInt64(High(NativeUInt) - NativeUInt(OutputDataPtr))) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.DecodeAndFilterDetections: dense tensor byte span overflows address space';
      Exit(False);
    end;
  CandidateList := TList<TCandidateDetection>.Create;
  try
    XFactor := OriginalImageSize.cx / ModelInputSize.cx;
    YFactor := OriginalImageSize.cy / ModelInputSize.cy;
    for I := 0 to NumBoxes - 1 do
    begin
      for J := 0 to 3 do
      begin
        P := GetDataElementPtr(OutputDataPtr, I, J, NumBoxes, ActualNumFeatures, NCHWLayout, ElementSize);
        if Float16Flag then
        begin
          Float16.Value := PWord(P)^;
          PlaceData[J] := Float16.ToSingle;
        end
        else
          PlaceData[J] := PSingle(P)^;
      end;
      X := PlaceData[0];
      Y := PlaceData[1];
      W := PlaceData[2];
      H := PlaceData[3];
      Objectness := 1;
      if HasObjectness then
      begin
        P := GetDataElementPtr(OutputDataPtr, I, 4, NumBoxes, ActualNumFeatures, NCHWLayout, ElementSize);
        if Float16Flag then
        begin
          Float16.Value := PWord(P)^;
          Objectness := Float16.ToSingle;
        end
        else
          Objectness := PSingle(P)^;
      end;
      MaxClassScore := 0;
      BestClass := -1;
      for J := 0 to NumClasses - 1 do
      begin
        P := GetDataElementPtr(OutputDataPtr, I, BoxColumns + J, NumBoxes, ActualNumFeatures,
          NCHWLayout, ElementSize);
        if Float16Flag then
        begin
          Float16.Value := PWord(P)^;
          ClassScore := Float16.ToSingle;
        end
        else
          ClassScore := PSingle(P)^;
        if ClassScore > MaxClassScore then
        begin
          MaxClassScore := ClassScore;
          BestClass := J;
        end;
      end;
      MaxClassScore := MaxClassScore * Objectness;
      if MaxClassScore > 1 + 1e-3 then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.DecodeAndFilterDetections: class score %.4f exceeds 1 at box %d. ' +
            'This output is not a field of probabilities, so the dense layout does not apply. ' +
            'Check that the model belongs to the v5u/v8/v11 family and was exported with the detection head intact',
            [MaxClassScore, I]);
        Exit(False);
      end;
      if MaxClassScore >= ConfidenceThreshold then
      begin
        Rect := {$IFDEF FPC}RectFOf{$ELSE}TRectF.Create{$ENDIF}(
          X - W * 0.5,
          Y - H * 0.5,
          X + W * 0.5,
          Y + H * 0.5);
        OriginRect := {$IFDEF FPC}RectFOf{$ELSE}TRectF.Create{$ENDIF}(
          Rect.TopLeft.X * XFactor,
          Rect.TopLeft.Y * YFactor,
          Rect.BottomRight.X * XFactor,
          Rect.BottomRight.Y * YFactor);
        DetectionRect := TRect.Create(
          Max(
            0,
            Min(
              OriginalImageSize.cx - 1,
              Round(OriginRect.TopLeft.X)
            )
          ),
          Max(
            0,
            Min(
              OriginalImageSize.cy - 1,
              Round(OriginRect.TopLeft.Y)
            )
          ),
          Max(
            0,
            Min(
              OriginalImageSize.cx - 1,
              Round(OriginRect.BottomRight.X)
            )
          ),
          Max(
            0,
            Min(
              OriginalImageSize.cy - 1,
              Round(OriginRect.BottomRight.Y)
            )
          )
        );
        if (DetectionRect.Right > DetectionRect.Left) and
          (DetectionRect.Bottom > DetectionRect.Top) then
            CandidateList.Add(TCandidateDetection.Make(DetectionRect, OriginRect, BestClass,
              MaxClassScore, I));
      end
    end;
    NMSResultIndices := TList<Integer>.Create;
    try
      if CandidateList.Count > 0 then
      begin
        SortedIndices := TList<Integer>.Create;
        try
          for I := 0 to CandidateList.Count - 1 do SortedIndices.Add(I);
          {$IFDEF FPC}
          SortedIndices.Sort(TIndexComparer<Integer>.Create(CandidateList));
          {$ELSE}
          SortedIndices.Sort(TComparer<Integer>.Construct(
            function(const L, R: Integer): Integer
            begin
              if CandidateList[L].Score > CandidateList[R].Score then
                Result := -1
              else
                if CandidateList[L].Score < CandidateList[R].Score then
                  Result := 1
              else
                Result := 0;
            end));
          {$ENDIF}
          SetLength(Flags, CandidateList.Count);
          for I := 0 to SortedIndices.Count - 1 do
          begin
            CandidateIndex := SortedIndices[I];
            if not Flags[CandidateIndex] then
            begin
              NMSResultIndices.Add(CandidateIndex);
              for J := I + 1 to SortedIndices.Count - 1 do
              begin
                K := SortedIndices[J];
                if not Flags[K] and (CandidateList[CandidateIndex].&Class = CandidateList[K].&Class) and
                  (CalcIoU(CandidateList[CandidateIndex].Place, CandidateList[K].Place) >= NMSThreshold) then
                    Flags[K] := True;
              end;
            end;
          end;
        finally
          SortedIndices.Free;
        end;
      end;
      SetLength(FilteredDetections, NMSResultIndices.Count);
      for K := 0 to NMSResultIndices.Count - 1 do
      begin
        CandidateIndex := NMSResultIndices[K];
        FilteredDetections[K] := TYolo11IntermediateDetection.Make(
          CandidateList[CandidateIndex].Place,
          CandidateList[CandidateIndex].&Class,
          CandidateList[CandidateIndex].Score,
          nil,
          CandidateList[CandidateIndex].OriginalIndex
        );
        FilteredDetections[K].PlaceExact := CandidateList[CandidateIndex].PlaceExact;
        SetLength(FilteredDetections[K].TrailingColumns, NumMaskCoeffs);
        for J := 0 to NumMaskCoeffs - 1 do
        begin
          CoeffFeatureIndex := BoxColumns + NumClasses + J;
          P := GetDataElementPtr(
            OutputDataPtr,
            FilteredDetections[K].OriginalIndex,
            CoeffFeatureIndex,
            NumBoxes,
            ActualNumFeatures,
            NCHWLayout,
            ElementSize
          );
          if Float16Flag then
          begin
            Float16.Value := PWord(P)^;
            FilteredDetections[K].TrailingColumns[J] := Float16.ToSingle;
          end
          else
            FilteredDetections[K].TrailingColumns[J] := PSingle(P)^;
        end;
      end;
    finally
      NMSResultIndices.Free;
    end;
  finally
    CandidateList.Free;
  end;
  Result := True;
end;

class function TYolo11.GenerateMasks(const FilteredDetections: TYolo11IntermediateDetections;
  const PrototypeDataPtr: Pointer; const NumPrototypes: Integer; const BoxSpaceSize, MaskPrototypeSize: TSize;
  out Detections: TYolo11Detections; const Error: PString): Boolean;
var
  XFactor, YFactor: Extended;
  I, NumMaskCoeffs: Integer;
  Detection: TYolo11IntermediateDetection;
  CoeffsPtr: PSingle;
  RawMaskDataSize: Int64;
  RawMaskData: TArray<Single>;
  Mask32F, SigmoidMask32F, ResizedMask32F, Mask8U: IMat;
  ScaledBox: TRect;
begin
  Result := False;
  Detections := nil;
  try
    if Length(FilteredDetections) = 0 then Exit(True);
    if not Assigned(PrototypeDataPtr) or (NumPrototypes <= 0) or (MaskPrototypeSize.cx <= 0) or
      (MaskPrototypeSize.cy <= 0) then
      begin
        if Assigned(Error) then
          Error^ := 'TYolo11.GenerateMasks: Invalid prototype data provided';
        Exit(False);
      end;
    SetLength(Detections, Length(FilteredDetections));
    XFactor := MaskPrototypeSize.cx / BoxSpaceSize.cx;
    YFactor := MaskPrototypeSize.cy / BoxSpaceSize.cy;
    for I := Low(FilteredDetections) to High(FilteredDetections) do
    begin
      Detection := FilteredDetections[I];
      NumMaskCoeffs := Length(Detection.TrailingColumns);
      if NumMaskCoeffs = 0 then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.GenerateMasks: No mask coefficients for detection %d', [I]);
        Exit(False);
      end;
      if NumMaskCoeffs > NumPrototypes then
        NumMaskCoeffs := NumPrototypes;
      CoeffsPtr := Pointer(Detection.TrailingColumns);
      RawMaskDataSize := MaskPrototypeSize.cx * MaskPrototypeSize.cy;
      if RawMaskDataSize > High(Integer) then
      begin
        if Assigned(Error) then
          Error^ := 'TYolo11.GenerateMasks: RawMaskDataSize exceeds Integer range';
        Exit(False);
      end;
      SetLength(RawMaskData, RawMaskDataSize);
      try
        if not MultiplyMaskCoeffsProtos(CoeffsPtr, PSingle(PrototypeDataPtr),
          PSingle(Pointer(RawMaskData)), NumMaskCoeffs, RawMaskDataSize) then
          begin
            if Assigned(Error) then
              Error^ := Format('TYolo11.GenerateMasks: Matrix multiplication failed for detection %d', [I]);
            Exit(False);
          end;
        Mask32F := nil;
        SigmoidMask32F := nil;
        ResizedMask32F := nil;
        Mask8U := nil;
        try
          Mask32F := NewMat(MaskPrototypeSize.cx, MaskPrototypeSize.cy, CV_MAKETYPE(CV_32F, 1));
          System.Move(Pointer(RawMaskData)^, Mask32F.DataPtr^, Length(RawMaskData) * SizeOf(Single));
          SigmoidMask32F := Mask32F.Clone;
          ApplySigmoidToImage(SigmoidMask32F);
          ScaledBox := TRect.Create(
            Round(Detection.Place.TopLeft.X * XFactor),
            Round(Detection.Place.TopLeft.Y * YFactor),
            Round(Detection.Place.BottomRight.X * XFactor),
            Round(Detection.Place.BottomRight.Y * YFactor)
          );
          ScaledBox := TRect.Create(
            Max(
              0,
              Min(
                MaskPrototypeSize.cx - 1,
                ScaledBox.TopLeft.X
              )
            ),
            Max(
              0,
              Min(
                MaskPrototypeSize.cy - 1,
                ScaledBox.TopLeft.Y
              )
            ),
            Max(
              0,
              Min(
                MaskPrototypeSize.cx - 1,
                ScaledBox.BottomRight.X
              )
            ),
            Max(
              0,
              Min(
                MaskPrototypeSize.cy - 1,
                ScaledBox.BottomRight.Y
              )
            )
          );
          if (ScaledBox.Right <= ScaledBox.Left) or (ScaledBox.Bottom <= ScaledBox.Top) then
            Mask8U := NewMat(Detection.Place.Width, Detection.Place.Height, CV_MAKETYPE(CV_8U, 1)).Fill(0)
          else begin
            ResizedMask32F := SigmoidMask32F.Crop(ScaledBox.Left, ScaledBox.Top, ScaledBox.Width,
              ScaledBox.Height)
                                            .Resize(Detection.Place.Width, Detection.Place.Height,
                                              INTER_LINEAR);
            Mask8U := ResizedMask32F.ConvertTo(CV_MAKETYPE(CV_8U, 1), 255, 0);
          end;
          Detections[I] := TYolo11Detection.Make(
            Detection.Place,
            Detection.&Class,
            Detection.Score,
            Mask8U
          );
        finally
          Mask32F := nil;
          SigmoidMask32F := nil;
          ResizedMask32F := nil;
          Mask8U := nil;
        end;
      finally
        RawMaskData := nil;
      end;
    end;
    Result := True;
  finally
    if not Result then
    begin
      for I := Low(Detections) to High(Detections) do
        if Assigned(Detections[I].Mask) then Detections[I].Mask := nil;
      Detections := nil;
    end;
  end;
end;

class function TYolo11.PrepareProtoData(const PrototypeTensor: PDLTensor; out PrototypeDataPtr: Pointer;
  out NumPrototypes: Integer; out MaskPrototypeSize: TSize; const Error: PString): Boolean;
var
  Shape: TArray<Int64>;
  ExpectedNDim: Integer;
begin
  if not Assigned(PrototypeTensor) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.PrepareProtoData: PrototypeTensor is nil';
    Exit(False);
  end;
  if PrototypeTensor.device.device_type <> kDLCPU then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.PrepareProtoData: PrototypeTensor must be on CPU';
    Exit(False);
  end;
  Shape := TTVMRuntime.GetTensorShape(PrototypeTensor);
  ExpectedNDim := 4;
  if (Length(Shape) <> ExpectedNDim) and (Length(Shape) <> ExpectedNDim - 1) then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.PrepareProtoData: Unexpected PrototypeTensor shape dimension count (expected %d or %d): %s',
        [ExpectedNDim, ExpectedNDim - 1, ShapeToString(Shape)]);
    Exit(False);
  end;
  if (Length(Shape) = ExpectedNDim) and (Shape[0] > 1) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.PrepareProtoData: PrototypeTensor batch size must be 1 or dynamic (multi-image batch not supported), but got ' +
        IntToStr(Shape[0]);
    Exit(False);
  end;
  FillChar(MaskPrototypeSize, SizeOf(TSize), 0);
  if Length(Shape) = 4 then
  begin
    NumPrototypes := Shape[1];
    MaskPrototypeSize := TSize.Create(Shape[3], Shape[2]);
  end
  else begin
    NumPrototypes := Shape[0];
    MaskPrototypeSize := TSize.Create(Shape[2], Shape[1]);
  end;
  if (NumPrototypes <= 0) or (MaskPrototypeSize.cy <= 0) or (MaskPrototypeSize.cx <= 0) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.PrepareProtoData: Invalid dimensions extracted from PrototypeTensor shape';
    Exit(False);
  end;
  if not TTVMRuntime.CheckTensor(PrototypeTensor, [kDLFloat], SizeOf(Single) * 8) then
  begin
    if Assigned(Error) then
      Error^ := Format('TYolo11.PrepareProtoData: PrototypeTensor data type must be Float32, got code=%d, bits=%d',
        [PrototypeTensor.dtype.code, PrototypeTensor.dtype.bits]);
    Exit(False);
  end;
  PrototypeDataPtr := PrototypeTensor.data;
  if not Assigned(PrototypeDataPtr) then
  begin
    if Assigned(Error) then
      Error^ := 'TYolo11.PrepareProtoData: PrototypeTensor data pointer is nil';
    Exit(False);
  end;
  Result := True;
end;

class function TYolo11.Detect(var Descriptor: TModelDescriptor; const Image: IMat;
  const OutputDeviceType, OutputDeviceID: Integer; const Threshold, NMSThreshold: Single;
  out Scores: TArray<Single>; out Labels: TArray<Integer>; out Places: TArray<TRect>;
  const InputTensorName: string; const Error: PString; const Layout: TYoloLayout): Boolean;
var
  OutputTensors: TArray<PDLTensor>;
  S: string;
begin
  OutputTensors := nil;
  try
    if not TTVMRuntime.InitializeOutputTensors(Descriptor, OutputTensors, OutputDeviceType,
      OutputDeviceID, @S) then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.Detect: Failed to initialize output tensors. Details: %s', [S]);
        Exit(False);
      end;
    Result := Detect(Descriptor, Image, OutputTensors, Threshold, NMSThreshold, Scores, Labels,
      Places, InputTensorName, Error, Layout);
  finally
    OutputTensors := nil;
  end;
end;

class function TYolo11.RunAndDecode(var Descriptor: TModelDescriptor; const Image: IMat;
  const OutputDeviceType, OutputDeviceID: Integer; const Threshold, NMSThreshold: Single;
  const NumTrailing: Integer; out Found: TYolo11IntermediateDetections;
  const InputTensorName: string; const Error: PString; const Layout: TYoloLayout): Boolean;
var
  OutputTensors: TArray<PDLTensor>;
  Scores: TArray<Single>;
  Labels: TArray<Integer>;
  Places: TArray<TRect>;
  S: string;
begin
  Found := nil;
  OutputTensors := nil;
  try
    if not TTVMRuntime.InitializeOutputTensors(Descriptor, OutputTensors, OutputDeviceType,
      OutputDeviceID, @S) then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.RunAndDecode: Failed to initialize output tensors. Details: %s', [S]);
        Exit(False);
      end;
    Result := Detect(Descriptor, Image, OutputTensors, Threshold, NMSThreshold,
      Scores, Labels, Places, InputTensorName, Error, Layout, NumTrailing, @Found);
  finally
    OutputTensors := nil;
  end;
end;

class function TYolo11.Detect(var Descriptor: TModelDescriptor; const Image: IMat;
  var OutputTensors: TArray<PDLTensor>; const Threshold, NMSThreshold: Single; out Scores: TArray<Single>;
  out Labels: TArray<Integer>; out Places: TArray<TRect>; const InputTensorName: string; const Error: PString;
  const Layout: TYoloLayout; const NumTrailing: Integer; const Found: PYolo11IntermediateDetections): Boolean;
type
  TTensors = record
    Input: PDLTensor;
    Image: PDLTensor;
    Output: PDLTensor;
    CpuOutput: PDLTensor;
  end;
var
  Tensors: TTensors;
  S: string;
  I, TensorIndex: Int64;
  Scale: Single;
  ModelSize, OriginSize: TSize;
  Intermediate: TYolo11IntermediateDetections;
  Img: IMat;
begin
  Scores := nil;
  Labels := nil;
  Places := nil;
  FillChar(Tensors, SizeOf(TTensors), 0);
  try
    if not TTVMRuntime.GetInputTensorIndex(Descriptor, InputTensorName, TensorIndex, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Detect: Failed to get input tensor index for "%s". Details: %s',
          [InputTensorName, S]);
      Exit(False);
    end;
    if not TTVMRuntime.GetInputTensor(Descriptor, TensorIndex, Tensors.Input, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Detect: Failed to get input tensor schema (index %d). Details: %s',
          [TensorIndex, S]);
      Exit(False);
    end;
    if not Assigned(Tensors.Input) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Detect: Input tensor definition handle is nil';
      Exit(False);
    end;
    case Tensors.Input.dtype.code of
      Ord(kDLFloat): Scale := 1 / 255;
      Ord(kDLUInt), Ord(kDLInt) : Scale := 1;
    else
      if Assigned(Error) then
        Error^ := 'TYolo11.Detect: Unsupported input tensor data type';
      Exit(False);
    end;
    if (Tensors.Input.ndim <> 4) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Detect: Input tensor definition must be 4D';
      Exit(False);
    end;
    if Tensors.Input.shape[Yolo11Tensor.CIndex] <> 3 then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Detect: Invalid channel count in input tensor definition (expected 3, NCHW format)';
      Exit(False);
    end;
    ModelSize := TSize.Create(
      Tensors.Input.shape[Yolo11Tensor.WIndex],
      Tensors.Input.shape[Yolo11Tensor.HIndex]
    );
    if (ModelSize.cx <= 0) or (ModelSize.cy <= 0) then
    begin
      ModelSize.cx := Image.Width;
      ModelSize.cy := Image.Height;
    end;
    OriginSize.cx := Image.Width;
    OriginSize.cy := Image.Height;
    Img := nil;
    try
      if not TTVMRuntime.ConvertImageToTensorFormat(Image, Img, Tensors.Input, Yolo11Tensor, Scale,
        INTER_CUBIC, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Detect: Image preprocessing (ConvertImageToTensorFormat) failed. Details: %s', [S]);
          Exit(False);
        end;
      if not Assigned(Img) then
      begin
        if Assigned(Error) then
          Error^ := 'TYolo11.Detect: Image preprocessing (ConvertImageToTensorFormat) returned nil image, but reported success';
        Exit(False);
      end;
      Img := ConvertBgrToRgb(Img);
      if not TTVMRuntime.ConvertImageToDLTensor(Img, Descriptor.Device, Yolo11Tensor, Tensors.Image,
        TDLDataTypeCode(Tensors.Input.dtype.code), Tensors.Input.dtype.bits,
        Tensors.Input.dtype.lanes, 1, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Detect: Failed to create input DLTensor (ConvertImageToDLTensor). Details: %s', [S]);
          Exit(False);
        end;
    finally
      Img := nil;
    end;
    if not Assigned(Tensors.Image) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Detect: Failed to create input DLTensor (ConvertImageToDLTensor returned success but tensor is nil)';
      Exit(False);
    end;
    if not TTVMRuntime.RunInference(Descriptor, InputTensorName, Tensors.Image, OutputTensors, Error) then
      Exit(False);
    if Length(OutputTensors) <> 1 then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Detect: Yolo11 model must have exactly one output tensor, got %d',
          [Length(OutputTensors)]);
      Exit(False);
    end;
    Tensors.Output := OutputTensors[0];
    Tensors.CpuOutput := TTVMRuntime.GetOrCreateCpuTensor(Tensors.Output, Tensors.CpuOutput,
      'Output', 'TYolo11.Detect', Error);
    if not Assigned(Tensors.CpuOutput) then Exit(False);
    if not TYolo11.DecodeAndFilterDetections(Tensors.CpuOutput, Threshold, NMSThreshold,
      ModelSize, OriginSize, NumTrailing, Intermediate, Error, Layout) then Exit(False);
    if Assigned(Found) then
      Found^ := Intermediate;
    SetLength(Scores, Length(Intermediate));
    SetLength(Labels, Length(Intermediate));
    SetLength(Places, Length(Intermediate));
    for I := 0 to High(Intermediate) do
    begin
      Scores[I] := Intermediate[I].Score;
      Labels[I] := Intermediate[I].&Class;
      Places[I] := Intermediate[I].Place;
    end;
  finally
    if Assigned(Tensors.Image) then
      FreeMem(Tensors.Image);
    if Assigned(Tensors.CpuOutput) and (Tensors.CpuOutput <> Tensors.Output) then
      FreeMem(Tensors.CpuOutput);
  end;
  Result := True;
end;

class function TYolo11.Segment(var Descriptor: TModelDescriptor; const Image: IMat;
  const OutputDeviceType, OutputDeviceID: Integer; const Threshold, NMSThreshold: Single;
  out Detections: TYolo11Detections; const InputTensorName: string; const Error: PString;
  const Layout: TYoloLayout): Boolean;
var
  OutputTensors: TArray<PDLTensor>;
  S: string;
begin
  OutputTensors := nil;
  try
    if not TTVMRuntime.InitializeOutputTensors(Descriptor, OutputTensors, OutputDeviceType,
      OutputDeviceID, @S) then
      begin
        if Assigned(Error) then
          Error^ := Format('TYolo11.Segment: Failed to initialize output tensors. Details: %s', [S]);
        Exit(False);
      end;
    Result := Segment(Descriptor, Image, OutputTensors, Threshold, NMSThreshold, Detections,
      InputTensorName, Error, Layout);
  finally
    OutputTensors := nil;
  end;
end;

class function TYolo11.Segment(var Descriptor: TModelDescriptor; const Image: IMat;
  var OutputTensors: TArray<PDLTensor>; const Threshold, NMSThreshold: Single;
  out Detections: TYolo11Detections; const InputTensorName: string; const Error: PString;
  const Layout: TYoloLayout): Boolean;
const
  EXPECTED_NUM_PROTOTYPES = 32;
var
  Tensors: TYolo11Tensors;
  S: string;
  TensorIndex: Int64;
  Scale: Single;
  Img: IMat;
  IntermediateDetections: TYolo11IntermediateDetections;
  ModelSize, OriginSize, MaskPrototypeSize: TSize;
  DetectionTensorIndex, PrototypeTensorIndex, NumPrototypes: Integer;
  Shape0, Shape1: TArray<Int64>;
  PrototypeDataPtr: Pointer;
begin
  FillChar(Tensors, SizeOf(Tensors), 0);
  try
    if not TTVMRuntime.GetInputTensorIndex(Descriptor, InputTensorName, TensorIndex, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Segment: Failed to get input tensor index for "%s". Details: %s',
          [InputTensorName, S]);
      Exit(False);
    end;
    if not TTVMRuntime.GetInputTensor(Descriptor, TensorIndex, Tensors.Input, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Segment: Failed to get input tensor schema (index %d). Details: %s',
          [TensorIndex, S]);
      Exit(False);
    end;
    if not Assigned(Tensors.Input) then
    begin
      if Assigned(Error) then
        Error^ := 'TYolo11.Segment: GetInputTensor succeeded but returned a nil input tensor schema';
      Exit(False);
    end;
    case Tensors.Input.dtype.code of
      Ord(kDLFloat): Scale := 1 / 255;
      Ord(kDLInt), Ord(kDLUInt): Scale := 1;
    else
      if Assigned(Error) then
        Error^ := Format('TYolo11.Segment: Unsupported input tensor data type code: %d. Cannot determine image scale',
          [Tensors.Input.dtype.code]);
      Exit(False);
    end;
    Img := nil;
    try
      if not TTVMRuntime.ConvertImageToTensorFormat(Image, Img, Tensors.Input, Yolo11Tensor, Scale,
        INTER_CUBIC, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Segment: Image preprocessing (ConvertImageToTensorFormat) failed. Details: %s', [S]);
          Exit(False);
        end;
      if not Assigned(Img) then
      begin
        if Assigned(Error) then
          Error^ := 'TYolo11.Segment: Image preprocessing (ConvertImageToTensorFormat) reported success but returned nil image';
        Exit(False);
      end;
      Img := ConvertBgrToRgb(Img);
      if not TTVMRuntime.ConvertImageToDLTensor(Img, Descriptor.Device, Yolo11Tensor, Tensors.Image,
        TDLDataTypeCode(Tensors.Input.dtype.code), Tensors.Input.dtype.bits,
        Tensors.Input.dtype.lanes, 1, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TYolo11.Segment: Failed to create input DLTensor (ConvertImageToDLTensor). Details: %s', [S]);
          Exit(False);
        end;
    finally
      Img := nil;
    end;
    if not Assigned(Tensors.Image) then
    begin
      if Assigned(Error) then
        if Assigned(Error) then
          Error^ := 'TYolo11.Segment: Failed to create input DLTensor (ConvertImageToDLTensor reported success but tensor is nil)';
      Exit(False);
    end;
    if not TTVMRuntime.RunInference(Descriptor, InputTensorName, Tensors.Image, OutputTensors, Error) then
      Exit(False);
    if Length(OutputTensors) <> 2 then
    begin
      if Assigned(Error) then
        Error^ := Format('TYolo11.Segment: Expected exactly 2 output tensors (Detection and Prototype), but got %d',
          [Length(OutputTensors)]);
      Exit(False);
    end;
    Shape0 := nil;
    Shape1 := nil;
    try
      Shape0 := TTVMRuntime.GetTensorShape(OutputTensors[0]);
      Shape1 := TTVMRuntime.GetTensorShape(OutputTensors[1]);
      repeat
        if (Length(Shape0) = 3) and (Length(Shape1) >= 4) then
        begin
          DetectionTensorIndex := 0;
          PrototypeTensorIndex := 1;
          Break;
        end;
        if (Length(Shape1) = 3) and (Length(Shape0) >= 4) then
        begin
          DetectionTensorIndex := 1;
          PrototypeTensorIndex := 0;
          Break;
        end;
        if (Length(Shape0) = 3) and (Length(Shape1) = 3) then
        begin
          if (Shape0[0] = EXPECTED_NUM_PROTOTYPES) and (Shape1[0] <> EXPECTED_NUM_PROTOTYPES) then
          begin
            PrototypeTensorIndex := 0;
            DetectionTensorIndex := 1;
            Break;
          end;
          if (Shape1[0] = EXPECTED_NUM_PROTOTYPES) and (Shape0[0] <> EXPECTED_NUM_PROTOTYPES) then
          begin
            PrototypeTensorIndex := 1;
            DetectionTensorIndex := 0;
            Break;
          end;
        end;
        if Assigned(Error) then
          Error^ := Format(
            'TYolo11.Segment: Cannot determine Detection/Prototype tensors. Shape0: %s (ndim %d), Shape1: %s (ndim %d). Expected one 3D tensor (detections) and one 3D/4D tensor (prototypes with %d protos)',
            [
              ShapeToString(Shape0),
              Length(Shape0),
              ShapeToString(Shape1),
              Length(Shape1),
              EXPECTED_NUM_PROTOTYPES
            ]
          );
        Exit(False);
      until True;
    finally
      Shape0 := nil;
      Shape1 := nil;
    end;
    Tensors.Detection := TTVMRuntime.GetOrCreateCpuTensor(OutputTensors[DetectionTensorIndex],
      Tensors.CpuDetection, 'Detection', 'TYolo11.Segment', Error);
    if not Assigned(Tensors.Detection) then Exit(False);
    Tensors.Prototype := TTVMRuntime.GetOrCreateCpuTensor(OutputTensors[PrototypeTensorIndex],
      Tensors.CpuPrototype, 'Prototype', 'TYolo11.Segment', Error);
    if not Assigned(Tensors.Prototype) then Exit(False);
    ModelSize := TSize.Create(Tensors.Input.shape[Yolo11Tensor.WIndex],
      Tensors.Input.shape[Yolo11Tensor.HIndex]);
    if (ModelSize.cx <= 0) or (ModelSize.cy <= 0) then
    begin
      ModelSize.cx := Image.Width;
      ModelSize.cy := Image.Height;
    end;
    OriginSize := TSize.Create(Image.Width, Image.Height);
    if not TYolo11.DecodeAndFilterDetections(Tensors.Detection, Threshold, NMSThreshold, ModelSize,
      OriginSize, 32, IntermediateDetections, Error, Layout) then
        Exit(False);
    if not TYolo11.PrepareProtoData(Tensors.Prototype, PrototypeDataPtr, NumPrototypes,
      MaskPrototypeSize, Error) then
        Exit(False);
    if not TYolo11.GenerateMasks(IntermediateDetections, PrototypeDataPtr, NumPrototypes,
      OriginSize, MaskPrototypeSize, Detections, Error) then
        Exit(False);
  finally
    if Assigned(Tensors.Image) then
      FreeMem(Tensors.Image);
    if Assigned(Tensors.CpuDetection) then
      FreeMem(Tensors.CpuDetection);
    if Assigned(Tensors.CpuPrototype) then
      FreeMem(Tensors.CpuPrototype);
  end;
  Result := True;
end;

end.
