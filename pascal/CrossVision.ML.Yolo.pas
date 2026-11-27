{ ************************************************************************** }
{                                                                            }
{ CrossVision.ML.Yolo                                                        }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossVision.ML.Yolo;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  CrossPascal.NativeIO, DLPack, CrossVision.Mat, CrossVision.ML, CrossVision.ML.Depth,
  TVM.Yolo.Profile, TVM.CpuGraph;

const
  YoloDefaultResultByteLimit = 512 * 1024 * 1024;

{ Pascal opens the Unicode path; OpenCV decodes the bytes as uint8 BGR. }
function LoadYoloImage(const FileName: string;
  const MaxEncodedBytes: NativeInt = 64 * 1024 * 1024): IMat;

{ Low-level factories for an already selected/validated TVM runtime and model.
  Each object exposes only its selected task interface. Lifetime may hold the
  verified bundle/runtime leases for as long as the model is referenced.
  Integer image boxes enclose the restored float box (floor/ceil); keypoints
  and oriented boxes retain their Float32 coordinates. Binary masks are
  original-size CV_8UC1 with values 0/255. }
function CreateYoloClassifier(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): IClassifier;
function CreateYoloDetector(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): IDetector;
function CreateYoloInstanceSegmenter(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): IInstanceSegmenter;
function CreateYoloPoseEstimator(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): IPoseEstimator;
function CreateYoloOrientedDetector(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): IOrientedDetector;
function CreateYoloSemanticSegmenter(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): ISemanticSegmenter;
function CreateYoloDepthEstimator(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface = nil;
  const ResultByteLimit: Int64 = YoloDefaultResultByteLimit;
  const ShouldStop: TCpuGraphStop = nil): IDepthEstimator;

implementation

uses
  {$IFDEF FPC}SysUtils, Classes, Math, Types{$ELSE}System.SysUtils, System.Classes, System.Math, System.Types{$ENDIF},
  CrossVision.Geometry.Types, OpenCV, OpenCVTypes, OpenCVConsts, TVM.Yolo.Native,
  TVM.Yolo.MatInput, TVM.Yolo.RunOptions;

type
  TYoloAdapter = class(TInterfacedObject)
  protected
    FCore: TYoloNativeModel;
    FProfile: TYoloProfile;
    FLifetime: IInterface;
    FResultByteLimit: Int64;
    FShouldStop: TCpuGraphStop;
    function Run(const Image: IMat): TYoloNativeResult;
    function ClassNameAt(const ID: Integer): string;
    procedure RequirePlaneSpace(const Width, Height, Count, BytesPerPixel: Integer);
  public
    constructor Create(const Profile: TYoloProfile; const ExpectedTask: TYoloTask;
      const ModelLibrary: string; const Device: TDLDeviceType; const Lifetime: IInterface;
      const ResultByteLimit: Int64; const ShouldStop: TCpuGraphStop);
    destructor Destroy; override;
  end;
  TYoloClassifier = class(TYoloAdapter, IClassifier)
    function Classify(const Image: IMat): TClassificationArray;
  end;
  TYoloDetector = class(TYoloAdapter, IDetector)
    function Detect(const Image: IMat): TDetectionArray;
  end;
  TYoloInstanceSegmenter = class(TYoloAdapter, IInstanceSegmenter)
    function SegmentInstances(const Image: IMat): TInstanceArray;
  end;
  TYoloPoseEstimator = class(TYoloAdapter, IPoseEstimator)
    function EstimatePoses(const Image: IMat): TPoseArray;
  end;
  TYoloOrientedDetector = class(TYoloAdapter, IOrientedDetector)
    function DetectOriented(const Image: IMat): TOrientedDetectionArray;
  end;
  TYoloSemanticSegmenter = class(TYoloAdapter, ISemanticSegmenter)
    function SegmentClasses(const Image: IMat): TSemanticMaskArray;
  end;
  TYoloDepthEstimator = class(TYoloAdapter, IDepthEstimator)
    function EstimateDepth(const Image: IMat): TDepthMap;
  end;

function LoadYoloImage(const FileName: string; const MaxEncodedBytes: NativeInt): IMat;
var
  Stream: TStream;
  Bytes: TBytes;
  Handle: TMat;
begin
  if not CV.Module.Loaded or not Assigned(CV.ImgCodecs.ImDecode) or not Assigned(CV.Core.DeleteMat) then
    raise Exception.Create('Select the OpenCV image runtime before loading a YOLO image');
  if (MaxEncodedBytes <= 0) or (MaxEncodedBytes > High(Integer)) then
    raise EArgumentException.Create('Invalid YOLO encoded image memory limit');
  Stream := TNativeReadStream.Create(FileName);
  try
    if (Stream.Size <= 0) or (Stream.Size > MaxEncodedBytes) then
      raise EArgumentException.Create('YOLO image file is empty or exceeds the configured encoded byte limit');
    SetLength(Bytes, Stream.Size); Stream.ReadBuffer(Bytes[0], Length(Bytes));
  finally
    Stream.Free;
  end;
  Handle := nil;
  try
    if not CV.ImgCodecs.ImDecode(Pointer(Bytes), Length(Bytes), IMREAD_COLOR, @Handle) or
      (Handle = nil) then
        raise EArgumentException.Create('Cannot decode the YOLO image: ' + CV.GetError);
    Result := WrapMat(Handle, True);
    Handle := nil;
    if Result.IsEmpty or (Result.GetDepth <> CV_8U) or (Result.GetChannels <> 3) then
      raise EArgumentException.Create('YOLO decoder did not return uint8 BGR pixels');
  finally
    if Handle <> nil then CV.Core.DeleteMat(@Handle);
  end;
end;

function IntegerBox(const Box: TRectF): TRectI;
begin
  Result.Left := Floor(Box.Left);
  Result.Top := Floor(Box.Top);
  Result.Right := Ceil(Box.Right);
  Result.Bottom := Ceil(Box.Bottom);
end;

function NewPlane(const Width, Height, MatType, BytesPerPixel: Integer): IMat;
begin
  Result := NewMat(Width, Height, MatType);
  if (Result = nil) or Result.IsEmpty or (Result.GetMatType <> MatType) or
    (Result.GetWidth <> Width) or (Result.GetHeight <> Height) or (Result.DataPtr = nil) or
    (Int64(Result.GetStep) < Int64(Width) * BytesPerPixel) then
      raise Exception.Create('Cannot allocate the YOLO result image');
end;

function BinaryPlane(const Bytes: TBytes; const Width, Height: Integer): IMat;
var
  X, Y: Integer;
  Row: PByte;
begin
  if Length(Bytes) <> Int64(Width) * Height then
    raise EArgumentException.Create('YOLO binary mask shape changed');
  Result := NewPlane(Width, Height, CV_MAKETYPE(CV_8U, 1), 1);
  for Y := 0 to Height - 1 do
  begin
    Row := PByte(NativeUInt(Result.DataPtr) + NativeUInt(Y) * NativeUInt(Result.GetStep));
    for X := 0 to Width - 1 do
    begin
      if Bytes[Y * Width + X] > 1 then
        raise EArgumentException.Create('Invalid YOLO binary mask value');
      PByte(NativeUInt(Row) + NativeUInt(X))^ := Bytes[Y * Width + X] * 255;
    end;
  end;
end;

constructor TYoloAdapter.Create(const Profile: TYoloProfile; const ExpectedTask: TYoloTask;
  const ModelLibrary: string; const Device: TDLDeviceType; const Lifetime: IInterface;
  const ResultByteLimit: Int64; const ShouldStop: TCpuGraphStop);
begin
  inherited Create;
  Profile.Validate;
  if Profile.Task <> ExpectedTask then
    raise EArgumentException.Create('YOLO profile does not implement this task');
  FLifetime := Lifetime;
  FProfile := CopyYoloProfile(Profile);
  FResultByteLimit := ResultByteLimit;
  FShouldStop := ShouldStop;
  FCore := TYoloNativeModel.Create(FProfile, ModelLibrary, Device, ResultByteLimit, ShouldStop);
end;

destructor TYoloAdapter.Destroy;
begin
  FCore.Free;
  FLifetime := nil;
  inherited;
end;

function TYoloAdapter.Run(const Image: IMat): TYoloNativeResult;
var
  Prepared: TYoloPreparedImage;
begin
  CheckYoloRunStop(FShouldStop);
  if (Image = nil) or Image.IsEmpty then
    raise EArgumentException.Create('YOLO input image is empty');
  Prepared.Width := Image.GetWidth;
  Prepared.Height := Image.GetHeight;
  if FProfile.Preprocessing = ypPillowClassification then
    Prepared.Pixels := PrepareYoloMatPillowClassification(Image, FProfile.InputWidth,
      FProfile.InputHeight, FProfile.Mean, FProfile.Std)
  else
    Prepared.Pixels := PrepareYoloMatPixels(Image,
      FProfile.PixelPlan(Prepared.Width, Prepared.Height), FProfile.Mean, FProfile.Std);
  CheckYoloRunStop(FShouldStop);
  Result := FCore.Infer(Prepared);
end;

function TYoloAdapter.ClassNameAt(const ID: Integer): string;
begin
  if (ID < 0) or (ID >= FProfile.ClassCount) then
    raise EArgumentException.Create('YOLO class ID is outside its profile');
  Result := FProfile.ClassNames[ID];
end;

procedure TYoloAdapter.RequirePlaneSpace(const Width, Height, Count, BytesPerPixel: Integer);
var
  Pixels: Int64;
begin
  if (Width <= 0) or (Height <= 0) or (Count < 0) or (BytesPerPixel <= 0) then
    raise EArgumentException.Create('Invalid YOLO result plane dimensions');
  if Count = 0 then Exit;
  Pixels := Int64(Width) * Height;
  if (Pixels > FResultByteLimit div BytesPerPixel) or
    (Count > FResultByteLimit div (Pixels * BytesPerPixel)) then
      raise EArgumentException.Create('YOLO typed result exceeds the configured result memory limit');
end;

function TYoloClassifier.Classify(const Image: IMat): TClassificationArray;
var
  Found: TYoloNativeResult;
  I: Integer;
begin
  Found := Run(Image); SetLength(Result, Length(Found.Classifications));
  for I := 0 to High(Result) do
  begin
    Result[I].ClassId := Found.Classifications[I].ClassID;
    Result[I].ClassName := ClassNameAt(Result[I].ClassId);
    Result[I].Score := Found.Classifications[I].Score;
  end;
end;

function TYoloDetector.Detect(const Image: IMat): TDetectionArray;
var
  Found: TYoloNativeResult;
  I: Integer;
begin
  Found := Run(Image); SetLength(Result, Length(Found.Detections));
  for I := 0 to High(Result) do
  begin
    Result[I].Box := IntegerBox(Found.Detections[I].Box);
    Result[I].ClassId := Found.Detections[I].ClassID;
    Result[I].ClassName := ClassNameAt(Result[I].ClassId);
    Result[I].Score := Found.Detections[I].Score;
  end;
end;

function TYoloInstanceSegmenter.SegmentInstances(const Image: IMat): TInstanceArray;
var
  Found: TYoloNativeResult;
  I: Integer;
begin
  Found := Run(Image);
  if Length(Found.InstanceMasks) <> Length(Found.Detections) then
    raise Exception.Create('YOLO instance result count changed');
  RequirePlaneSpace(Found.Width, Found.Height, Length(Found.InstanceMasks), 1);
  SetLength(Result, Length(Found.Detections));
  for I := 0 to High(Result) do
  begin
    Result[I].Box := IntegerBox(Found.Detections[I].Box);
    Result[I].ClassId := Found.Detections[I].ClassID;
    Result[I].ClassName := ClassNameAt(Result[I].ClassId);
    Result[I].Score := Found.Detections[I].Score;
    CheckYoloRunStop(FShouldStop);
    Result[I].Mask := BinaryPlane(Found.InstanceMasks[I], Found.Width, Found.Height);
  end;
end;

function TYoloPoseEstimator.EstimatePoses(const Image: IMat): TPoseArray;
var
  Found: TYoloNativeResult;
  I, K, Offset: Integer;
begin
  Found := Run(Image); SetLength(Result, Length(Found.Detections));
  for I := 0 to High(Result) do
  begin
    Result[I].Box := IntegerBox(Found.Detections[I].Box);
    Result[I].ClassId := Found.Detections[I].ClassID;
    Result[I].ClassName := ClassNameAt(Result[I].ClassId);
    Result[I].Score := Found.Detections[I].Score;
    SetLength(Result[I].Keypoints, FProfile.KeypointCount);
    for K := 0 to High(Result[I].Keypoints) do
    begin
      Offset := K * FProfile.KeypointDimensions;
      Result[I].Keypoints[K].X := Found.Detections[I].Extra[Offset];
      Result[I].Keypoints[K].Y := Found.Detections[I].Extra[Offset + 1];
      Result[I].Keypoints[K].Visibility := 1; { A two-coordinate head supplies no visibility score. }
      if FProfile.KeypointDimensions = 3 then
        Result[I].Keypoints[K].Visibility := Found.Detections[I].Extra[Offset + 2];
    end;
  end;
end;

function TYoloOrientedDetector.DetectOriented(const Image: IMat): TOrientedDetectionArray;
var
  Found: TYoloNativeResult;
  I: Integer;
begin
  Found := Run(Image); SetLength(Result, Length(Found.Oriented));
  for I := 0 to High(Result) do
  begin
    Result[I].Box.CenterX := Found.Oriented[I].Box.CenterX;
    Result[I].Box.CenterY := Found.Oriented[I].Box.CenterY;
    Result[I].Box.Width := Found.Oriented[I].Box.Width;
    Result[I].Box.Height := Found.Oriented[I].Box.Height;
    Result[I].Box.Angle := Found.Oriented[I].Box.Angle;
    Result[I].ClassId := Found.Oriented[I].ClassID;
    Result[I].ClassName := ClassNameAt(Result[I].ClassId);
    Result[I].Score := Found.Oriented[I].Score;
  end;
end;

function TYoloSemanticSegmenter.SegmentClasses(const Image: IMat): TSemanticMaskArray;
var
  Found: TYoloNativeResult;
  I, X, Y, ClassID: Integer;
  Rows: TArray<PByte>;
  Row: PByte;
begin
  Found := Run(Image);
  RequirePlaneSpace(Found.Width, Found.Height, FProfile.ClassCount, 1);
  if Length(Found.ClassMap) <> Int64(Found.Width) * Found.Height then
    raise EArgumentException.Create('YOLO semantic map shape changed');
  SetLength(Result, FProfile.ClassCount);
  SetLength(Rows, Length(Result));
  for I := 0 to High(Result) do
  begin
    Result[I].ClassId := I;
    Result[I].ClassName := ClassNameAt(I);
    CheckYoloRunStop(FShouldStop);
    Result[I].Mask := NewPlane(Found.Width, Found.Height, CV_MAKETYPE(CV_8U, 1), 1);
    for Y := 0 to Found.Height - 1 do
    begin
      Row := PByte(NativeUInt(Result[I].Mask.DataPtr) + NativeUInt(Y) * NativeUInt(Result[I].Mask.GetStep));
      FillChar(Row^, Found.Width, 0);
    end;
  end;
  { Each source pixel belongs to exactly one class. Preserve separate 0/255
    planes without scanning the source map again for every class. }
  for Y := 0 to Found.Height - 1 do
  begin
    if (Y and 31) = 0 then CheckYoloRunStop(FShouldStop);
    for I := 0 to High(Result) do
      Rows[I] := PByte(NativeUInt(Result[I].Mask.DataPtr) + NativeUInt(Y) * NativeUInt(Result[I].Mask.GetStep));
    for X := 0 to Found.Width - 1 do
    begin
      ClassID := Found.ClassMap[Y * Found.Width + X];
      if ClassID >= Length(Rows) then
        raise EArgumentException.Create('YOLO semantic map contains an invalid class ID');
      PByte(NativeUInt(Rows[ClassID]) + NativeUInt(X))^ := 255;
    end;
  end;
  CheckYoloRunStop(FShouldStop);
end;

function TYoloDepthEstimator.EstimateDepth(const Image: IMat): TDepthMap;
var
  Found: TYoloNativeResult;
  Y: Integer;
  Row: Pointer;
begin
  Found := Run(Image); RequirePlaneSpace(Found.Width, Found.Height, 1, SizeOf(Single));
  Result.Units := duMeters;
  Result.Values := NewPlane(Found.Width, Found.Height, CV_MAKETYPE(CV_32F, 1), SizeOf(Single));
  for Y := 0 to Found.Height - 1 do
  begin
    Row := Pointer(NativeUInt(Result.Values.DataPtr) + NativeUInt(Y) * NativeUInt(Result.Values.GetStep));
    Move(Found.Depth[Y * Found.Width], Row^, Found.Width * SizeOf(Single));
  end;
end;

function CreateYoloClassifier(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): IClassifier;
begin
  Result := TYoloClassifier.Create(Profile, ytClassification, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

function CreateYoloDetector(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): IDetector;
begin
  Result := TYoloDetector.Create(Profile, ytDetection, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

function CreateYoloInstanceSegmenter(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): IInstanceSegmenter;
begin
  Result := TYoloInstanceSegmenter.Create(Profile, ytInstances, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

function CreateYoloPoseEstimator(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): IPoseEstimator;
begin
  Result := TYoloPoseEstimator.Create(Profile, ytPose, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

function CreateYoloOrientedDetector(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): IOrientedDetector;
begin
  Result := TYoloOrientedDetector.Create(Profile, ytOriented, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

function CreateYoloSemanticSegmenter(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): ISemanticSegmenter;
begin
  Result := TYoloSemanticSegmenter.Create(Profile, ytSemantic, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

function CreateYoloDepthEstimator(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const Lifetime: IInterface; const ResultByteLimit: Int64;
  const ShouldStop: TCpuGraphStop): IDepthEstimator;
begin
  Result := TYoloDepthEstimator.Create(Profile, ytDepth, ModelLibrary, Device, Lifetime,
    ResultByteLimit, ShouldStop);
end;

end.
