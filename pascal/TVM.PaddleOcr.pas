{ Copyright © 2026 Yuriy Pisarev. MIT license; see LICENSE. }

unit TVM.PaddleOcr;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes, Windows{$ELSE}System.SysUtils, System.Classes, Winapi.Windows{$ENDIF},
  DLPack, TVM, TVM.CpuGraph, TVM.CudaExecution, CrossVision.Mat, OpenCVTypes;

const
  PaddleDetectionSide = 960;
  PaddleRecognitionHeight = 48;
  PaddleRecognitionWidth = 320;
  PaddleRecognitionClasses = 18385;
  PaddleDictionarySHA256 = '1ea29636956177e400af712d9782e7693f3fb25f98617bed10479d2965a836fd';
  PaddleClipperSHA256 = '8f12ed9cfb68b3b8ab7b86b8c554e5905b2390ec63a5c917e83988c699315d68';

type
  TPaddleQuad = array[0..3] of TOCVPoint2f;
  TPaddleDetection = record
    Quad: TPaddleQuad;
    Score: Single;
  end;
  TPaddleDetections = TArray<TPaddleDetection>;
  TPaddleTextLine = record
    Detection: TPaddleDetection;
    Text: string;
    Score: Single;
  end;
  TPaddleTextLines = TArray<TPaddleTextLine>;
  TPaddleProgress = procedure(const MessageText: string) of object;

  TPaddleOcrModel = class
  private
    FGraph: TCpuGraph;
    FCuda: TCudaExecution;
    FWorkspace: TCudaWorkspace;
    FLease: TFileStream;
    FDevice: TDLDeviceType;
    FShape: TArray<Int64>;
    FKind: string;
    FDeviceName: string;
    FLibraryName: string;
    FDictionary: TArray<string>;
  public
    constructor Create(const Bundle, RuntimeDirectory: string);
    destructor Destroy; override;
    function Run(const Input: TArray<Single>; out OutputShape: TArray<Int64>;
      const ShouldStop: TCpuGraphStop = nil): TArray<Single>;
    function Recognize(const Image: IMat; out Score: Single;
      const ShouldStop: TCpuGraphStop = nil): string;
    property Kind: string read FKind;
    property DeviceName: string read FDeviceName;
  end;

  TPaddleDbPostprocess = class
  private type
    TOffset = function(XY: PInt64; Count: Integer; Distance: Double; OutputXY: PInt64;
      Capacity: Integer; var OutputCount, OutputPaths: Integer): Integer; cdecl;
  private
    FModule: HMODULE;
    FLease: TFileStream;
    FOffset: TOffset;
    function Unclip(const Quad: TPaddleQuad; out Expanded: TPaddleQuad): Boolean;
  public
    constructor Create(const ClipperLibrary: string);
    destructor Destroy; override;
    function Detect(const Probabilities: TArray<Single>;
      const ImageWidth, ImageHeight: Integer): TPaddleDetections;
  end;

{ These preparation and decoding routines are shared by applications and tests. }
function PaddleDetectionInput(const Image: IMat): TArray<Single>;
function PaddleRecognitionInput(const Image: IMat): TArray<Single>;
function PaddleCrop(const Image: IMat; const Quad: TPaddleQuad): IMat;
function PaddleLoadDictionary(const FileName: string): TArray<string>;
function PaddleDecodeCTC(const Probabilities: TArray<Single>; const Classes: Integer;
  const Dictionary: TArray<string>; out Score: Single): string;
function PaddleRead(const Detector, Recognizer: TPaddleOcrModel; const Postprocess: TPaddleDbPostprocess;
  const Image: IMat; const Progress: TPaddleProgress = nil;
  const ShouldStop: TCpuGraphStop = nil): TPaddleTextLines;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, CrossPascal.Json, CrossPascal.SHA256,
  QwenCuda, OpenCV, OpenCVConsts;

type
  TContour = TArray<TOCVPoint>;
  PCollector = ^TCollector;
  TCollector = record
    Seen: Integer;
    Failed: Boolean;
    Contours: TArray<TContour>;
  end;

procedure CheckImage(const Image: IMat);
begin
  RequireCpu(
    (Image <> nil) and not Image.IsEmpty and (Image.Depth = CV_8U) and (Image.Channels = 3) and (Image.Width > 0) and (Image.Height > 0) and (Image.Width <= 4096) and (Image.Height <= 4096) and (Image.DataPtr <> nil) and (Image.Step >= NativeInt(Image.Width) * 3),
    'PaddleOCR requires a BGR8 image up to 4096 pixels per side'
  );
end;

function PaddleDetectionInput(const Image: IMat): TArray<Single>;
const
  Mean: array[0..2] of Single = (0.485, 0.456, 0.406);
  Std: array[0..2] of Single = (0.229, 0.224, 0.225);
var
  Square, Region, Resized, Floats: IMat;
  Side, X, Y, C: Integer;
  Raw: TMat;
  Row: PSingle;
  Value: Single;
begin
  CheckImage(Image);
  Side := Max(Image.Width, Image.Height);
  Square := ZerosMat(Side, Side, CV_MAKETYPE(CV_8U, 3));
  RequireCpu((Square <> nil) and not Square.IsEmpty, 'PaddleOCR square allocation failed');
  Raw := Square.Raw;
  RequireCpu(CV.Core.SetTo(@Raw, OCVScalar(0, 0, 0, 0), nil) and (Raw = Square.Raw),
    'PaddleOCR square initialization failed');
  Region := RegionView(Square, (Side - Image.Width) div 2, (Side - Image.Height) div 2,
    Image.Width, Image.Height);
  RequireCpu((Region <> nil) and not Region.IsEmpty and CopyInto(Image, Region),
    'PaddleOCR source image copy failed');
  Resized := Square.Resize(PaddleDetectionSide, PaddleDetectionSide, INTER_LINEAR);
  RequireCpu((Resized <> nil) and not Resized.IsEmpty, 'PaddleOCR detector resize failed');
  Floats := Resized.ConvertTo(CV_MAKETYPE(CV_32F, 3), 1 / 255, 0);
  RequireCpu((Floats <> nil) and not Floats.IsEmpty and (Floats.Depth = CV_32F),
    'PaddleOCR detector float conversion failed');
  SetLength(Result, 3 * PaddleDetectionSide * PaddleDetectionSide);
  for Y := 0 to PaddleDetectionSide - 1 do
  begin
    Row := PSingle(PByte(Floats.DataPtr) + NativeInt(Y) * Floats.Step);
    for X := 0 to PaddleDetectionSide - 1 do
      for C := 0 to 2 do
      begin
        Value := Row[X * 3 + C] - Mean[C];
        Value := Value / Std[C];
        Result[(C * PaddleDetectionSide + Y) * PaddleDetectionSide + X] := Value;
      end;
  end;
end;

function PaddleRecognitionInput(const Image: IMat): TArray<Single>;
var
  Width, X, Y, C: Integer;
  Resized, Floats: IMat;
  Row: PSingle;
  Value: Single;
begin
  CheckImage(Image);
  Width := Min(PaddleRecognitionWidth,
    (Int64(PaddleRecognitionHeight) * Image.Width + Image.Height - 1) div Image.Height);
  Width := Max(1, Width);
  Resized := Image.Resize(Width, PaddleRecognitionHeight, INTER_LINEAR);
  RequireCpu((Resized <> nil) and not Resized.IsEmpty, 'PaddleOCR recognizer resize failed');
  Floats := Resized.ConvertTo(CV_MAKETYPE(CV_32F, 3), 1 / 255, 0);
  RequireCpu((Floats <> nil) and not Floats.IsEmpty and (Floats.Depth = CV_32F),
    'PaddleOCR recognizer float conversion failed');
  SetLength(Result, 3 * PaddleRecognitionHeight * PaddleRecognitionWidth);
  FillChar(Result[0], Length(Result) * SizeOf(Single), 0);
  for Y := 0 to PaddleRecognitionHeight - 1 do
  begin
    Row := PSingle(PByte(Floats.DataPtr) + NativeInt(Y) * Floats.Step);
    for X := 0 to Width - 1 do
      for C := 0 to 2 do
      begin
        Value := Row[X * 3 + C] * 2;
        Result[(C * PaddleRecognitionHeight + Y) * PaddleRecognitionWidth + X] := Value - 1;
      end;
  end;
end;

function PaddleLoadDictionary(const FileName: string): TArray<string>;
var
  Stream: TFileStream;
  Data: UTF8String;
  Text: string;
  I, Start, Count: Integer;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu(SHA256Stream(Stream) = PaddleDictionarySHA256, 'PaddleOCR dictionary SHA-256 differs');
    Stream.Position := 0;
    SetLength(Data, Stream.Size);
    Stream.ReadBuffer(Data[1], Length(Data));
  finally
    Stream.Free;
  end;
  Text := UTF8Decode(Data);
  SetLength(Result, PaddleRecognitionClasses);
  Result[0] := '';
  Start := 1;
  Count := 1;
  for I := 1 to Length(Text) + 1 do
    if (I > Length(Text)) or (Text[I] = #10) then
    begin
      if (I > Length(Text)) and (Start = I) then Break;
      RequireCpu(Count < PaddleRecognitionClasses - 1, 'PaddleOCR dictionary has too many entries');
      Result[Count] := Copy(Text, Start, I - Start);
      if (Result[Count] <> '') and (Result[Count][Length(Result[Count])] = #13) then
        Delete(Result[Count], Length(Result[Count]), 1);
      RequireCpu(Result[Count] <> '', 'PaddleOCR dictionary contains an empty entry');
      Inc(Count);
      Start := I + 1;
    end;
  RequireCpu(Count = PaddleRecognitionClasses - 1, 'PaddleOCR dictionary size differs');
  Result[High(Result)] := ' ';
end;

function PaddleDecodeCTC(const Probabilities: TArray<Single>; const Classes: Integer;
  const Dictionary: TArray<string>; out Score: Single): string;
var
  Step, C, Best, Previous, Count: Integer;
  Value, Maximum: Single;
  Total: Double;
begin
  RequireCpu((Classes > 1) and (Length(Dictionary) = Classes) and
    (Length(Probabilities) > 0) and (Length(Probabilities) mod Classes = 0),
    'PaddleOCR CTC dimensions differ');
  Result := '';
  Score := 0;
  Count := 0;
  Total := 0;
  Previous := -1;
  for Step := 0 to Length(Probabilities) div Classes - 1 do
  begin
    Maximum := -1;
    Best := 0;
    for C := 0 to Classes - 1 do
    begin
      Value := Probabilities[Step * Classes + C];
      RequireCpu(not IsNan(Value) and not IsInfinite(Value) and (Value >= -1e-6) and (Value <= 1.00001),
        'PaddleOCR CTC received an invalid probability');
      if Value > Maximum then
      begin
        Maximum := Value;
        Best := C;
      end;
    end;
    if (Best <> 0) and (Best <> Previous) then
    begin
      Result := Result + Dictionary[Best];
      Total := Total + Maximum;
      Inc(Count);
    end;
    Previous := Best;
  end;
  if Count > 0 then Score := Total / Count;
end;

constructor TPaddleOcrModel.Create(const Bundle, RuntimeDirectory: string);
var
  Document: TJSONValue;
  Root, Artifact, IO, OCR: TJSONObject;
  Architecture, Device, Folder: string;
begin
  inherited Create;
  Folder := ExpandFileName(Bundle);
  Document := ReadModelJSON(BundleMember(Folder, 'manifest.json'));
  try
    Root := ModelObject(Document, 'PaddleOCR manifest');
    RequireCpu((ModelInt(Root.GetValue('format_version'), 'format version') = 1) and
      (ModelText(Root.GetValue('adapter')) = 'paddleocr'), 'Unsupported PaddleOCR manifest');
    FKind := ModelText(Root.GetValue('task_kind'));
    RequireCpu((FKind = 'ocr_detection') or (FKind = 'ocr_recognition'), 'Unsupported PaddleOCR task');
    IO := ModelObject(Root.GetValue('io'), 'PaddleOCR input');
    FShape := ModelShape(IO.GetValue('shape'));
    RequireCpu((Length(FShape) = 4) and (FShape[0] = 1) and (FShape[1] = 3) and
      (ModelText(IO.GetValue('name')) = 'x') and (ModelText(IO.GetValue('layout')) = 'NCHW') and
      (ModelText(IO.GetValue('dtype')) = 'float32'), 'Invalid PaddleOCR input contract');
    OCR := ModelObject(Root.GetValue('ocr'), 'PaddleOCR settings');
    if FKind = 'ocr_detection' then
      RequireCpu((FShape[2] = PaddleDetectionSide) and (FShape[3] = PaddleDetectionSide) and
        (ModelText(OCR.GetValue('preprocessing')) = 'centered-square-bgr-imagenet') and
        (Abs(ModelFloat(OCR.GetValue('threshold'), 'threshold') - 0.3) < 1e-7) and
        (Abs(ModelFloat(OCR.GetValue('box_threshold'), 'box threshold') - 0.6) < 1e-7) and
        (Abs(ModelFloat(OCR.GetValue('unclip_ratio'), 'unclip ratio') - 1.5) < 1e-7),
        'Unsupported PaddleOCR detection profile')
    else begin
      RequireCpu((FShape[2] = PaddleRecognitionHeight) and (FShape[3] = PaddleRecognitionWidth) and
        (ModelText(OCR.GetValue('preprocessing')) = 'bgr-resize-pad-minus-one-to-one') and
        (ModelInt(OCR.GetValue('classes'), 'classes') = PaddleRecognitionClasses) and
        (ModelInt(OCR.GetValue('blank_index'), 'blank index') = 0) and
        (ModelText(OCR.GetValue('dictionary_sha256')) = PaddleDictionarySHA256),
        'Unsupported PaddleOCR recognition profile');
      FDictionary := PaddleLoadDictionary(BundleMember(Folder, 'ppocrv5_dict.txt'));
    end;
    Artifact := ModelObject(Root.GetValue('artifact'), 'PaddleOCR artifact');
    RequireCpu((ModelText(Artifact.GetValue('os')) = 'windows') and
      (ModelText(Artifact.GetValue('arch')) = 'x86_64') and
      (ModelText(Artifact.GetValue('dtype')) = 'float32') and
      (ModelText(Artifact.GetValue('entry')) = 'main'), 'Unsupported PaddleOCR artifact');
    Device := ModelText(Artifact.GetValue('device'));
    RequireCpu((Device = 'cpu') or (Device = 'cuda'), 'Unsupported PaddleOCR device');
    FDevice := kDLCPU;
    FDeviceName := 'CPU';
    if Device = 'cuda' then
    begin
      FDevice := kDLCUDA;
      FCuda := TCudaExecution.Create;
      Architecture := ModelText(Artifact.GetValue('cuda_arch'));
      FCuda.RequireArchitecture(Architecture);
      FDeviceName := FCuda.DeviceName + ' (' + Architecture + ')';
    end;
    FLibraryName := BundleMember(Folder, ModelText(Artifact.GetValue('file')));
    FLease := TFileStream.Create(FLibraryName, fmOpenRead or fmShareDenyWrite);
    RequireCpu(SameText(SHA256Stream(FLease), ModelText(Artifact.GetValue('hash'))),
      'PaddleOCR model SHA-256 differs');
    InitializeRuntime(IncludeTrailingPathDelimiter(ExpandFileName(RuntimeDirectory)), FDevice = kDLCUDA);
    if FCuda <> nil then FCuda.Enter;
    try
      if FCuda <> nil then RequireCpu(FCuda.FreeBytes >= UInt64(FLease.Size) + 512 * 1024 * 1024,
        'PaddleOCR requires at least the model size plus 512 MiB of free CUDA memory');
      FGraph := TCpuGraph.Create(FLibraryName, FDevice);
      if FCuda <> nil then
        FWorkspace := TCudaWorkspace.Create(FCuda, FLibraryName, 512 * 1024 * 1024);
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    Document.Free;
  end;
end;

destructor TPaddleOcrModel.Destroy;
begin
  if FCuda <> nil then FCuda.Enter;
  try
    FWorkspace.Free;
    FGraph.Free;
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
  FCuda.Free;
  FLease.Free;
  inherited;
end;

function TPaddleOcrModel.Run(const Input: TArray<Single>; out OutputShape: TArray<Int64>;
  const ShouldStop: TCpuGraphStop): TArray<Single>;
var
  Argument, Returned, Output: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
begin
  RequireCpu(Length(Input) = TensorElements(FShape), 'PaddleOCR input element count differs');
  if Assigned(ShouldStop) and ShouldStop() then raise ECpuGraphStopped.Create('PaddleOCR stopped');
  Argument := nil;
  Returned := nil;
  Output := nil;
  if FCuda <> nil then FCuda.Enter;
  try
    Argument := DeviceTensor(FShape, kDLFloat, 32, Pointer(Input), FDevice);
    if FWorkspace <> nil then
      Value := FWorkspace.Run(FGraph, [Argument], ShouldStop)
    else
      Value := FGraph.Run([Argument], ShouldStop);
    Returned := Value.VObj;
    if Value.TypeIndex = kTVMFFITensor then
    begin
      OutputShape := TTVMRuntime.GetTensorShape(TTVMRuntime.TensorDLTensor(Returned));
      Result := ReadFloatTensor(Returned);
    end
    else begin
      Output := TensorItem(Returned, 0);
      OutputShape := TTVMRuntime.GetTensorShape(TTVMRuntime.TensorDLTensor(Output));
      Result := ReadFloatTensor(Output);
    end;
    if FKind = 'ocr_detection' then
      RequireCpu((Length(OutputShape) = 4) and (OutputShape[0] = 1) and (OutputShape[1] = 1) and
        (OutputShape[2] = PaddleDetectionSide) and (OutputShape[3] = PaddleDetectionSide),
        'PaddleOCR detector output shape differs')
    else
      RequireCpu((Length(OutputShape) = 3) and (OutputShape[0] = 1) and
        (OutputShape[1] = 40) and (OutputShape[2] = PaddleRecognitionClasses),
        'PaddleOCR recognizer output shape differs');
  finally
    TTVMRuntime.ObjectRelease(Output);
    TTVMRuntime.ObjectRelease(Returned);
    TTVMRuntime.ObjectRelease(Argument);
    if FCuda <> nil then FCuda.Leave;
  end;
end;

function TPaddleOcrModel.Recognize(const Image: IMat; out Score: Single;
  const ShouldStop: TCpuGraphStop): string;
var
  Shape: TArray<Int64>;
  Probabilities: TArray<Single>;
begin
  RequireCpu(FKind = 'ocr_recognition', 'Select a PaddleOCR recognition model');
  Probabilities := Run(PaddleRecognitionInput(Image), Shape, ShouldStop);
  Result := PaddleDecodeCTC(Probabilities, PaddleRecognitionClasses, FDictionary, Score);
end;

procedure CollectContour(Count: Integer; From, Data: Pointer); cdecl;
var
  Collector: PCollector;
  I: Integer;
begin
  Collector := PCollector(Data);
  if (Collector = nil) or Collector.Failed then Exit;
  Inc(Collector.Seen);
  if (Collector.Seen > 1000) or (Count < 3) then Exit;
  if (From = nil) or (Count > PaddleDetectionSide * PaddleDetectionSide) then
  begin
    Collector.Failed := True;
    Exit;
  end;
  try
    I := Length(Collector.Contours);
    SetLength(Collector.Contours, I + 1);
    SetLength(Collector.Contours[I], Count);
    Move(PByte(From)^, Collector.Contours[I][0], NativeInt(Count) * SizeOf(TOCVPoint));
  except
    Collector.Failed := True;
  end;
end;

function MiniBox(const Points: TContour; out Quad: TPaddleQuad): Single;
var
  Mat: IMat;
  Rect: TOCVRotatedRect;
  I, J: Integer;
  DX, DY, CA, SA: Double;
  Sorted: TPaddleQuad;
  Corner: TOCVPoint2f;
begin
  RequireCpu(Length(Points) >= 3, 'PaddleOCR contour has fewer than three vertices');
  Mat := NewMat(Length(Points), 1, CV_MAKETYPE(CV_32S, 2));
  RequireCpu((Mat <> nil) and not Mat.IsEmpty, 'PaddleOCR contour allocation failed');
  Move(Points[0], PByte(Mat.DataPtr)^, Length(Points) * SizeOf(TOCVPoint));
  RequireCpu(CV.Core.MinAreaRect(Mat.Raw, @Rect), 'PaddleOCR minimum-area rectangle failed');
  RequireCpu(not IsNan(Rect.Width) and not IsInfinite(Rect.Width) and
    not IsNan(Rect.Height) and not IsInfinite(Rect.Height) and
    not IsNan(Rect.CenterX) and not IsInfinite(Rect.CenterX) and
    not IsNan(Rect.CenterY) and not IsInfinite(Rect.CenterY) and
    not IsNan(Rect.Angle) and not IsInfinite(Rect.Angle), 'PaddleOCR invalid rectangle geometry');
  Result := Min(Rect.Width, Rect.Height);
  CA := Cos(Rect.Angle * Pi / 180);
  SA := Sin(Rect.Angle * Pi / 180);
  for I := 0 to 3 do
  begin
    case I of
      0:
        begin
          DX := -Rect.Width / 2;
          DY := -Rect.Height / 2;
        end;
      1:
        begin
          DX := Rect.Width / 2;
          DY := -Rect.Height / 2;
        end;
      2:
        begin
          DX := Rect.Width / 2;
          DY := Rect.Height / 2;
        end;
    else begin
        DX := -Rect.Width / 2;
        DY := Rect.Height / 2;
      end;
    end;
    Sorted[I] := OCVPoint2f(Rect.CenterX + DX * CA - DY * SA, Rect.CenterY + DX * SA + DY * CA);
  end;
  for I := 1 to 3 do
  begin
    Corner := Sorted[I];
    J := I - 1;
    while (J >= 0) and (Sorted[J].X > Corner.X) do
    begin
      Sorted[J + 1] := Sorted[J];
      Dec(J);
    end;
    Sorted[J + 1] := Corner;
  end;
  if Sorted[1].Y > Sorted[0].Y then
  begin
    Quad[0] := Sorted[0];
    Quad[3] := Sorted[1];
  end
  else begin
    Quad[0] := Sorted[1];
    Quad[3] := Sorted[0];
  end;
  if Sorted[3].Y > Sorted[2].Y then
  begin
    Quad[1] := Sorted[2];
    Quad[2] := Sorted[3];
  end
  else begin
    Quad[1] := Sorted[3];
    Quad[2] := Sorted[2];
  end;
end;

constructor TPaddleDbPostprocess.Create(const ClipperLibrary: string);
var
  FileName: string;
begin
  inherited Create;
  FileName := ExpandFileName(ClipperLibrary);
  FLease := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  RequireCpu(SHA256Stream(FLease) = PaddleClipperSHA256, 'PaddleOCR Clipper SHA-256 differs');
  FModule := LoadLibraryExW(PWideChar(FileName), 0, $00000100 or $00001000);
  RequireCpu(FModule <> 0, 'PaddleOCR Clipper library failed to load');
  FOffset := TOffset(GetProcAddress(FModule, 'ppocr_db_jt_round_v1'));
  RequireCpu(Assigned(FOffset), 'PaddleOCR Clipper ABI is missing');
end;

destructor TPaddleDbPostprocess.Destroy;
begin
  if FModule <> 0 then FreeLibrary(FModule);
  FLease.Free;
  inherited;
end;

function TPaddleDbPostprocess.Unclip(const Quad: TPaddleQuad; out Expanded: TPaddleQuad): Boolean;
var
  Input: array[0..7] of Int64;
  Output: TArray<Int64>;
  Polygon: TContour;
  I, J, Count, Paths, Code: Integer;
  Area, Perimeter, Distance, X0, Y0, X1, Y1: Double;
begin
  Area := 0;
  Perimeter := 0;
  for I := 0 to 3 do
  begin
    J := (I + 1) mod 4;
    X0 := Quad[I].X;
    Y0 := Quad[I].Y;
    X1 := Quad[J].X;
    Y1 := Quad[J].Y;
    Area := Area + X0 * Y1 - X1 * Y0;
    Perimeter := Perimeter + Hypot(X1 - X0, Y1 - Y0);
    Input[2 * I] := Trunc(Quad[I].X);
    Input[2 * I + 1] := Trunc(Quad[I].Y);
  end;
  Result := False;
  if (Perimeter < 0.001) or (Abs(Area) < 0.001) then Exit;
  Distance := 1.5 * Abs(Area) / 2 / Perimeter;
  SetLength(Output, 32768);
  Count := 0;
  Paths := 0;
  Code := FOffset(@Input[0], 4, Distance, @Output[0], 16384, Count, Paths);
  if Code = 2 then Exit;
  RequireCpu((Code = 0) and (Paths = 1) and (Count >= 3) and (Count <= 16384),
    Format('PaddleOCR Clipper failed: code=%d paths=%d points=%d', [Code, Paths, Count]));
  SetLength(Polygon, Count);
  for I := 0 to Count - 1 do
    Polygon[I] := OCVPoint(Output[2 * I], Output[2 * I + 1]);
  Result := MiniBox(Polygon, Expanded) >= 5;
end;

function BoxScore(const Probability: IMat; const Quad: TPaddleQuad): Single;
var
  I, Left, Top, Right, Bottom, Count, Pixels: Integer;
  MinX, MinY, MaxX, MaxY: Single;
  Points: array[0..3] of TOCVPoint;
  Mask, Region: IMat;
  Raw: TMat;
  Average: TOCVScalar;
begin
  MinX := Quad[0].X;
  MaxX := MinX;
  MinY := Quad[0].Y;
  MaxY := MinY;
  for I := 1 to 3 do
  begin
    MinX := Min(MinX, Quad[I].X);
    MaxX := Max(MaxX, Quad[I].X);
    MinY := Min(MinY, Quad[I].Y);
    MaxY := Max(MaxY, Quad[I].Y);
  end;
  Left := Max(0, Floor(MinX));
  Top := Max(0, Floor(MinY));
  Right := Min(Probability.Width - 1, Ceil(MaxX));
  Bottom := Min(Probability.Height - 1, Ceil(MaxY));
  Result := 0;
  if (Right < Left) or (Bottom < Top) then Exit;
  Mask := ZerosMat(Right - Left + 1, Bottom - Top + 1, CV_MAKETYPE(CV_8U, 1));
  Region := RegionView(Probability, Left, Top, Right - Left + 1, Bottom - Top + 1);
  RequireCpu((Mask <> nil) and not Mask.IsEmpty and (Region <> nil) and not Region.IsEmpty,
    'PaddleOCR box scoring allocation failed');
  for I := 0 to 3 do Points[I] := OCVPoint(Trunc(Quad[I].X) - Left, Trunc(Quad[I].Y) - Top);
  Count := 4;
  Raw := Mask.Raw;
  RequireCpu(CV.ImgProc.FillPoly(@Raw, @Points[0], @Count, 1, OCVScalar(1, 0, 0, 0), LINE_8) and
    (Raw = Mask.Raw), 'PaddleOCR box score mask failed');
  Pixels := 0;
  RequireCpu(CV.Core.CountNonZero(Mask.Raw, @Pixels), 'PaddleOCR box score count failed');
  if Pixels = 0 then Exit;
  RequireCpu(CV.Core.Mean(Region.Raw, Mask.Raw, @Average) and not IsNan(Average.V0) and
    not IsInfinite(Average.V0), 'PaddleOCR box score mean failed');
  Result := Average.V0;
end;

function TPaddleDbPostprocess.Detect(const Probabilities: TArray<Single>;
  const ImageWidth, ImageHeight: Integer): TPaddleDetections;
var
  Probability, Binary: IMat;
  Collector: TCollector;
  I, J, Count, Y, Side, OffsetX, OffsetY: Integer;
  Quad, Expanded: TPaddleQuad;
  Score, Scale: Single;
  Entry: TPaddleDetection;
begin
  RequireCpu((Length(Probabilities) = PaddleDetectionSide * PaddleDetectionSide) and
    (ImageWidth > 0) and (ImageHeight > 0) and (ImageWidth <= 4096) and (ImageHeight <= 4096),
    'PaddleOCR detector geometry differs');
  for I := 0 to High(Probabilities) do
    RequireCpu(
      not IsNan(Probabilities[I]) and not IsInfinite(Probabilities[I]) and (Probabilities[I] >= -1e-6) and (Probabilities[I] <= 1.00001),
      'Invalid PaddleOCR detector probability'
    );
  Probability := NewMat(PaddleDetectionSide, PaddleDetectionSide, CV_MAKETYPE(CV_32F, 1));
  RequireCpu((Probability <> nil) and not Probability.IsEmpty, 'PaddleOCR probability map allocation failed');
  for Y := 0 to PaddleDetectionSide - 1 do
    Move(Probabilities[Y * PaddleDetectionSide],
      (PByte(Probability.DataPtr) + NativeInt(Y) * Probability.Step)^,
      PaddleDetectionSide * SizeOf(Single));
  Binary := Probability.ThresholdEx(0.3, 1, THRESH_BINARY).ConvertTo(CV_MAKETYPE(CV_8U, 1), 255, 0);
  RequireCpu((Binary <> nil) and not Binary.IsEmpty, 'PaddleOCR thresholding failed');
  Collector.Seen := 0;
  Collector.Failed := False;
  Collector.Contours := nil;
  RequireCpu(CV.ImgProc.FindContours(Binary.Raw, RETR_LIST, CHAIN_APPROX_SIMPLE, OCVPoint(0, 0),
    {$IFDEF FPC}@{$ENDIF}CollectContour, @Collector) and not Collector.Failed,
    'PaddleOCR contour enumeration failed');
  Side := Max(ImageWidth, ImageHeight);
  Scale := Side / PaddleDetectionSide;
  OffsetX := (Side - ImageWidth) div 2;
  OffsetY := (Side - ImageHeight) div 2;
  Result := nil;
  for I := 0 to High(Collector.Contours) do
  begin
    if MiniBox(Collector.Contours[I], Quad) < 3 then Continue;
    Score := BoxScore(Probability, Quad);
    if Score < 0.6 then Continue;
    if not Unclip(Quad, Expanded) then Continue;
    Count := Length(Result);
    SetLength(Result, Count + 1);
    Result[Count].Score := Score;
    for J := 0 to 3 do
      Result[Count].Quad[J] := OCVPoint2f(
        Max(
          0,
          Min(
            ImageWidth,
            Round(Expanded[J].X * Scale) - OffsetX
          )
        ),
        Max(
          0,
          Min(
            ImageHeight,
            Round(Expanded[J].Y * Scale) - OffsetY
          )
        )
      );
  end;
  { Stable reading order, followed by the upstream ten-pixel line adjustment. }
  for I := 1 to High(Result) do
  begin
    Entry := Result[I];
    J := I - 1;
    while (J >= 0) and ((Result[J].Quad[0].Y > Entry.Quad[0].Y) or
      ((Result[J].Quad[0].Y = Entry.Quad[0].Y) and (Result[J].Quad[0].X > Entry.Quad[0].X))) do
      begin
        Result[J + 1] := Result[J];
        Dec(J);
      end;
    Result[J + 1] := Entry;
  end;
  for I := 1 to High(Result) do
  begin
    J := I;
    while (J > 0) and (Abs(Result[J].Quad[0].Y - Result[J - 1].Quad[0].Y) < 10) and
      (Result[J].Quad[0].X < Result[J - 1].Quad[0].X) do
      begin
        Entry := Result[J - 1];
        Result[J - 1] := Result[J];
        Result[J] := Entry;
        Dec(J);
      end;
  end;
end;

function PaddleCrop(const Image: IMat; const Quad: TPaddleQuad): IMat;
var
  Destination: TPaddleQuad;
  Width, Height: Integer;
  TransformRaw, OutputRaw: TMat;
  Transform: IMat;
begin
  CheckImage(Image);
  Result := nil;
  Width := Trunc(Max(Hypot(Quad[1].X - Quad[0].X, Quad[1].Y - Quad[0].Y),
    Hypot(Quad[2].X - Quad[3].X, Quad[2].Y - Quad[3].Y)));
  Height := Trunc(Max(Hypot(Quad[3].X - Quad[0].X, Quad[3].Y - Quad[0].Y),
    Hypot(Quad[2].X - Quad[1].X, Quad[2].Y - Quad[1].Y)));
  if (Width < 3) or (Height < 3) then Exit;
  RequireCpu((Width <= 4096) and (Height <= 4096), 'PaddleOCR crop exceeds image limits');
  Destination[0] := OCVPoint2f(0, 0);
  Destination[1] := OCVPoint2f(Width, 0);
  Destination[2] := OCVPoint2f(Width, Height);
  Destination[3] := OCVPoint2f(0, Height);
  TransformRaw := nil;
  OutputRaw := nil;
  try
    RequireCpu(CV.ImgProc.GetPerspectiveTransform(@Quad[0], @Destination[0], @TransformRaw) and
      (TransformRaw <> nil), 'PaddleOCR perspective transform failed');
    Transform := WrapMat(TransformRaw, True);
    TransformRaw := nil;
    RequireCpu(
      CV.ImgProc.WarpPerspective(Image.Raw, @OutputRaw, Transform.Raw, OCVSize(Width, Height), INTER_CUBIC, BORDER_REPLICATE, OCVScalar(0, 0, 0, 0)) and (OutputRaw <> nil),
      'PaddleOCR perspective crop failed'
    );
    Result := WrapMat(OutputRaw, True);
    OutputRaw := nil;
    if Height / Width >= 1.5 then Result := Result.Rotate90(ROTATE_90_COUNTERCLOCKWISE);
    RequireCpu((Result <> nil) and not Result.IsEmpty, 'PaddleOCR crop is empty');
  finally
    if TransformRaw <> nil then CV.Core.DeleteMat(@TransformRaw);
    if OutputRaw <> nil then CV.Core.DeleteMat(@OutputRaw);
  end;
end;

function PaddleRead(const Detector, Recognizer: TPaddleOcrModel; const Postprocess: TPaddleDbPostprocess;
  const Image: IMat; const Progress: TPaddleProgress; const ShouldStop: TCpuGraphStop): TPaddleTextLines;
var
  Probabilities: TArray<Single>;
  Shape: TArray<Int64>;
  Detections: TPaddleDetections;
  Crop: IMat;
  I, Count: Integer;
begin
  RequireCpu((Detector <> nil) and (Detector.Kind = 'ocr_detection') and
    (Recognizer <> nil) and (Recognizer.Kind = 'ocr_recognition') and (Postprocess <> nil),
    'PaddleOCR requires detection, recognition and DB postprocessing');
  RequireCpu(SameText(Detector.DeviceName, Recognizer.DeviceName),
    'Choose PaddleOCR detector and recognizer bundles for the same device');
  if Assigned(Progress) then Progress('Detecting text on ' + Detector.DeviceName + '...');
  Probabilities := Detector.Run(PaddleDetectionInput(Image), Shape, ShouldStop);
  Detections := Postprocess.Detect(Probabilities, Image.Width, Image.Height);
  Result := nil;
  for I := 0 to High(Detections) do
  begin
    if Assigned(ShouldStop) and ShouldStop() then
      raise ECpuGraphStopped.Create('PaddleOCR stopped');
    if Assigned(Progress) then Progress(Format('Recognizing line %d of %d on %s...',
      [I + 1, Length(Detections), Recognizer.DeviceName]));
    Crop := PaddleCrop(Image, Detections[I].Quad);
    if Crop = nil then Continue;
    Count := Length(Result);
    SetLength(Result, Count + 1);
    Result[Count].Detection := Detections[I];
    Result[Count].Text := Recognizer.Recognize(Crop, Result[Count].Score, ShouldStop);
  end;
end;

end.
