{ ************************************************************************** }
{                                                                            }
{ TVM.RtDetr                                                                 }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.RtDetr;

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
  TRtDetrDecoded = record
  public
    Place: TRect;
    Score: Single;
    &Class: Integer;
  end;

  TRtDetrDecodedArray = TArray<TRtDetrDecoded>;

  TRtDetr = class
  public
    class function DecodeDetections(const LogitsTensor, BoxesTensor: PDLTensor;
      const Threshold: Single; const OriginalImageSize: TSize; out Detections: TRtDetrDecodedArray;
      const Error: PString = nil): Boolean; static;
    class function DecodeFusedDetections(const FusedTensor: PDLTensor; const Threshold: Single;
      const OriginalImageSize: TSize; out Detections: TRtDetrDecodedArray;
      const Error: PString = nil): Boolean; static;
    class function Detect(var Descriptor: TModelDescriptor; const Image: IMat;
      const OutputDeviceType, OutputDeviceID: Integer; const Threshold: Single;
      out Scores: TArray<Single>; out Labels: TArray<Integer>; out Places: TArray<TRect>;
      const InputTensorName: string = 'pixel_values'; const Error: PString = nil): Boolean; static;
  end;

implementation

const
  BoxFractionSlack = 0.05;

function ReadSingleAt(const Data: Pointer; const Index: Integer; const Float16: Boolean): Single;
var
  Half: TFloat16;
begin
  if Float16 then
  begin
    Half.Value := PWord(NativeUInt(Data) + NativeUInt(Index) * 2)^;
    Result := Half.ToSingle;
  end
  else
    Result := PSingle(NativeUInt(Data) + NativeUInt(Index) * 4)^;
end;

function Sigmoid(const X: Single): Single;
begin
  if X >= 0 then
    Result := 1 / (1 + Exp(-X))
  else
    Result := Exp(X) / (1 + Exp(X));
end;

class function TRtDetr.DecodeDetections(const LogitsTensor, BoxesTensor: PDLTensor;
  const Threshold: Single; const OriginalImageSize: TSize; out Detections: TRtDetrDecodedArray;
  const Error: PString): Boolean;
var
  ShapeLogits, ShapeBoxes: TArray<Int64>;
  Queries, Classes, Q, C, Taken, I, J: Integer;
  Saved: TRtDetrDecoded;
  LogitsFloat16, BoxesFloat16: Boolean;
  Best, Value, Score, Extreme: Single;
  BestClass: Integer;
  CenterX, CenterY, BoxWidth, BoxHeight: Single;
  Left, Top, Right, Bottom: Single;
  LogitsBytes, BoxesBytes: UInt64;
begin
  Result := False;
  Detections := nil;
  if (LogitsTensor = nil) or (BoxesTensor = nil) or
    (LogitsTensor.data = nil) or (BoxesTensor.data = nil) or
    (LogitsTensor.shape = nil) or (BoxesTensor.shape = nil) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.DecodeDetections: missing logits or boxes tensor, data or shape';
      Exit;
    end;
  if (LogitsTensor.ndim <> 3) or (BoxesTensor.ndim <> 3) then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeDetections: expected two 3-D tensors, got %d-D and %d-D',
        [LogitsTensor.ndim, BoxesTensor.ndim]);
    Exit;
  end;
  if (LogitsTensor.device.device_type <> kDLCPU) or (BoxesTensor.device.device_type <> kDLCPU) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeDetections: logits and boxes tensors must reside on CPU';
    Exit;
  end;
  if not (TTVMRuntime.CheckTensor(LogitsTensor, [kDLFloat], 16) or
    TTVMRuntime.CheckTensor(LogitsTensor, [kDLFloat], 32)) or
    not (TTVMRuntime.CheckTensor(BoxesTensor, [kDLFloat], 16) or
    TTVMRuntime.CheckTensor(BoxesTensor, [kDLFloat], 32)) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.DecodeDetections: logits and boxes must be Float16 or Float32 with one lane';
      Exit;
    end;
  ShapeLogits := TTVMRuntime.GetTensorShape(LogitsTensor);
  ShapeBoxes := TTVMRuntime.GetTensorShape(BoxesTensor);
  if (ShapeLogits[0] <> 1) or (ShapeBoxes[0] <> 1) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeDetections: logits and boxes must have actual batch size 1';
    Exit;
  end;
  if ShapeBoxes[2] <> 4 then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeDetections: boxes tensor last dimension must be 4, got %d',
        [ShapeBoxes[2]]);
    Exit;
  end;
  if ShapeLogits[1] <> ShapeBoxes[1] then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeDetections: query count differs, logits %d against boxes %d',
        [ShapeLogits[1], ShapeBoxes[1]]);
    Exit;
  end;
  if (ShapeLogits[1] < 1) or (ShapeLogits[2] < 1) or
    (ShapeLogits[1] > High(Integer) div 4) or
    (ShapeLogits[2] > High(Integer)) or
    (ShapeLogits[1] > High(Integer) div ShapeLogits[2]) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.DecodeDetections: query and class dimensions must be positive and fit decoder indexing';
      Exit;
    end;
  Queries := Integer(ShapeLogits[1]);
  Classes := Integer(ShapeLogits[2]);
  if (Assigned(LogitsTensor.strides) and
    ((LogitsTensor.strides[0] <> Int64(Queries) * Classes) or
    (LogitsTensor.strides[1] <> Classes) or (LogitsTensor.strides[2] <> 1))) or
    (Assigned(BoxesTensor.strides) and
    ((BoxesTensor.strides[0] <> Int64(Queries) * 4) or
    (BoxesTensor.strides[1] <> 4) or (BoxesTensor.strides[2] <> 1))) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.DecodeDetections: logits and boxes must have contiguous row-major strides';
      Exit;
    end;
  if (LogitsTensor.byte_offset <> 0) or (BoxesTensor.byte_offset <> 0) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeDetections: nonzero tensor byte offsets are not supported';
    Exit;
  end;
  LogitsBytes := UInt64(Queries) * UInt64(Classes) * (LogitsTensor.dtype.bits div 8);
  BoxesBytes := UInt64(Queries) * 4 * (BoxesTensor.dtype.bits div 8);
  if (LogitsBytes - 1 > UInt64(High(NativeUInt) - NativeUInt(LogitsTensor.data))) or
    (BoxesBytes - 1 > UInt64(High(NativeUInt) - NativeUInt(BoxesTensor.data))) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.DecodeDetections: tensor byte span overflows address space';
      Exit;
    end;
  LogitsFloat16 := (LogitsTensor.dtype.code = LongWord(kDLFloat)) and
    (LogitsTensor.dtype.bits = 16);
  BoxesFloat16 := (BoxesTensor.dtype.code = LongWord(kDLFloat)) and (BoxesTensor.dtype.bits = 16);
  Extreme := 0;
  for Q := 0 to Queries * 4 - 1 do
  begin
    Value := Abs(ReadSingleAt(BoxesTensor.data, Q, BoxesFloat16));
    if Value > Extreme then
      Extreme := Value;
  end;
  if Extreme > 1 + BoxFractionSlack then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeDetections: boxes must be fractions of the image, ' +
        'the largest absolute value is %.3f', [Extreme]);
    Exit;
  end;
  SetLength(Detections, Queries);
  Taken := 0;
  for Q := 0 to Queries - 1 do
  begin
    Best := -1.0e30;
    BestClass := -1;
    for C := 0 to Classes - 1 do
    begin
      Value := ReadSingleAt(LogitsTensor.data, Q * Classes + C, LogitsFloat16);
      if Value > Best then
      begin
        Best := Value;
        BestClass := C;
      end;
    end;
    Score := Sigmoid(Best);
    if Score < Threshold then
      Continue;
    CenterX := ReadSingleAt(BoxesTensor.data, Q * 4 + 0, BoxesFloat16);
    CenterY := ReadSingleAt(BoxesTensor.data, Q * 4 + 1, BoxesFloat16);
    BoxWidth := ReadSingleAt(BoxesTensor.data, Q * 4 + 2, BoxesFloat16);
    BoxHeight := ReadSingleAt(BoxesTensor.data, Q * 4 + 3, BoxesFloat16);
    Left := (CenterX - BoxWidth / 2) * OriginalImageSize.cx;
    Top := (CenterY - BoxHeight / 2) * OriginalImageSize.cy;
    Right := (CenterX + BoxWidth / 2) * OriginalImageSize.cx;
    Bottom := (CenterY + BoxHeight / 2) * OriginalImageSize.cy;
    Detections[Taken].Place := TRect.Create(
      Max(
        0,
        Round(Left)
      ),
      Max(
        0,
        Round(Top)
      ),
      Min(
        OriginalImageSize.cx,
        Round(Right)
      ),
      Min(
        OriginalImageSize.cy,
        Round(Bottom)
      )
    );
    Detections[Taken].Score := Score;
    Detections[Taken].&Class := BestClass;
    Inc(Taken);
  end;
  SetLength(Detections, Taken);
  for I := 1 to Taken - 1 do
  begin
    Saved := Detections[I];
    J := I - 1;
    while (J >= 0) and (Detections[J].Score < Saved.Score) do
    begin
      Detections[J + 1] := Detections[J];
      Dec(J);
    end;
    Detections[J + 1] := Saved;
  end;
  Result := True;
end;

class function TRtDetr.DecodeFusedDetections(const FusedTensor: PDLTensor; const Threshold: Single;
  const OriginalImageSize: TSize; out Detections: TRtDetrDecodedArray;
  const Error: PString): Boolean;
var
  Shape: TArray<Int64>;
  DataPtr: Pointer;
  DataBytes: UInt64;
  Queries, Q, Taken, I, J, ClassID: Integer;
  Float16: Boolean;
  CenterX, CenterY, BoxWidth, BoxHeight, Score, ClassValue: Single;
  CenterX64, CenterY64, BoxWidth64, BoxHeight64: Double;
  X1, Y1, X2, Y2: Double;
  LeftPx, TopPx, RightPx, BottomPx: Integer;
  Saved: TRtDetrDecoded;
begin
  Result := False;
  Detections := nil;
  if (FusedTensor = nil) or (FusedTensor.data = nil) or (FusedTensor.shape = nil) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeFusedDetections: missing fused tensor, data or shape';
    Exit;
  end;
  if FusedTensor.device.device_type <> kDLCPU then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeFusedDetections: tensor must reside on CPU';
    Exit;
  end;
  if IsNan(Threshold) or IsInfinite(Threshold) or (Threshold < 0) or (Threshold > 1) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeFusedDetections: threshold must be finite within [0,1]';
    Exit;
  end;
  if (OriginalImageSize.cx <= 0) or (OriginalImageSize.cy <= 0) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeFusedDetections: image dimensions must be positive';
    Exit;
  end;
  if FusedTensor.ndim <> 3 then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeFusedDetections: expected [1,N,6], got %d dimensions',
        [FusedTensor.ndim]);
    Exit;
  end;
  Shape := TTVMRuntime.GetTensorShape(FusedTensor);
  if Length(Shape) <> 3 then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeFusedDetections: expected [1,N,6], got %d dimensions',
        [Length(Shape)]);
    Exit;
  end;
  if (Shape[0] <> 1) or (Shape[1] < 1) or (Shape[2] <> 6) or (Shape[1] > High(Integer) div 6) then
  begin
    if Assigned(Error) then
      Error^ := Format('TRtDetr.DecodeFusedDetections: expected [1,N,6] with N in 1..%d, got [%d,%d,%d]',
        [High(Integer) div 6, Shape[0], Shape[1], Shape[2]]);
    Exit;
  end;
  if (FusedTensor.dtype.code <> Ord(kDLFloat)) or
    (not (FusedTensor.dtype.bits in [16, 32])) or (FusedTensor.dtype.lanes <> 1) then
    begin
      if Assigned(Error) then
        Error^ := Format('TRtDetr.DecodeFusedDetections: expected float16/float32 with one lane, got code=%d bits=%d lanes=%d',
          [FusedTensor.dtype.code, FusedTensor.dtype.bits, FusedTensor.dtype.lanes]);
      Exit;
    end;
  if Assigned(FusedTensor.strides) and
    ((FusedTensor.strides[0] <> Shape[1] * 6) or
    (FusedTensor.strides[1] <> 6) or (FusedTensor.strides[2] <> 1)) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.DecodeFusedDetections: fused tensor must have contiguous row-major strides';
      Exit;
    end;
  if FusedTensor.byte_offset > UInt64(High(NativeUInt) - NativeUInt(FusedTensor.data)) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeFusedDetections: tensor byte offset overflows address space';
    Exit;
  end;
  DataPtr := Pointer(NativeUInt(FusedTensor.data) + NativeUInt(FusedTensor.byte_offset));
  DataBytes := UInt64(Shape[1]) * 6 * (FusedTensor.dtype.bits div 8);
  if DataBytes - 1 > UInt64(High(NativeUInt) - NativeUInt(DataPtr)) then
  begin
    if Assigned(Error) then
      Error^ := 'TRtDetr.DecodeFusedDetections: tensor byte span overflows address space';
    Exit;
  end;
  Float16 := FusedTensor.dtype.bits = 16;
  Queries := Integer(Shape[1]);
  SetLength(Detections, Queries);
  Taken := 0;
  for Q := 0 to Queries - 1 do
  begin
    Score := ReadSingleAt(DataPtr, Q * 6 + 4, Float16);
    if IsNan(Score) or IsInfinite(Score) or (Score < 0) or (Score > 1) or (Score < Threshold) then
      Continue;
    ClassValue := ReadSingleAt(DataPtr, Q * 6 + 5, Float16);
    if IsNan(ClassValue) or IsInfinite(ClassValue) or (ClassValue < 0) or (ClassValue >= 80) then
      Continue;
    if ClassValue <> Trunc(ClassValue) then
      Continue;
    ClassID := Trunc(ClassValue);
    CenterX := ReadSingleAt(DataPtr, Q * 6 + 0, Float16);
    CenterY := ReadSingleAt(DataPtr, Q * 6 + 1, Float16);
    BoxWidth := ReadSingleAt(DataPtr, Q * 6 + 2, Float16);
    BoxHeight := ReadSingleAt(DataPtr, Q * 6 + 3, Float16);
    if IsNan(CenterX) or IsInfinite(CenterX) or
      IsNan(CenterY) or IsInfinite(CenterY) or
      IsNan(BoxWidth) or IsInfinite(BoxWidth) or
      IsNan(BoxHeight) or IsInfinite(BoxHeight) then
        Continue;
    if (CenterX < -BoxFractionSlack) or (CenterX > 1 + BoxFractionSlack) or
      (CenterY < -BoxFractionSlack) or (CenterY > 1 + BoxFractionSlack) or
      (BoxWidth <= 0) or (BoxWidth > 1 + BoxFractionSlack) or
      (BoxHeight <= 0) or (BoxHeight > 1 + BoxFractionSlack) then
        Continue;
    CenterX64 := CenterX;
    CenterY64 := CenterY;
    BoxWidth64 := BoxWidth;
    BoxHeight64 := BoxHeight;
    X1 := CenterX64 - BoxWidth64 / 2;
    Y1 := CenterY64 - BoxHeight64 / 2;
    X2 := CenterX64 + BoxWidth64 / 2;
    Y2 := CenterY64 + BoxHeight64 / 2;
    if (X2 <= 0) or (Y2 <= 0) or (X1 >= 1) or (Y1 >= 1) then
      Continue;
    if X1 < 0 then X1 := 0 else if X1 > 1 then X1 := 1;
    if Y1 < 0 then Y1 := 0 else if Y1 > 1 then Y1 := 1;
    if X2 < 0 then X2 := 0 else if X2 > 1 then X2 := 1;
    if Y2 < 0 then Y2 := 0 else if Y2 > 1 then Y2 := 1;
    LeftPx := Round(Double(X1) * OriginalImageSize.cx);
    TopPx := Round(Double(Y1) * OriginalImageSize.cy);
    RightPx := Round(Double(X2) * OriginalImageSize.cx);
    BottomPx := Round(Double(Y2) * OriginalImageSize.cy);
    if (RightPx <= LeftPx) or (BottomPx <= TopPx) then
      Continue;
    Detections[Taken].Place := TRect.Create(LeftPx, TopPx, RightPx, BottomPx);
    Detections[Taken].Score := Score;
    Detections[Taken].&Class := ClassID;
    Inc(Taken);
  end;
  SetLength(Detections, Taken);
  for I := 1 to Taken - 1 do
  begin
    Saved := Detections[I];
    J := I - 1;
    while (J >= 0) and (Detections[J].Score < Saved.Score) do
    begin
      Detections[J + 1] := Detections[J];
      Dec(J);
    end;
    Detections[J + 1] := Saved;
  end;
  Result := True;
end;

class function TRtDetr.Detect(var Descriptor: TModelDescriptor; const Image: IMat;
  const OutputDeviceType, OutputDeviceID: Integer; const Threshold: Single;
  out Scores: TArray<Single>; out Labels: TArray<Integer>; out Places: TArray<TRect>;
  const InputTensorName: string; const Error: PString): Boolean;
var
  InputTensor, ImageTensor: PDLTensor;
  CpuLogits, CpuBoxes, OwnedLogits, OwnedBoxes: PDLTensor;
  CpuFused, OwnedFused: PDLTensor;
  OutputTensors: TArray<PDLTensor>;
  ShapeFirst: TArray<Int64>;
  Decoded: TRtDetrDecodedArray;
  OriginSize: TSize;
  TensorIndex: Int64;
  LogitsIndex, BoxesIndex, I: Integer;
  Scale: Single;
  Img: IMat;
  S: string;
begin
  Result := False;
  Scores := nil;
  Labels := nil;
  Places := nil;
  ImageTensor := nil;
  OwnedLogits := nil;
  OwnedBoxes := nil;
  OwnedFused := nil;
  try
    if not TTVMRuntime.GetInputTensorIndex(Descriptor, InputTensorName, TensorIndex, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TRtDetr.Detect: cannot resolve input tensor. Details: %s', [S]);
      Exit;
    end;
    if not TTVMRuntime.GetInputTensor(Descriptor, TensorIndex, InputTensor, @S) then
    begin
      if Assigned(Error) then
        Error^ := Format('TRtDetr.Detect: cannot obtain input tensor. Details: %s', [S]);
      Exit;
    end;
    if not Assigned(InputTensor) or (InputTensor.ndim <> 4) then
    begin
      if Assigned(Error) then
        Error^ := 'TRtDetr.Detect: input tensor must be four-dimensional NCHW';
      Exit;
    end;
    case InputTensor.dtype.code of
      Ord(kDLFloat): Scale := 1 / 255;
      Ord(kDLUInt), Ord(kDLInt): Scale := 1;
    else begin
        if Assigned(Error) then
          Error^ := Format('TRtDetr.Detect: unsupported input dtype code %d',
            [InputTensor.dtype.code]);
        Exit;
      end;
    end;
    OriginSize.cx := Image.Width;
    OriginSize.cy := Image.Height;
    Img := nil;
    try
      if not TTVMRuntime.ConvertImageToTensorFormat(Image, Img, InputTensor, NCHWTensor,
        Scale, INTER_CUBIC, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TRtDetr.Detect: image preprocessing failed. Details: %s', [S]);
          Exit;
        end;
      Img := ConvertBgrToRgb(Img);
      if not TTVMRuntime.ConvertImageToDLTensor(Img, Descriptor.Device, NCHWTensor, ImageTensor,
        TDLDataTypeCode(InputTensor.dtype.code), InputTensor.dtype.bits,
        InputTensor.dtype.lanes, 1, @S) then
        begin
          if Assigned(Error) then
            Error^ := Format('TRtDetr.Detect: cannot build input DLTensor. Details: %s', [S]);
          Exit;
        end;
    finally
      Img := nil;
    end;
    if not TTVMRuntime.RunInference(Descriptor, InputTensorName, ImageTensor, OutputTensors, Error) then
      Exit;
    case Length(OutputTensors) of
      1:
        begin
          CpuFused := TTVMRuntime.GetOrCreateCpuTensor(OutputTensors[0], OwnedFused,
            'FusedDetections', 'TRtDetr.Detect', Error);
          if not Assigned(CpuFused) then
            Exit;
          if not DecodeFusedDetections(CpuFused, Threshold, OriginSize, Decoded, Error) then
            Exit;
        end;
      2:
        begin
          ShapeFirst := TTVMRuntime.GetTensorShape(OutputTensors[0]);
          if (Length(ShapeFirst) = 3) and (ShapeFirst[2] = 4) then
          begin
            BoxesIndex := 0;
            LogitsIndex := 1;
          end
          else begin
            LogitsIndex := 0;
            BoxesIndex := 1;
          end;
          CpuLogits := TTVMRuntime.GetOrCreateCpuTensor(OutputTensors[LogitsIndex], OwnedLogits,
            'Logits', 'TRtDetr.Detect', Error);
          if not Assigned(CpuLogits) then
            Exit;
          CpuBoxes := TTVMRuntime.GetOrCreateCpuTensor(OutputTensors[BoxesIndex], OwnedBoxes,
            'Boxes', 'TRtDetr.Detect', Error);
          if not Assigned(CpuBoxes) then
            Exit;
          if not DecodeDetections(CpuLogits, CpuBoxes, Threshold, OriginSize, Decoded, Error) then
            Exit;
        end;
    else begin
        if Assigned(Error) then
          Error^ := Format('TRtDetr.Detect: expected one fused or two legacy output tensors, got %d',
            [Length(OutputTensors)]);
        Exit;
      end;
    end;
    SetLength(Scores, Length(Decoded));
    SetLength(Labels, Length(Decoded));
    SetLength(Places, Length(Decoded));
    for I := 0 to High(Decoded) do
    begin
      Scores[I] := Decoded[I].Score;
      Labels[I] := Decoded[I].&Class;
      Places[I] := Decoded[I].Place;
    end;
  finally
    if Assigned(ImageTensor) then
      FreeMem(ImageTensor);
    if Assigned(OwnedLogits) then
      FreeMem(OwnedLogits);
    if Assigned(OwnedBoxes) then
      FreeMem(OwnedBoxes);
    if Assigned(OwnedFused) then
      FreeMem(OwnedFused);
  end;
  Result := True;
end;

end.
