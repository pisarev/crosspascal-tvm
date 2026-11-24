{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Masks                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Masks;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM.Yolo.Detections;

type
  TYoloInstanceMaskKind = (yimClassicV5, yimModernNative);
  TYoloInstanceMasks = TArray<TBytes>;

{ Input boxes are already projected into original-image coordinates. Returns
  one original-size binary uint8 (0/1) plane per detection, in the same order. }
function RestoreYoloInstanceMasks(const Prototypes: TArray<Single>;
  const PrototypeWidth, PrototypeHeight, Channels: Integer; const Detections: TYoloRawDetections;
  const OriginalWidth, OriginalHeight: Integer; const Kind: TYoloInstanceMaskKind;
  const PrototypeStride: Integer): TYoloInstanceMasks;
function RestoreYoloClassMap(const Source: TBytes;
  const SourceWidth, SourceHeight, InputWidth, InputHeight, OriginalWidth, OriginalHeight: Integer): TBytes;
function RestoreYoloDepth(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, InputWidth, InputHeight, OriginalWidth, OriginalHeight: Integer): TArray<Single>;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.Image.FloatResize, TVM.Yolo.Transform;

function Gcd(const First, Second: Integer): Integer;
var
  A, B, Remainder: Integer;
begin
  A := First;
  B := Second;
  while B <> 0 do
  begin
    Remainder := A mod B;
    A := B;
    B := Remainder;
  end;
  Result := A;
end;

function Sigmoid(const Value: Single): Single;
var
  Exponential: Double;
begin
  if Value > 100 then Exit(1);
  if Value < -100 then Exit(0);
  if Value >= 0 then
    Result := 1 / (1 + Exp(-Value))
  else begin
    Exponential := Exp(Value);
    Result := Exponential / (1 + Exponential);
  end;
end;

function RestoreYoloInstanceMasks(const Prototypes: TArray<Single>;
  const PrototypeWidth, PrototypeHeight, Channels: Integer; const Detections: TYoloRawDetections;
  const OriginalWidth, OriginalHeight: Integer; const Kind: TYoloInstanceMaskKind;
  const PrototypeStride: Integer): TYoloInstanceMasks;
var
  Logits, Cropped, Resized, Upsampled: TArray<Single>;
  Span: TYoloImageTransform;
  Pixels, Mask, C, P, X, Y, Factor, Padding, WorkWidth, WorkHeight: Integer;
  Left, Top, Right, Bottom: Integer;
  Gain, PadX, PadY: Double;
  Coefficient, Threshold: Single;
begin
  if (PrototypeWidth <= 0) or (PrototypeHeight <= 0) or (Channels <= 0) or
    (OriginalWidth <= 0) or (OriginalHeight <= 0) or
    (Int64(PrototypeWidth) * PrototypeHeight > High(Integer) div SizeOf(Single)) or
    (Int64(OriginalWidth) * OriginalHeight > High(Integer) div SizeOf(Single)) or
    not (Kind in [yimClassicV5, yimModernNative]) then
      raise EArgumentException.Create('Invalid YOLO instance mask dimensions');
  Pixels := PrototypeWidth * PrototypeHeight;
  if Int64(Pixels) * Channels <> Length(Prototypes) then
    raise EArgumentException.Create('YOLO prototype byte count differs from its shape');
  for P := 0 to High(Prototypes) do
    if IsNan(Prototypes[P]) or IsInfinite(Prototypes[P]) then
      raise EArgumentException.Create('Nonfinite YOLO prototype value');
  Factor := 1;
  if Kind = yimModernNative then
  begin
    if (PrototypeStride <= 0) or ((PrototypeStride and (PrototypeStride - 1)) <> 0) or
      (Int64(PrototypeStride) * PrototypeWidth > High(Integer)) or
      (Int64(PrototypeStride) * PrototypeHeight > High(Integer)) then
        raise EArgumentException.Create('Invalid YOLO prototype stride');
    Span := TYoloImageTransform.Letterbox(OriginalWidth, OriginalHeight,
      PrototypeWidth * PrototypeStride, PrototypeHeight * PrototypeStride);
    Padding := (Span.Top + Span.Bottom) or (Span.Left + Span.Right);
    Factor := PrototypeStride div Gcd(PrototypeStride, Padding or (Padding div 2));
  end;
  WorkWidth := PrototypeWidth * Factor;
  WorkHeight := PrototypeHeight * Factor;
  if Kind = yimClassicV5 then
  begin
    Gain := Min(PrototypeWidth / OriginalWidth, PrototypeHeight / OriginalHeight);
    PadX := (PrototypeWidth - OriginalWidth * Gain) / 2;
    PadY := (PrototypeHeight - OriginalHeight * Gain) / 2;
    Left := Trunc(PadX);
    Top := Trunc(PadY);
    Right := Trunc(PrototypeWidth - PadX);
    Bottom := Trunc(PrototypeHeight - PadY);
    Threshold := 0.5;
  end
  else begin
    Span := TYoloImageTransform.Letterbox(OriginalWidth, OriginalHeight, WorkWidth, WorkHeight);
    Left := Span.Left;
    Top := Span.Top;
    Right := Left + Span.ResizedWidth;
    Bottom := Top + Span.ResizedHeight;
    Threshold := 0;
  end;
  SetLength(Result, Length(Detections));
  SetLength(Logits, Pixels);
  for Mask := 0 to High(Detections) do
  begin
    if Length(Detections[Mask].Extra) <> Channels then
      raise EArgumentException.Create('YOLO mask coefficient count differs from its prototypes');
    if IsNan(Detections[Mask].Box.Left) or IsInfinite(Detections[Mask].Box.Left) or
      IsNan(Detections[Mask].Box.Right) or IsInfinite(Detections[Mask].Box.Right) or
      IsNan(Detections[Mask].Box.Top) or IsInfinite(Detections[Mask].Box.Top) or
      IsNan(Detections[Mask].Box.Bottom) or IsInfinite(Detections[Mask].Box.Bottom) then
        raise EArgumentException.Create('Nonfinite YOLO mask crop box');
    FillChar(Logits[0], Pixels * SizeOf(Single), 0);
    for C := 0 to Channels - 1 do
    begin
      Coefficient := Detections[Mask].Extra[C];
      if IsNan(Coefficient) or IsInfinite(Coefficient) then
        raise EArgumentException.Create('Nonfinite YOLO mask coefficient');
      for P := 0 to Pixels - 1 do
        Logits[P] := Logits[P] + Coefficient * Prototypes[C * Pixels + P];
    end;
    if Kind = yimClassicV5 then
      for P := 0 to Pixels - 1 do Logits[P] := Sigmoid(Logits[P]);
    if Factor > 1 then
      Upsampled := ResizeFloatBilinear(Logits, PrototypeWidth, PrototypeHeight,
        WorkWidth, WorkHeight, False)
    else
      Upsampled := Logits;
    Cropped := CropFloatPlane(Upsampled, WorkWidth, WorkHeight, Left, Top, Right - Left, Bottom - Top);
    Resized := ResizeFloatBilinear(Cropped, Right - Left, Bottom - Top,
      OriginalWidth, OriginalHeight, False);
    SetLength(Result[Mask], OriginalWidth * OriginalHeight);
    for Y := 0 to OriginalHeight - 1 do
      for X := 0 to OriginalWidth - 1 do
        Result[Mask][Y * OriginalWidth + X] := Ord((Resized[Y * OriginalWidth + X] > Threshold) and
          (X >= Detections[Mask].Box.Left) and (X < Detections[Mask].Box.Right) and
          (Y >= Detections[Mask].Box.Top) and (Y < Detections[Mask].Box.Bottom));
  end;
end;

function RestoreYoloClassMap(const Source: TBytes;
  const SourceWidth, SourceHeight, InputWidth, InputHeight, OriginalWidth, OriginalHeight: Integer): TBytes;
var
  Input, Cropped: TBytes;
  Span: TYoloImageTransform;
begin
  Input := ResizeClassMapNearest(Source, SourceWidth, SourceHeight, InputWidth, InputHeight);
  Span := TYoloImageTransform.Letterbox(OriginalWidth, OriginalHeight, InputWidth, InputHeight);
  Cropped := CropClassMap(Input, InputWidth, InputHeight, Span.Left, Span.Top,
    Span.ResizedWidth, Span.ResizedHeight);
  Result := ResizeClassMapNearest(Cropped, Span.ResizedWidth, Span.ResizedHeight,
    OriginalWidth, OriginalHeight);
end;

function RestoreYoloDepth(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, InputWidth, InputHeight, OriginalWidth, OriginalHeight: Integer): TArray<Single>;
var
  Input: TArray<Single>;
begin
  Input := ResizeFloatBilinear(Source, SourceWidth, SourceHeight, InputWidth, InputHeight, True);
  Result := ResizeFloatBilinear(Input, InputWidth, InputHeight, OriginalWidth, OriginalHeight, True);
end;

end.
