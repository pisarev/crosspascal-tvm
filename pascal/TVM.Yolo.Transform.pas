{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Transform                                                         }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Transform;

{$IFDEF FPC}{$MODE DELPHI}{$ENDIF}

interface

uses
  {$IFDEF FPC}Types{$ELSE}System.Types{$ENDIF};

type
  TYoloCoordinateMode = (ycmRoundedResizeAxes, ycmLegacySymmetricGain, ycmLegacyRoundedGain);

  TYoloImageTransform = record
  private
    FOriginalWidth: Integer;
    FOriginalHeight: Integer;
    FInputWidth: Integer;
    FInputHeight: Integer;
    FResizedWidth: Integer;
    FResizedHeight: Integer;
    FLeft: Integer;
    FTop: Integer;
    FRight: Integer;
    FBottom: Integer;
    FGain: Double;
    FGainX: Double;
    FGainY: Double;
    FOffsetX: Double;
    FOffsetY: Double;
    procedure RequireInitialized;
  public
    class function Letterbox(const OriginalWidth, OriginalHeight, InputWidth, InputHeight: Integer;
      const ScaleUp: Boolean = True;
      const CoordinateMode: TYoloCoordinateMode = ycmRoundedResizeAxes): TYoloImageTransform; static;
    function ImagePoint(const ModelPoint: TPointF; const Clip: Boolean = True): TPointF;
    function ImageBox(const ModelBox: TRectF; const Clip: Boolean = True): TRectF;
    function ModelPoint(const OriginalPoint: TPointF): TPointF;
    property OriginalWidth: Integer read FOriginalWidth;
    property OriginalHeight: Integer read FOriginalHeight;
    property InputWidth: Integer read FInputWidth;
    property InputHeight: Integer read FInputHeight;
    property ResizedWidth: Integer read FResizedWidth;
    property ResizedHeight: Integer read FResizedHeight;
    property Left: Integer read FLeft;
    property Top: Integer read FTop;
    property Right: Integer read FRight;
    property Bottom: Integer read FBottom;
    property Gain: Double read FGain;
    property GainX: Double read FGainX;
    property GainY: Double read FGainY;
    property OffsetX: Double read FOffsetX;
    property OffsetY: Double read FOffsetY;
  end;

implementation

uses
  {$IFDEF FPC}Math, SysUtils{$ELSE}System.Math, System.SysUtils{$ENDIF};

function RoundEvenPositive(const Value: Double): Integer;
var
  Integral: Int64;
  Fraction: Double;
begin
  Integral := Trunc(Value);
  Fraction := Value - Integral;
  if (Fraction > 0.5) or ((Fraction = 0.5) and Odd(Integral)) then Inc(Integral);
  if Integral > High(Integer) then
    raise EArgumentException.Create('YOLO resized image dimension exceeds Integer');
  Result := Integer(Integral);
end;

function BoundedCoordinate(const Value: Double; const Limit: Integer; const Clip: Boolean): Single;
var
  Upper: Double;
begin
  if IsNan(Value) or IsInfinite(Value) then
    raise EArgumentException.Create('YOLO coordinate must be finite');
  Upper := Limit;
  if Clip then
    Result := Max(0.0, Min(Upper, Value))
  else begin
    if Abs(Value) > MaxSingle then
      raise EArgumentException.Create('YOLO coordinate exceeds Single');
    Result := Value;
  end;
end;

class function TYoloImageTransform.Letterbox(const OriginalWidth, OriginalHeight, InputWidth, InputHeight: Integer;
  const ScaleUp: Boolean; const CoordinateMode: TYoloCoordinateMode): TYoloImageTransform;
var
  WidthRatio, HeightRatio: Double;
begin
  if (OriginalWidth <= 0) or (OriginalHeight <= 0) or (InputWidth <= 0) or (InputHeight <= 0) then
    raise EArgumentException.Create('YOLO image dimensions must be positive');
  if not (CoordinateMode in [ycmRoundedResizeAxes, ycmLegacySymmetricGain, ycmLegacyRoundedGain]) then
    raise EArgumentException.Create('Invalid YOLO coordinate mapping');
  Result.FOriginalWidth := OriginalWidth;
  Result.FOriginalHeight := OriginalHeight;
  Result.FInputWidth := InputWidth;
  Result.FInputHeight := InputHeight;
  WidthRatio := InputWidth;
  HeightRatio := InputHeight;
  Result.FGain := Min(WidthRatio / OriginalWidth, HeightRatio / OriginalHeight);
  if not ScaleUp then Result.FGain := Min(1.0, Result.FGain);
  Result.FResizedWidth := RoundEvenPositive(OriginalWidth * Result.FGain);
  Result.FResizedHeight := RoundEvenPositive(OriginalHeight * Result.FGain);
  if (Result.FResizedWidth <= 0) or (Result.FResizedHeight <= 0) then
    raise EArgumentException.Create('YOLO letterbox would produce an empty resized image');
  Result.FLeft := (InputWidth - Result.FResizedWidth) div 2;
  Result.FTop := (InputHeight - Result.FResizedHeight) div 2;
  Result.FRight := InputWidth - Result.FResizedWidth - Result.FLeft;
  Result.FBottom := InputHeight - Result.FResizedHeight - Result.FTop;
  Result.FGainX := Result.FGain;
  Result.FGainY := Result.FGain;
  Result.FOffsetX := (InputWidth - OriginalWidth * Result.FGain) / 2;
  Result.FOffsetY := (InputHeight - OriginalHeight * Result.FGain) / 2;
  case CoordinateMode of
    ycmRoundedResizeAxes:
      begin
        Result.FGainX := Result.FResizedWidth / OriginalWidth;
        Result.FGainY := Result.FResizedHeight / OriginalHeight;
        Result.FOffsetX := Result.FLeft;
        Result.FOffsetY := Result.FTop;
      end;
    ycmLegacyRoundedGain:
      begin
        Result.FOffsetX := RoundEvenPositive(Max(0.0, Result.FOffsetX - 0.1));
        Result.FOffsetY := RoundEvenPositive(Max(0.0, Result.FOffsetY - 0.1));
      end;
  end;
end;

procedure TYoloImageTransform.RequireInitialized;
begin
  if (FGain <= 0) or (FGainX <= 0) or (FGainY <= 0) or
    (FOriginalWidth <= 0) or (FOriginalHeight <= 0) then
      raise EArgumentException.Create('YOLO image transform has not been initialized');
end;

function TYoloImageTransform.ImagePoint(const ModelPoint: TPointF; const Clip: Boolean): TPointF;
begin
  RequireInitialized;
  Result.X := BoundedCoordinate((ModelPoint.X - FOffsetX) / FGainX, FOriginalWidth, Clip);
  Result.Y := BoundedCoordinate((ModelPoint.Y - FOffsetY) / FGainY, FOriginalHeight, Clip);
end;

function TYoloImageTransform.ImageBox(const ModelBox: TRectF; const Clip: Boolean): TRectF;
begin
  RequireInitialized;
  if (ModelBox.Right < ModelBox.Left) or (ModelBox.Bottom < ModelBox.Top) then
    raise EArgumentException.Create('YOLO box edges are reversed');
  Result.Left := BoundedCoordinate((ModelBox.Left - FOffsetX) / FGainX, FOriginalWidth, Clip);
  Result.Top := BoundedCoordinate((ModelBox.Top - FOffsetY) / FGainY, FOriginalHeight, Clip);
  Result.Right := BoundedCoordinate((ModelBox.Right - FOffsetX) / FGainX, FOriginalWidth, Clip);
  Result.Bottom := BoundedCoordinate((ModelBox.Bottom - FOffsetY) / FGainY, FOriginalHeight, Clip);
end;

function TYoloImageTransform.ModelPoint(const OriginalPoint: TPointF): TPointF;
begin
  RequireInitialized;
  Result.X := BoundedCoordinate(OriginalPoint.X * FGainX + FOffsetX, FInputWidth, False);
  Result.Y := BoundedCoordinate(OriginalPoint.Y * FGainY + FOffsetY, FInputHeight, False);
end;

end.
