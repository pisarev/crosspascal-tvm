{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.PixelPreprocess                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.PixelPreprocess;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM.Image.Resample, TVM.Yolo.Transform;

type
  TYoloPixelMode = (ypmLetterbox, ypmScaleFill, ypmClassicClassification);

  { The native image adapter performs this crop and INTER_LINEAR resize.
    Packing below adds the recorded border and converts uint8 RGB/BGR to
    Float32 RGB NCHW. No interpolation implementation is substituted here. }
  TYoloPixelPlan = record
  private
    FMode: TYoloPixelMode;
    FOriginalWidth: Integer;
    FOriginalHeight: Integer;
    FWidth: Integer;
    FHeight: Integer;
    FCropX: Integer;
    FCropY: Integer;
    FCropWidth: Integer;
    FCropHeight: Integer;
    FResizedWidth: Integer;
    FResizedHeight: Integer;
    FLeft: Integer;
    FTop: Integer;
    FPadding: Byte;
    procedure RequireInitialized;
  public
    class function Create(const Mode: TYoloPixelMode;
      const OriginalWidth, OriginalHeight, Width, Height: Integer; const ScaleUp: Boolean = True;
      const Padding: Byte = 114): TYoloPixelPlan; static;
    property Mode: TYoloPixelMode read FMode;
    property OriginalWidth: Integer read FOriginalWidth;
    property OriginalHeight: Integer read FOriginalHeight;
    property Width: Integer read FWidth;
    property Height: Integer read FHeight;
    property CropX: Integer read FCropX;
    property CropY: Integer read FCropY;
    property CropWidth: Integer read FCropWidth;
    property CropHeight: Integer read FCropHeight;
    property ResizedWidth: Integer read FResizedWidth;
    property ResizedHeight: Integer read FResizedHeight;
    property Left: Integer read FLeft;
    property Top: Integer read FTop;
    property Padding: Byte read FPadding;
  end;

function PackYoloResizedInput(const Source: Pointer; const SourceBytes, SourceStride: NativeInt;
  const Plan: TYoloPixelPlan; const Order: TRGBByteOrder;
  const Mean, Std: array of Single): TArray<Single>;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF};

class function TYoloPixelPlan.Create(const Mode: TYoloPixelMode;
  const OriginalWidth, OriginalHeight, Width, Height: Integer; const ScaleUp: Boolean;
  const Padding: Byte): TYoloPixelPlan;
var
  Geometry: TYoloImageTransform;
  Side: Integer;
begin
  if not (Mode in [ypmLetterbox, ypmScaleFill, ypmClassicClassification]) or
    (OriginalWidth <= 0) or (OriginalHeight <= 0) or (Width <= 0) or (Height <= 0) or
    (Int64(Width) * Height > High(Integer) div (3 * SizeOf(Single))) then
      raise EArgumentException.Create('Invalid YOLO pixel preparation dimensions or mode');
  Result := Default(TYoloPixelPlan);
  Result.FMode := Mode;
  Result.FOriginalWidth := OriginalWidth;
  Result.FOriginalHeight := OriginalHeight;
  Result.FWidth := Width;
  Result.FHeight := Height;
  Result.FCropWidth := OriginalWidth;
  Result.FCropHeight := OriginalHeight;
  Result.FResizedWidth := Width;
  Result.FResizedHeight := Height;
  Result.FPadding := Padding;
  case Mode of
    ypmLetterbox:
      begin
        Geometry := TYoloImageTransform.Letterbox(OriginalWidth, OriginalHeight, Width, Height, ScaleUp);
        Result.FResizedWidth := Geometry.ResizedWidth;
        Result.FResizedHeight := Geometry.ResizedHeight;
        Result.FLeft := Geometry.Left;
        Result.FTop := Geometry.Top;
      end;
    ypmClassicClassification:
      begin
        Side := Min(OriginalWidth, OriginalHeight);
        Result.FCropX := (OriginalWidth - Side) div 2;
        Result.FCropY := (OriginalHeight - Side) div 2;
        Result.FCropWidth := Side;
        Result.FCropHeight := Side;
      end;
  end;
end;

procedure TYoloPixelPlan.RequireInitialized;
begin
  if (FWidth <= 0) or (FHeight <= 0) or (FResizedWidth <= 0) or (FResizedHeight <= 0) or
    (FOriginalWidth <= 0) or (FOriginalHeight <= 0) or (FCropWidth <= 0) or (FCropHeight <= 0) then
      raise EArgumentException.Create('YOLO pixel preparation plan has not been initialized');
end;

function PackYoloResizedInput(const Source: Pointer; const SourceBytes, SourceStride: NativeInt;
  const Plan: TYoloPixelPlan; const Order: TRGBByteOrder;
  const Mean, Std: array of Single): TArray<Single>;
var
  Channel, SourceChannel, Plane, X, Y, Offset: Integer;
  Required: Int64;
  Pixel, Border: Single;
  Row: PByte;
begin
  Plan.RequireInitialized;
  Required := Int64(Plan.ResizedWidth) * 3;
  if (Source = nil) or (SourceBytes <= 0) or (SourceStride < Required) or
    not (Order in [rboRGB, rboBGR]) or
    (Length(Mean) <> 3) or (Length(Std) <> 3) then
      raise EArgumentException.Create('Invalid YOLO resized pixel buffer or normalization');
  if SourceStride > High(NativeInt) div Plan.ResizedHeight then
    raise EArgumentException.Create('YOLO resized pixel stride exceeds address range');
  Required := Int64(Plan.ResizedHeight - 1) * SourceStride + Required;
  if Required > SourceBytes then
    raise EArgumentException.Create('YOLO resized pixel buffer is truncated');
  for Channel := 0 to 2 do
    if IsNan(Mean[Channel]) or IsInfinite(Mean[Channel]) or
      IsNan(Std[Channel]) or IsInfinite(Std[Channel]) or (Std[Channel] <= 0) then
        raise EArgumentException.Create('Invalid YOLO pixel normalization');
  Plane := Plan.Width * Plan.Height;
  SetLength(Result, Plane * 3);
  for Channel := 0 to 2 do
  begin
    SourceChannel := Channel;
    if Order = rboBGR then SourceChannel := 2 - Channel;
    Border := Plan.Padding;
    Border := Border / 255;
    Border := Border - Mean[Channel];
    Border := Border / Std[Channel];
    for Offset := 0 to Plane - 1 do Result[Channel * Plane + Offset] := Border;
    for Y := 0 to Plan.ResizedHeight - 1 do
    begin
      Row := PByte(NativeUInt(Source) + NativeUInt(Y) * NativeUInt(SourceStride));
      for X := 0 to Plan.ResizedWidth - 1 do
      begin
        Pixel := PByte(NativeUInt(Row) + NativeUInt(X) * 3 + NativeUInt(SourceChannel))^;
        Pixel := Pixel / 255;
        Pixel := Pixel - Mean[Channel];
        Result[Channel * Plane + (Y + Plan.Top) * Plan.Width + X + Plan.Left] := Pixel / Std[Channel];
      end;
    end;
  end;
end;

end.
