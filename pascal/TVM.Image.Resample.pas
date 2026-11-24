{ ************************************************************************** }
{                                                                            }
{ TVM.Image.Resample                                                         }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Image.Resample;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF};

type
  TRGBByteOrder = (rboRGB, rboBGR);

{ Top-down uint8 RGB/BGR with an explicit accessible extent and row stride.
  Matches Pillow's uint8 BILINEAR/BICUBIC antialiasing, not OpenCV INTER_LINEAR. }
function ResizePillowRGB(const Source: Pointer; const SourceBytes: NativeInt;
  const SourceWidth, SourceHeight: Integer; const SourceStride: NativeInt;
  const Width, Height: Integer; const Order: TRGBByteOrder; const Bicubic: Boolean = False): TBytes;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF};

type
  TResizeRow = record
    First: Integer;
    Weights: TArray<Integer>;
  end;

function CubicWeight(const Distance: Double): Double;
var
  X: Double;
begin
  X := Abs(Distance);
  if X < 1 then
    Result := ((1.5 * X - 2.5) * X) * X + 1
  else if X < 2 then
    Result := (((-0.5 * X + 2.5) * X - 4) * X) + 2
  else
    Result := 0;
end;

function MakeResizeRows(const InputSize, OutputSize: Integer; const Bicubic: Boolean): TArray<TResizeRow>;
const
  UnitWeight = 1 shl 22;
var
  Scale, FilterScale, Radius, Center, Weight, Sum: Double;
  I, J, First, Finish: Integer;
  Coefficients: TArray<Double>;
begin
  SetLength(Result, OutputSize);
  Scale := InputSize / OutputSize;
  FilterScale := Max(1.0, Scale);
  Radius := FilterScale;
  if Bicubic then Radius := 2 * Radius;
  for I := 0 to OutputSize - 1 do
  begin
    Center := (I + 0.5) * Scale;
    First := Max(0, Trunc(Center - Radius + 0.5));
    Finish := Min(InputSize, Trunc(Center + Radius + 0.5));
    Result[I].First := First;
    SetLength(Coefficients, Finish - First);
    SetLength(Result[I].Weights, Length(Coefficients));
    Sum := 0;
    for J := 0 to High(Coefficients) do
    begin
      Weight := (J + First + 0.5 - Center) / FilterScale;
      if Bicubic then
        Weight := CubicWeight(Weight)
      else
        Weight := Max(0.0, 1.0 - Abs(Weight));
      Coefficients[J] := Weight;
      Sum := Sum + Weight;
    end;
    for J := 0 to High(Coefficients) do
    begin
      Weight := Coefficients[J] * UnitWeight / Sum;
      if Weight >= 0 then
        Result[I].Weights[J] := Trunc(Weight + 0.5)
      else
        Result[I].Weights[J] := Trunc(Weight - 0.5);
    end;
  end;
end;

function ResizePillowRGB(const Source: Pointer; const SourceBytes: NativeInt;
  const SourceWidth, SourceHeight: Integer; const SourceStride: NativeInt;
  const Width, Height: Integer; const Order: TRGBByteOrder; const Bicubic: Boolean): TBytes;
const
  Half = 1 shl 21;
var
  XWeights, YWeights: TArray<TResizeRow>;
  Horizontal: TBytes;
  Line: PByte;
  X, Y, C, I, InputChannel: Integer;
  Sum, Extent: Int64;
begin
  if (Source = nil) or (SourceWidth <= 0) or (SourceHeight <= 0) or
    (SourceWidth > High(Integer) div 3) or (Width <= 0) or (Height <= 0) or
    not (Order in [rboRGB, rboBGR]) then
      raise EArgumentException.Create('Invalid RGB resize input');
  if (SourceStride < Int64(SourceWidth) * 3) or
    (SourceStride > High(NativeInt) div SourceHeight) then
      raise EArgumentException.Create('Invalid RGB row stride');
  Extent := Int64(SourceHeight - 1) * SourceStride + Int64(SourceWidth) * 3;
  if (SourceBytes < Extent) or (Int64(Width) * SourceHeight > High(Integer) div 3) or
    (Int64(Width) * Height > High(Integer) div 3) then
      raise EArgumentException.Create('RGB resize exceeds accessible storage');
  XWeights := MakeResizeRows(SourceWidth, Width, Bicubic);
  YWeights := MakeResizeRows(SourceHeight, Height, Bicubic);
  SetLength(Horizontal, SourceHeight * Width * 3);
  for Y := 0 to SourceHeight - 1 do
  begin
    Line := PByte(Source) + NativeInt(Y) * SourceStride;
    for X := 0 to Width - 1 do
      for C := 0 to 2 do
      begin
        if Order = rboRGB then
          InputChannel := C
        else
          InputChannel := 2 - C;
        Sum := Half;
        for I := 0 to High(XWeights[X].Weights) do
          Inc(Sum, Int64(XWeights[X].Weights[I]) *
            Line[(XWeights[X].First + I) * 3 + InputChannel]);
        if Sum < 0 then Sum := 0;
        Horizontal[(Y * Width + X) * 3 + C] := Min(255, Sum shr 22);
      end;
  end;
  SetLength(Result, Height * Width * 3);
  for Y := 0 to Height - 1 do
    for X := 0 to Width - 1 do
      for C := 0 to 2 do
      begin
        Sum := Half;
        for I := 0 to High(YWeights[Y].Weights) do
          Inc(Sum, Int64(YWeights[Y].Weights[I]) *
            Horizontal[((YWeights[Y].First + I) * Width + X) * 3 + C]);
        if Sum < 0 then Sum := 0;
        Result[(Y * Width + X) * 3 + C] := Min(255, Sum shr 22);
      end;
end;

end.
