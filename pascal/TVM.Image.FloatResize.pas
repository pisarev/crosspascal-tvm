{ ************************************************************************** }
{                                                                            }
{ TVM.Image.FloatResize                                                      }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Image.FloatResize;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF};

{ Contiguous single-plane images. Bilinear uses Float32 coordinates and
  weights as in PyTorch interpolate; class IDs use its legacy nearest mode. }
function ResizeFloatBilinear(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, Width, Height: Integer; const AlignCorners: Boolean): TArray<Single>;
function ResizeClassMapNearest(const Source: TBytes;
  const SourceWidth, SourceHeight, Width, Height: Integer): TBytes;
function CropFloatPlane(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, Left, Top, Width, Height: Integer): TArray<Single>;
function CropClassMap(const Source: TBytes;
  const SourceWidth, SourceHeight, Left, Top, Width, Height: Integer): TBytes;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF};

type
  TLinearSample = record
    First: Integer;
    Second: Integer;
    Weight0: Single;
    Weight1: Single;
  end;

function PixelCount(const Width, Height, ElementBytes: Integer): Integer;
var
  Count: Int64;
begin
  Count := Int64(Width) * Height;
  if (Width <= 0) or (Height <= 0) or (Count > High(Integer) div ElementBytes) then
    raise EArgumentException.Create('Image dimensions exceed the indexing range');
  Result := Count;
end;

function LinearAxis(const InputSize, OutputSize: Integer;
  const AlignCorners: Boolean): TArray<TLinearSample>;
var
  Scale, Coordinate: Single;
  I: Integer;
begin
  if AlignCorners then
  begin
    Scale := 0;
    if OutputSize > 1 then Scale := (InputSize - 1) / (OutputSize - 1);
  end
  else
    Scale := InputSize / OutputSize;
  SetLength(Result, OutputSize);
  for I := 0 to OutputSize - 1 do
  begin
    if InputSize = OutputSize then
    begin
      Result[I].First := I;
      Result[I].Second := I;
      Result[I].Weight0 := 1;
      Result[I].Weight1 := 0;
      Continue;
    end;
    if AlignCorners then
      Coordinate := Scale * I
    else begin
      Coordinate := I + 0.5;
      Coordinate := Coordinate * Scale;
      Coordinate := Coordinate - 0.5;
      if Coordinate < 0 then Coordinate := 0;
    end;
    Result[I].First := Min(Floor(Coordinate), InputSize - 1);
    Result[I].Second := Min(Result[I].First + 1, InputSize - 1);
    Result[I].Weight1 := Min(1.0, Max(0.0, Coordinate - Result[I].First));
    Result[I].Weight0 := 1 - Result[I].Weight1;
  end;
end;

function ResizeFloatBilinear(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, Width, Height: Integer; const AlignCorners: Boolean): TArray<Single>;
var
  XAxis, YAxis: TArray<TLinearSample>;
  X, Y, Row0, Row1, Count: Integer;
  A, B, Upper, Lower: Single;
begin
  if Length(Source) <> PixelCount(SourceWidth, SourceHeight, SizeOf(Single)) then
    raise EArgumentException.Create('Float image byte count differs from its shape');
  Count := PixelCount(Width, Height, SizeOf(Single));
  if (Width = SourceWidth) and (Height = SourceHeight) then Exit(Copy(Source));
  XAxis := LinearAxis(SourceWidth, Width, AlignCorners);
  YAxis := LinearAxis(SourceHeight, Height, AlignCorners);
  SetLength(Result, Count);
  for Y := 0 to Height - 1 do
  begin
    Row0 := YAxis[Y].First * SourceWidth;
    Row1 := YAxis[Y].Second * SourceWidth;
    for X := 0 to Width - 1 do
    begin
      A := Source[Row0 + XAxis[X].First] * XAxis[X].Weight0;
      B := Source[Row0 + XAxis[X].Second] * XAxis[X].Weight1;
      Upper := A + B;
      A := Source[Row1 + XAxis[X].First] * XAxis[X].Weight0;
      B := Source[Row1 + XAxis[X].Second] * XAxis[X].Weight1;
      Lower := A + B;
      A := Upper * YAxis[Y].Weight0;
      B := Lower * YAxis[Y].Weight1;
      Result[Y * Width + X] := A + B;
    end;
  end;
end;

function NearestAxis(const InputSize, OutputSize: Integer): TArray<Integer>;
var
  Scale, Coordinate: Single;
  I: Integer;
begin
  Scale := InputSize / OutputSize;
  SetLength(Result, OutputSize);
  for I := 0 to OutputSize - 1 do
    if InputSize = OutputSize then
      Result[I] := I
    else if Int64(OutputSize) = Int64(InputSize) * 2 then
      Result[I] := I div 2
    else begin
      Coordinate := I * Scale;
      Result[I] := Min(Floor(Coordinate), InputSize - 1);
    end;
end;

function ResizeClassMapNearest(const Source: TBytes;
  const SourceWidth, SourceHeight, Width, Height: Integer): TBytes;
var
  XAxis, YAxis: TArray<Integer>;
  X, Y, Count: Integer;
begin
  if Length(Source) <> PixelCount(SourceWidth, SourceHeight, 1) then
    raise EArgumentException.Create('Class-map byte count differs from its shape');
  Count := PixelCount(Width, Height, 1);
  XAxis := NearestAxis(SourceWidth, Width);
  YAxis := NearestAxis(SourceHeight, Height);
  SetLength(Result, Count);
  for Y := 0 to Height - 1 do
    for X := 0 to Width - 1 do
      Result[Y * Width + X] := Source[YAxis[Y] * SourceWidth + XAxis[X]];
end;

procedure ValidateCrop(const SourceWidth, SourceHeight, Left, Top, Width, Height: Integer);
begin
  if (Left < 0) or (Top < 0) or (Width <= 0) or (Height <= 0) or
    (Width > SourceWidth) or (Height > SourceHeight) or
    (Left > SourceWidth - Width) or (Top > SourceHeight - Height) then
      raise EArgumentException.Create('Image crop lies outside the source');
end;

function CropFloatPlane(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, Left, Top, Width, Height: Integer): TArray<Single>;
var
  Y: Integer;
begin
  if Length(Source) <> PixelCount(SourceWidth, SourceHeight, SizeOf(Single)) then
    raise EArgumentException.Create('Float image byte count differs from its shape');
  ValidateCrop(SourceWidth, SourceHeight, Left, Top, Width, Height);
  SetLength(Result, PixelCount(Width, Height, SizeOf(Single)));
  for Y := 0 to Height - 1 do
    Move(Source[(Top + Y) * SourceWidth + Left], Result[Y * Width], Width * SizeOf(Single));
end;

function CropClassMap(const Source: TBytes;
  const SourceWidth, SourceHeight, Left, Top, Width, Height: Integer): TBytes;
var
  Y: Integer;
begin
  if Length(Source) <> PixelCount(SourceWidth, SourceHeight, 1) then
    raise EArgumentException.Create('Class-map byte count differs from its shape');
  ValidateCrop(SourceWidth, SourceHeight, Left, Top, Width, Height);
  SetLength(Result, PixelCount(Width, Height, 1));
  for Y := 0 to Height - 1 do
    Move(Source[(Top + Y) * SourceWidth + Left], Result[Y * Width], Width);
end;

end.
