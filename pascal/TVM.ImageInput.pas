{ ************************************************************************** }
{                                                                            }
{ TVM.ImageInput                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.ImageInput;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossVision.Mat;

function ReadImageBytes(const FileName: string): TBytes;
function DecodeColorImage(const Bytes: TBytes; const CVLibrary: string): IMat;
function ResizeColorImage(const Image: IMat; const Width, Height: Integer;
  const Bicubic: Boolean): TBytes;

implementation

uses
  {$IFDEF FPC}Classes, Math{$ELSE}System.Classes, System.Math{$ENDIF}, OpenCV, OpenCVTypes,
  OpenCVConsts, TVM.CpuGraph;

type
  TResizeRow = record
    First: Integer;
    Weights: TArray<Integer>;
  end;

function ReadImageBytes(const FileName: string): TBytes;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= 256 * 1024 * 1024), 'Invalid image file size');
    SetLength(Result, Stream.Size);
    Stream.ReadBuffer(Result[0], Length(Result));
  finally
    Stream.Free;
  end;
end;

function DecodeColorImage(const Bytes: TBytes; const CVLibrary: string): IMat;
var
  Raw: TMat;
begin
  if not CV.Module.Loaded then
    if not CVInitialize(CVLibrary) then raise ECpuGraph.Create(CV.Module.Error);
  Raw := nil;
  RequireCpu(CV.ImgCodecs.ImDecode(Pointer(Bytes), Length(Bytes), IMREAD_COLOR, @Raw),
    'Could not decode the image');
  Result := WrapMat(Raw);
  RequireCpu(not Result.IsEmpty and (Result.Channels = 3) and (Int64(Result.Width) * Result.Height <= 64000000),
    'Expected a color image of at most 64 megapixels');
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

function ResizeColorImage(const Image: IMat; const Width, Height: Integer;
  const Bicubic: Boolean): TBytes;
const
  Half = 1 shl 21;
var
  XWeights, YWeights: TArray<TResizeRow>;
  Horizontal: TBytes;
  Line: PByte;
  X, Y, C, I, Sum: Integer;
begin
  RequireCpu(
    (Width > 0) and (Height > 0) and (Width <= 4096) and (Height <= 4096) and (Int64(Image.Height) * Width * 3 <= 512 * 1024 * 1024),
    'Image resize exceeds the working memory limit'
  );
  XWeights := MakeResizeRows(Image.Width, Width, Bicubic);
  YWeights := MakeResizeRows(Image.Height, Height, Bicubic);
  SetLength(Horizontal, Image.Height * Width * 3);
  for Y := 0 to Image.Height - 1 do
  begin
    Line := PByte(Image.DataPtr) + NativeInt(Y) * Image.Step;
    for X := 0 to Width - 1 do
      for C := 0 to 2 do
      begin
        Sum := Half;
        for I := 0 to High(XWeights[X].Weights) do
          Inc(Sum, XWeights[X].Weights[I] * Line[(XWeights[X].First + I) * 3 + 2 - C]);
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
          Inc(Sum, YWeights[Y].Weights[I] * Horizontal[((YWeights[Y].First + I) * Width + X) * 3 + C]);
        if Sum < 0 then Sum := 0;
        Result[(Y * Width + X) * 3 + C] := Min(255, Sum shr 22);
      end;
end;

end.
