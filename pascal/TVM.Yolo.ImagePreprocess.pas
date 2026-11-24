{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.ImagePreprocess                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.ImagePreprocess;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM.Image.Resample;

{ Evaluation transform used by modern Ultralytics classification profiles:
  uint8 Pillow BILINEAR resize, shortest edge for a square target, CenterCrop,
  RGB Float32 NCHW, /255, then channel-wise normalization. The caller supplies
  the normalization from its selected profile; this does not run the model. }
function PrepareYoloPillowClassification(const Source: Pointer; const SourceBytes: NativeInt;
  const SourceWidth, SourceHeight: Integer; const SourceStride: NativeInt;
  const Width, Height: Integer; const Order: TRGBByteOrder;
  const Mean, Std: array of Single): TArray<Single>;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF};

function CenterCropOffset(const Difference: Integer): Integer;
begin
  Result := Difference div 2;
  { Python round: exact half-integer ties select the even integer. }
  if Odd(Difference) and Odd(Result) then Inc(Result);
end;

function PrepareYoloPillowClassification(const Source: Pointer; const SourceBytes: NativeInt;
  const SourceWidth, SourceHeight: Integer; const SourceStride: NativeInt;
  const Width, Height: Integer; const Order: TRGBByteOrder;
  const Mean, Std: array of Single): TArray<Single>;
var
  RGB: TBytes;
  ResizedWidth, ResizedHeight: Int64;
  Left, Top, X, Y, Channel, Plane: Integer;
  Pixel: Single;
begin
  if (SourceWidth <= 0) or (SourceHeight <= 0) or (Width <= 0) or (Height <= 0) or
    (Int64(Width) * Height > High(Integer) div 3) or
    (Length(Mean) <> 3) or (Length(Std) <> 3) then
      raise EArgumentException.Create('Invalid YOLO classification input profile');
  for Channel := 0 to 2 do
    if IsNan(Mean[Channel]) or IsInfinite(Mean[Channel]) or
      IsNan(Std[Channel]) or IsInfinite(Std[Channel]) or (Std[Channel] <= 0) then
        raise EArgumentException.Create('Invalid YOLO classification normalization');
  ResizedWidth := Width;
  ResizedHeight := Height;
  if Width = Height then
    if SourceWidth <= SourceHeight then
      ResizedHeight := Int64(Width) * SourceHeight div SourceWidth
    else
      ResizedWidth := Int64(Height) * SourceWidth div SourceHeight;
  if (ResizedWidth > High(Integer)) or (ResizedHeight > High(Integer)) then
    raise EArgumentException.Create('YOLO classification resize exceeds Integer dimensions');
  RGB := ResizePillowRGB(Source, SourceBytes, SourceWidth, SourceHeight, SourceStride,
    ResizedWidth, ResizedHeight, Order, False);
  Left := CenterCropOffset(ResizedWidth - Width);
  Top := CenterCropOffset(ResizedHeight - Height);
  Plane := Width * Height;
  SetLength(Result, Plane * 3);
  for Channel := 0 to 2 do
    for Y := 0 to Height - 1 do
      for X := 0 to Width - 1 do
      begin
        Pixel := RGB[((Y + Top) * ResizedWidth + X + Left) * 3 + Channel];
        Pixel := Pixel / 255;
        Pixel := Pixel - Mean[Channel];
        Result[Channel * Plane + Y * Width + X] := Pixel / Std[Channel];
      end;
end;

end.
