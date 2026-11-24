{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.MatInput                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.MatInput;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossVision.Mat, TVM.Yolo.PixelPreprocess;

function PrepareYoloMatPixels(const Image: IMat; const Plan: TYoloPixelPlan;
  const Mean, Std: array of Single): TArray<Single>;
function PrepareYoloMatPillowClassification(const Image: IMat; const Width, Height: Integer;
  const Mean, Std: array of Single): TArray<Single>;

implementation

uses
  OpenCVConsts, TVM.Image.Resample, TVM.Yolo.ImagePreprocess;

function RequireBGRImage(const Image: IMat): NativeInt;
var
  Required: Int64;
begin
  if (Image = nil) or Image.IsEmpty then
    raise EArgumentException.Create('YOLO input image is empty');
  if (Image.GetDepth <> CV_8U) or (Image.GetChannels <> 3) or
    (Image.GetWidth <= 0) or (Image.GetHeight <= 0) or (Image.DataPtr = nil) or
    (Int64(Image.GetStep) < Int64(Image.GetWidth) * 3) then
      raise EArgumentException.Create('YOLO input image must contain uint8 BGR pixels');
  Required := Int64(Image.GetHeight - 1) * Image.GetStep + Int64(Image.GetWidth) * 3;
  if Required > High(NativeInt) then
    raise EArgumentException.Create('YOLO image buffer exceeds address range');
  Result := Required;
end;

function PrepareYoloMatPixels(const Image: IMat; const Plan: TYoloPixelPlan;
  const Mean, Std: array of Single): TArray<Single>;
var
  Cropped, Resized: IMat;
  Bytes: NativeInt;
begin
  RequireBGRImage(Image);
  if (Image.GetWidth <> Plan.OriginalWidth) or (Image.GetHeight <> Plan.OriginalHeight) then
    raise EArgumentException.Create('YOLO image does not match its pixel preparation plan');
  Cropped := Image;
  if (Plan.CropX <> 0) or (Plan.CropY <> 0) or
    (Plan.CropWidth <> Image.GetWidth) or (Plan.CropHeight <> Image.GetHeight) then
      Cropped := Image.Crop(Plan.CropX, Plan.CropY, Plan.CropWidth, Plan.CropHeight);
  RequireBGRImage(Cropped);
  Resized := Cropped;
  if (Cropped.GetWidth <> Plan.ResizedWidth) or (Cropped.GetHeight <> Plan.ResizedHeight) then
    Resized := Cropped.Resize(Plan.ResizedWidth, Plan.ResizedHeight, INTER_LINEAR);
  Bytes := RequireBGRImage(Resized);
  if (Resized.GetWidth <> Plan.ResizedWidth) or (Resized.GetHeight <> Plan.ResizedHeight) then
    raise EArgumentException.Create('YOLO native resize returned unexpected image dimensions');
  Result := PackYoloResizedInput(Resized.DataPtr, Bytes, Resized.GetStep, Plan, rboBGR, Mean, Std);
end;

function PrepareYoloMatPillowClassification(const Image: IMat; const Width, Height: Integer;
  const Mean, Std: array of Single): TArray<Single>;
var
  Bytes: NativeInt;
begin
  Bytes := RequireBGRImage(Image);
  Result := PrepareYoloPillowClassification(Image.DataPtr, Bytes, Image.GetWidth, Image.GetHeight,
    Image.GetStep, Width, Height, rboBGR, Mean, Std);
end;

end.
