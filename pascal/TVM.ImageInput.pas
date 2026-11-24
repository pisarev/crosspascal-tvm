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
  OpenCVConsts, TVM.CpuGraph, TVM.Image.Resample, CrossPascal.NativeIO;

function ReadImageBytes(const FileName: string): TBytes;
var
  Stream: TStream;
begin
  Stream := TNativeReadStream.Create(FileName);
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

function ResizeColorImage(const Image: IMat; const Width, Height: Integer;
  const Bicubic: Boolean): TBytes;
var
  Extent: NativeInt;
begin
  RequireCpu(Image <> nil, 'Expected a color image');
  RequireCpu(not Image.IsEmpty and (Image.Channels = 3) and (Image.Depth = CV_8U),
    'Expected a nonempty uint8 three-channel color image');
  RequireCpu(
    (Width > 0) and (Height > 0) and (Width <= 4096) and (Height <= 4096) and (Int64(Image.Height) * Width * 3 <= 512 * 1024 * 1024),
    'Image resize exceeds the working memory limit'
  );
  RequireCpu((Image.Step > 0) and (Image.Step <= High(NativeInt) div Image.Height),
    'Image row stride exceeds the address space');
  Extent := NativeInt(Image.Height - 1) * Image.Step + NativeInt(Image.Width) * 3;
  Result := ResizePillowRGB(Image.DataPtr, Extent, Image.Width, Image.Height,
    Image.Step, Width, Height, rboBGR, Bicubic);
end;

end.
