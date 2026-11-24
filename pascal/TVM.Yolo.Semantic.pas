unit TVM.Yolo.Semantic;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM.CpuGraph;

{ Preserve float class evidence through both source bilinear resizes. ArgMax
  runs last at original resolution. Planes are streamed to bound working RAM. }
function RestoreYoloSemanticLogits(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, Classes, InputWidth, InputHeight, OriginalWidth, OriginalHeight: Integer;
  const ShouldStop: TCpuGraphStop = nil): TBytes;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.Image.FloatResize, TVM.Yolo.Transform;

function RestoreYoloSemanticLogits(const Source: TArray<Single>;
  const SourceWidth, SourceHeight, Classes, InputWidth, InputHeight, OriginalWidth, OriginalHeight: Integer;
  const ShouldStop: TCpuGraphStop): TBytes;
var
  Plane, Input, Cropped, Restored, Best: TArray<Single>;
  C, P, Pixels, SourcePixels, Left, Top, Right, Bottom: Integer;
  Span: TYoloImageTransform;

  procedure CheckStop;
  begin
    if Assigned(ShouldStop) and ShouldStop() then
      raise ECpuGraphStopped.Create('YOLO semantic restoration stopped');
  end;

begin
  CheckStop;
  if (SourceWidth <= 0) or (SourceHeight <= 0) or (Classes < 2) or (Classes > 256) or
    (InputWidth <= 0) or (InputHeight <= 0) or (OriginalWidth <= 0) or (OriginalHeight <= 0) then
      raise EArgumentException.Create('Invalid YOLO semantic logit dimensions');
  if (Int64(SourceWidth) * SourceHeight > High(Integer) div Classes) or
    (Int64(InputWidth) * InputHeight > High(Integer) div SizeOf(Single)) or
    (Int64(OriginalWidth) * OriginalHeight > High(Integer) div SizeOf(Single)) or
    (Int64(SourceWidth) * SourceHeight * Classes <> Length(Source)) then
      raise EArgumentException.Create('YOLO semantic logits exceed the indexing range');
  for P := 0 to High(Source) do
    if IsNan(Source[P]) or IsInfinite(Source[P]) then
      raise EArgumentException.Create('Nonfinite YOLO semantic logit');
  SourcePixels := SourceWidth * SourceHeight;
  Pixels := OriginalWidth * OriginalHeight;
  { The pinned source scale_masks uses letterbox_span, including its rounded
    content dimensions and asymmetric integer padding. }
  Span := TYoloImageTransform.Letterbox(OriginalWidth, OriginalHeight, InputWidth, InputHeight);
  Left := Span.Left;
  Top := Span.Top;
  Right := Left + Span.ResizedWidth;
  Bottom := Top + Span.ResizedHeight;
  SetLength(Result, Pixels);
  SetLength(Best, Pixels);
  for C := 0 to Classes - 1 do
  begin
    CheckStop;
    Plane := Copy(Source, C * SourcePixels, SourcePixels);
    Input := ResizeFloatBilinear(Plane, SourceWidth, SourceHeight, InputWidth, InputHeight, False);
    Cropped := CropFloatPlane(Input, InputWidth, InputHeight, Left, Top, Right - Left, Bottom - Top);
    Restored := ResizeFloatBilinear(Cropped, Right - Left, Bottom - Top, OriginalWidth,
      OriginalHeight, False);
    for P := 0 to Pixels - 1 do
    begin
      if (P and 16383) = 0 then CheckStop;
      if (C = 0) or (Restored[P] > Best[P]) then
      begin
        Best[P] := Restored[P];
        Result[P] := C;
      end;
    end;
  end;
  CheckStop;
end;

end.
