{ ************************************************************************** }
{                                                                            }
{ TVM.OmDet                                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.OmDet;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossVision.Mat;

const
  OmDetImageSide = 640;
  OmDetQueryCount = 900;
  OmDetClassCapacity = 16;

type
  TOmDetDetection = record
    Left: Single;
    Top: Single;
    Right: Single;
    Bottom: Single;
    Score: Single;
    ClassIndex: Integer;
    QueryIndex: Integer;
  end;
  TOmDetDetections = TArray<TOmDetDetection>;

function OmDetImageInput(const Image: IMat): TArray<Single>;
function OmDetDecode(const Boxes, Logits: TArray<Single>; const ClassCount, ImageWidth, ImageHeight: Integer;
  const ScoreThreshold, NmsThreshold: Single): TOmDetDetections;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.CpuGraph, TVM.ImageInput,
  OpenCVConsts, OpenCVTypes;

type
  TCandidate = record
    Left: Double;
    Top: Double;
    Right: Double;
    Bottom: Double;
    Score: Single;
    ClassIndex: Integer;
    QueryIndex: Integer;
  end;
  TCandidates = TArray<TCandidate>;

function OmDetImageInput(const Image: IMat): TArray<Single>;
const
  Mean: array[0..2] of Single = (123.675, 116.28, 103.53);
  Std: array[0..2] of Single = (58.395, 57.12, 57.375);
var
  Pixels: TBytes;
  X, Y, C: Integer;
  Value: Single;
begin
  RequireCpu((Image <> nil) and not Image.IsEmpty and (Image.Depth = CV_8U) and
    (Image.Channels = 3) and (Image.Width > 0) and (Image.Height > 0) and
    (Image.Width <= 4096) and (Image.Height <= 4096) and (Image.DataPtr <> nil) and
    (Image.Step >= NativeInt(Image.Width) * 3),
    'OmDet requires a BGR8 image up to 4096 pixels per side');
  Pixels := ResizeColorImage(Image, OmDetImageSide, OmDetImageSide, False);
  SetLength(Result, 3 * OmDetImageSide * OmDetImageSide);
  for Y := 0 to OmDetImageSide - 1 do
    for X := 0 to OmDetImageSide - 1 do
      for C := 0 to 2 do
      begin
        Value := Pixels[(Y * OmDetImageSide + X) * 3 + C];
        Value := Value - Mean[C];
        Result[(C * OmDetImageSide + Y) * OmDetImageSide + X] := Value / Std[C];
      end;
end;

function Finite(const Value: Single): Boolean;
var
  Bits: Cardinal;
begin
  Move(Value, Bits, SizeOf(Bits));
  Result := (Bits and $7F800000) <> $7F800000;
end;

function Probability(const Logit: Single): Single;
var
  Value: Double;
begin
  if Logit >= 80 then Exit(1);
  if Logit <= -104 then Exit(0);
  Value := Logit;
  if Value >= 0 then
    Result := 1 / (1 + Exp(-Value))
  else begin
    Value := Exp(Value);
    Result := Value / (1 + Value);
  end;
end;

function Before(const A, B: TCandidate): Boolean;
begin
  if A.Score <> B.Score then Exit(A.Score > B.Score);
  if A.QueryIndex <> B.QueryIndex then Exit(A.QueryIndex < B.QueryIndex);
  Result := A.ClassIndex < B.ClassIndex;
end;

procedure Sort(var Items: TCandidates; const First, Last: Integer);
var
  I, J: Integer;
  Pivot, Temporary: TCandidate;
begin
  I := First;
  J := Last;
  Pivot := Items[First + (Last - First) div 2];
  repeat
    while Before(Items[I], Pivot) do Inc(I);
    while Before(Pivot, Items[J]) do Dec(J);
    if I <= J then
    begin
      Temporary := Items[I];
      Items[I] := Items[J];
      Items[J] := Temporary;
      Inc(I);
      Dec(J);
    end;
  until I > J;
  if First < J then Sort(Items, First, J);
  if I < Last then Sort(Items, I, Last);
end;

function Overlap(const A, B: TCandidate): Double;
var
  Width, Height, Intersection, UnionArea: Double;
begin
  Width := Max(0.0, Min(A.Right, B.Right) - Max(A.Left, B.Left));
  Height := Max(0.0, Min(A.Bottom, B.Bottom) - Max(A.Top, B.Top));
  Intersection := Width * Height;
  UnionArea := (A.Right - A.Left) * (A.Bottom - A.Top) +
    (B.Right - B.Left) * (B.Bottom - B.Top) - Intersection;
  if UnionArea > 0 then
    Result := Intersection / UnionArea
  else
    Result := 0;
end;

function OmDetDecode(const Boxes, Logits: TArray<Single>; const ClassCount, ImageWidth, ImageHeight: Integer;
  const ScoreThreshold, NmsThreshold: Single): TOmDetDetections;
var
  Candidates, Kept: TCandidates;
  Item: TCandidate;
  Query, ClassIndex, I, J, Count, Limit: Integer;
  CX, CY, HalfWidth, HalfHeight: Double;
  Keep: Boolean;
begin
  RequireCpu((ClassCount >= 1) and (ClassCount <= OmDetClassCapacity) and
    (ImageWidth > 0) and (ImageWidth <= 4096) and (ImageHeight > 0) and
    (ImageHeight <= 4096), 'Invalid OmDet class count or image size');
  RequireCpu((Length(Boxes) = OmDetQueryCount * 4) and
    (Length(Logits) = OmDetQueryCount * OmDetClassCapacity), 'OmDet output sizes differ');
  RequireCpu(Finite(ScoreThreshold) and Finite(NmsThreshold) and
    (ScoreThreshold >= 0) and (ScoreThreshold <= 1) and
    (NmsThreshold >= 0) and (NmsThreshold <= 1), 'Invalid OmDet thresholds');
  for I := 0 to High(Boxes) do
    RequireCpu(Finite(Boxes[I]) and (Boxes[I] >= 0) and (Boxes[I] <= 1),
      'OmDet boxes must be finite normalized coordinates');
  for I := 0 to High(Logits) do
    RequireCpu(Finite(Logits[I]), 'Non-finite OmDet logit');
  SetLength(Candidates, OmDetQueryCount * ClassCount);
  Count := 0;
  for Query := 0 to OmDetQueryCount - 1 do
  begin
    CX := Boxes[Query * 4];
    CY := Boxes[Query * 4 + 1];
    HalfWidth := Boxes[Query * 4 + 2];
    HalfHeight := Boxes[Query * 4 + 3];
    HalfWidth := HalfWidth * 0.5;
    HalfHeight := HalfHeight * 0.5;
    for ClassIndex := 0 to ClassCount - 1 do
    begin
      Item.QueryIndex := Query;
      Item.ClassIndex := ClassIndex;
      Item.Score := Probability(Logits[Query * OmDetClassCapacity + ClassIndex]);
      Item.Left := (CX - HalfWidth) * ImageWidth;
      Item.Top := (CY - HalfHeight) * ImageHeight;
      Item.Right := (CX + HalfWidth) * ImageWidth;
      Item.Bottom := (CY + HalfHeight) * ImageHeight;
      Candidates[Count] := Item;
      Inc(Count);
    end;
  end;
  Sort(Candidates, 0, High(Candidates));
  Limit := Min(OmDetQueryCount, Length(Candidates));
  SetLength(Kept, Limit);
  Count := 0;
  for I := 0 to Limit - 1 do
  begin
    Item := Candidates[I];
    if Item.Score <= ScoreThreshold then Break;
    Keep := True;
    for J := 0 to Count - 1 do
      if (Kept[J].ClassIndex = Item.ClassIndex) and (Overlap(Kept[J], Item) > NmsThreshold) then
      begin
        Keep := False;
        Break;
      end;
    if Keep then
    begin
      Kept[Count] := Item;
      Inc(Count);
    end;
  end;
  SetLength(Result, Count);
  for I := 0 to Count - 1 do
  begin
    Item := Kept[I];
    Result[I].Left := Max(0.0, Min(ImageWidth, Item.Left));
    Result[I].Top := Max(0.0, Min(ImageHeight, Item.Top));
    Result[I].Right := Max(0.0, Min(ImageWidth, Item.Right));
    Result[I].Bottom := Max(0.0, Min(ImageHeight, Item.Bottom));
    Result[I].Score := Item.Score;
    Result[I].ClassIndex := Item.ClassIndex;
    Result[I].QueryIndex := Item.QueryIndex;
  end;
end;

end.
