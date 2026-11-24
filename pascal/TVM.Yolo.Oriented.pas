{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Oriented                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Oriented;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  TVM.Yolo.Detections;

type
  TYoloRotatedBox = record
    CenterX: Single;
    CenterY: Single;
    Width: Single;
    Height: Single;
    Angle: Single;
  end;

  TYoloOrientedDetection = record
    Box: TYoloRotatedBox;
    Score: Single;
    ClassID: Integer;
    SourceIndex: Integer;
  end;
  TYoloOrientedDetections = TArray<TYoloOrientedDetection>;

function YoloProbabilisticIoU(const A, B: TYoloRotatedBox): Double;
function RegularizeYoloRotatedBox(const Box: TYoloRotatedBox): TYoloRotatedBox;
{ Dense [xywh, class scores, angle], or end-to-end [xywh, score, class, angle].
  Dense output uses source ProbIoU and fast NMS; end-to-end output skips NMS.
  Suppressed dense boxes still suppress lower-scored boxes.
  The output is regularized but remains in model-input coordinates. }
function DecodeYoloOrientedDetections(const Data: TArray<Single>;
  const BoxCount, FeatureCount: Integer; const Options: TYoloDetectionOptions;
  const ClassOffset: Single): TYoloOrientedDetections;

implementation

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF};

type
  TGaussianBox = record
    X: Double;
    Y: Double;
    A: Double;
    B: Double;
    C: Double;
  end;

procedure FiniteValue(const Value: Single);
begin
  if IsNan(Value) or IsInfinite(Value) then
    raise EArgumentException.Create('Nonfinite YOLO oriented box');
end;

function Gaussian(const Box: TYoloRotatedBox): TGaussianBox;
var
  Width, Height, VarianceX, VarianceY, Cosine, Sine: Double;
begin
  FiniteValue(Box.CenterX);
  FiniteValue(Box.CenterY);
  FiniteValue(Box.Width);
  FiniteValue(Box.Height);
  FiniteValue(Box.Angle);
  if (Box.Width < 0) or (Box.Height < 0) then
    raise EArgumentException.Create('Negative YOLO oriented box size');
  Width := Box.Width;
  Height := Box.Height;
  VarianceX := Width * Width / 12;
  VarianceY := Height * Height / 12;
  Cosine := Cos(Box.Angle);
  Sine := Sin(Box.Angle);
  Result.X := Box.CenterX;
  Result.Y := Box.CenterY;
  Result.A := VarianceX * Cosine * Cosine + VarianceY * Sine * Sine;
  Result.B := VarianceX * Sine * Sine + VarianceY * Cosine * Cosine;
  Result.C := (VarianceX - VarianceY) * Cosine * Sine;
end;

function ProbabilityIoU(const First, Second: TGaussianBox): Double;
const
  Epsilon = 1E-7;
var
  A, B, C, DX, DY, Determinant, Denominator, T1, T2, T3, Distance: Double;
begin
  A := First.A + Second.A;
  B := First.B + Second.B;
  C := First.C + Second.C;
  DX := First.X - Second.X;
  DY := First.Y - Second.Y;
  Determinant := A * B - C * C;
  Denominator := Determinant + Epsilon;
  T1 := (A * DY * DY + B * DX * DX) / Denominator * 0.25;
  T2 := -C * DX * DY / Denominator * 0.5;
  T3 := Ln(Determinant / (4 * Sqrt(Max(0.0, First.A * First.B - First.C * First.C) *
    Max(0.0, Second.A * Second.B - Second.C * Second.C)) + Epsilon) + Epsilon) * 0.5;
  Distance := Min(100.0, Max(Epsilon, T1 + T2 + T3));
  Result := 1 - Sqrt(1 - Exp(-Distance) + Epsilon);
end;

function YoloProbabilisticIoU(const A, B: TYoloRotatedBox): Double;
begin
  Result := ProbabilityIoU(Gaussian(A), Gaussian(B));
end;

function PositiveModulo(const Value, Period: Double): Single;
begin
  Result := Value - Floor(Value / Period) * Period;
end;

function RegularizeYoloRotatedBox(const Box: TYoloRotatedBox): TYoloRotatedBox;
const
  SinglePi: Single = Pi;
  SingleHalfPi: Single = Pi / 2;
begin
  FiniteValue(Box.Angle);
  Result := Box;
  if PositiveModulo(Box.Angle, SinglePi) >= SingleHalfPi then
  begin
    Result.Width := Box.Height;
    Result.Height := Box.Width;
  end;
  Result.Angle := PositiveModulo(Box.Angle, SingleHalfPi);
end;

function DecodeYoloOrientedDetections(const Data: TArray<Single>;
  const BoxCount, FeatureCount: Integer; const Options: TYoloDetectionOptions;
  const ClassOffset: Single): TYoloOrientedDetections;
var
  Candidates: TYoloRawDetections;
  Boxes: TArray<TYoloRotatedBox>;
  Gaussians: TArray<TGaussianBox>;
  Suppressed: TArray<Boolean>;
  Shifted: TYoloRotatedBox;
  I, J, Count, Index: Integer;
begin
  FiniteValue(ClassOffset);
  if not (Options.Layout in [yblChannelsFirst, yblEndToEndRows]) or (Options.ExtraCount <> 1) or
    Options.ApplyObjectnessToExtra or (ClassOffset <= 0) then
      raise EArgumentException.Create('Unsupported YOLO oriented prediction profile');
  Candidates := DecodeYoloCandidates(Data, BoxCount, FeatureCount, Options);
  SetLength(Boxes, Length(Candidates));
  SetLength(Gaussians, Length(Candidates));
  SetLength(Suppressed, Length(Candidates));
  for I := 0 to High(Candidates) do
  begin
    Index := Candidates[I].SourceIndex;
    if Options.Layout = yblChannelsFirst then
    begin
      Boxes[I].CenterX := Data[Index];
      Boxes[I].CenterY := Data[BoxCount + Index];
      Boxes[I].Width := Data[2 * BoxCount + Index];
      Boxes[I].Height := Data[3 * BoxCount + Index];
    end
    else begin
      Index := Index * FeatureCount;
      Boxes[I].CenterX := Data[Index];
      Boxes[I].CenterY := Data[Index + 1];
      Boxes[I].Width := Data[Index + 2];
      Boxes[I].Height := Data[Index + 3];
    end;
    Boxes[I].Angle := Candidates[I].Extra[0];
    if (Boxes[I].Width < 0) or (Boxes[I].Height < 0) then
      raise EArgumentException.Create('Negative YOLO oriented box size');
    if Options.Layout = yblEndToEndRows then Continue;
    Shifted := Boxes[I];
    { Keep the source Float32 class-offset rounding before the covariance test. }
    Shifted.CenterX := Shifted.CenterX + Candidates[I].ClassID * ClassOffset;
    Shifted.CenterY := Shifted.CenterY + Candidates[I].ClassID * ClassOffset;
    Gaussians[I] := Gaussian(Shifted);
  end;
  if Options.Layout = yblChannelsFirst then
    for I := 0 to High(Candidates) do
      for J := I + 1 to High(Candidates) do
        if not Suppressed[J] and (Candidates[I].ClassID = Candidates[J].ClassID) and
          (ProbabilityIoU(Gaussians[I], Gaussians[J]) > Options.NmsIoU) then
            Suppressed[J] := True;
  SetLength(Result, Min(Length(Candidates), Options.MaxDetections));
  Count := 0;
  for I := 0 to High(Candidates) do
  begin
    if Suppressed[I] then Continue;
    Result[Count].Box := RegularizeYoloRotatedBox(Boxes[I]);
    Result[Count].Score := Candidates[I].Score;
    Result[Count].ClassID := Candidates[I].ClassID;
    Result[Count].SourceIndex := Candidates[I].SourceIndex;
    Inc(Count);
    if Count = Length(Result) then Break;
  end;
  SetLength(Result, Count);
end;

end.
