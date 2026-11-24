{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Detections                                                        }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Detections;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}Types{$ELSE}System.Types{$ENDIF};

type
  TYoloBoxLayout = (yblChannelsFirst, yblObjectnessRows, yblEndToEndRows);

  TYoloDetectionOptions = record
    Layout: TYoloBoxLayout;
    ClassCount: Integer;
    ExtraCount: Integer;
    Confidence: Single;
    NmsIoU: Single;
    MaxDetections: Integer;
    MaxNmsCandidates: Integer;
    ApplyObjectnessToExtra: Boolean;
  end;

  TYoloRawDetection = record
    Box: TRectF;
    Score: Single;
    ClassID: Integer;
    SourceIndex: Integer;
    Extra: TArray<Single>;
  end;
  TYoloRawDetections = TArray<TYoloRawDetection>;

{ Batch one, Float32, contiguous data. Boxes remain in model-input coordinates.
  Extra columns follow the profile's objectness rule (classic v5 scales masks).
  Layout and class/extra counts come from the selected model's metadata. }
function DecodeYoloCandidates(const Data: TArray<Single>; const BoxCount, FeatureCount: Integer;
  const Options: TYoloDetectionOptions): TYoloRawDetections;
function DecodeYoloDetections(const Data: TArray<Single>; const BoxCount, FeatureCount: Integer;
  const Options: TYoloDetectionOptions): TYoloRawDetections;

implementation

uses
  {$IFDEF FPC}Math, SysUtils{$ELSE}System.Math, System.SysUtils{$ENDIF};

procedure RequireFinite(const Value: Single);
begin
  if IsNan(Value) or IsInfinite(Value) then
    raise EArgumentException.Create('Nonfinite YOLO prediction value');
end;

procedure ValidateOptions(const Options: TYoloDetectionOptions);
begin
  RequireFinite(Options.Confidence);
  RequireFinite(Options.NmsIoU);
  if not (Options.Layout in [yblChannelsFirst, yblObjectnessRows, yblEndToEndRows]) or
    (Options.ClassCount <= 0) or (Options.ExtraCount < 0) or
    (Options.Confidence < 0) or (Options.Confidence > 1) or
    (Options.NmsIoU < 0) or (Options.NmsIoU > 1) or
    (Options.ApplyObjectnessToExtra and (Options.Layout <> yblObjectnessRows)) or
    (Options.MaxDetections <= 0) or (Options.MaxNmsCandidates < Options.MaxDetections) then
      raise EArgumentException.Create('Invalid YOLO detection options');
end;

procedure SortCandidates(var Values: TYoloRawDetections);
var
  Work: TYoloRawDetections;

  procedure Merge(const First, Finish: Integer);
  var
    Middle, Left, Right, At: Integer;
    TakeLeft: Boolean;
  begin
    if Finish - First <= 1 then Exit;
    Middle := First + (Finish - First) div 2;
    Merge(First, Middle);
    Merge(Middle, Finish);
    Left := First;
    Right := Middle;
    for At := First to Finish - 1 do
    begin
      TakeLeft := Right >= Finish;
      if (Left < Middle) and (Right < Finish) then
        TakeLeft := (Values[Left].Score > Values[Right].Score) or
          ((Values[Left].Score = Values[Right].Score) and
          (Values[Left].SourceIndex < Values[Right].SourceIndex));
      if (Left < Middle) and TakeLeft then
      begin
        Work[At] := Values[Left];
        Inc(Left);
      end
      else begin
        Work[At] := Values[Right];
        Inc(Right);
      end;
    end;
    for At := First to Finish - 1 do Values[At] := Work[At];
  end;

begin
  SetLength(Work, Length(Values));
  Merge(0, Length(Values));
end;

function DecodeYoloCandidates(const Data: TArray<Single>; const BoxCount, FeatureCount: Integer;
  const Options: TYoloDetectionOptions): TYoloRawDetections;
var
  I, C, Count, FirstClass, ExtraStart, BestClass: Integer;
  RequiredFeatures: Int64;
  Objectness, BestScore, Score, X, Y, Width, Height: Single;
  Candidate: TYoloRawDetection;

  function Feature(const Row, Column: Integer): Single;
  begin
    if Options.Layout = yblChannelsFirst then
      Result := Data[Column * BoxCount + Row]
    else
      Result := Data[Row * FeatureCount + Column];
    RequireFinite(Result);
  end;

begin
  ValidateOptions(Options);
  if Options.Layout = yblEndToEndRows then
    RequiredFeatures := 6 + Int64(Options.ExtraCount)
  else
    RequiredFeatures := 4 + Int64(Ord(Options.Layout = yblObjectnessRows)) +
      Options.ClassCount + Options.ExtraCount;
  if (BoxCount < 0) or (FeatureCount <> RequiredFeatures) or (Length(Data) > High(Integer)) or
    (Int64(BoxCount) * FeatureCount <> Length(Data)) then
      raise EArgumentException.Create('YOLO prediction shape differs from its profile');
  SetLength(Result, BoxCount);
  Count := 0;
  for I := 0 to BoxCount - 1 do
  begin
    if Options.Layout = yblEndToEndRows then
    begin
      BestScore := Feature(I, 4);
      if BestScore <= Options.Confidence then Continue;
      Score := Feature(I, 5);
      if (Score < 0) or (Score >= Options.ClassCount) or (Trunc(Score) <> Score) then
        raise EArgumentException.Create('Invalid YOLO end-to-end class index');
      BestClass := Trunc(Score);
      Candidate.Box.Left := Feature(I, 0);
      Candidate.Box.Top := Feature(I, 1);
      Candidate.Box.Right := Feature(I, 2);
      Candidate.Box.Bottom := Feature(I, 3);
      ExtraStart := 6;
    end
    else begin
      Objectness := 1;
      FirstClass := 4;
      if Options.Layout = yblObjectnessRows then
      begin
        Objectness := Feature(I, 4);
        if Objectness <= Options.Confidence then Continue;
        FirstClass := 5;
      end;
      BestScore := -Infinity;
      BestClass := 0;
      for C := 0 to Options.ClassCount - 1 do
      begin
        Score := Feature(I, FirstClass + C) * Objectness;
        RequireFinite(Score);
        if Score > BestScore then
        begin
          BestScore := Score;
          BestClass := C;
        end;
      end;
      if BestScore <= Options.Confidence then Continue;
      X := Feature(I, 0);
      Y := Feature(I, 1);
      Width := Feature(I, 2);
      Height := Feature(I, 3);
      Candidate.Box.Left := X - Width * 0.5;
      Candidate.Box.Top := Y - Height * 0.5;
      Candidate.Box.Right := X + Width * 0.5;
      Candidate.Box.Bottom := Y + Height * 0.5;
      ExtraStart := FirstClass + Options.ClassCount;
    end;
    RequireFinite(Candidate.Box.Left);
    RequireFinite(Candidate.Box.Top);
    RequireFinite(Candidate.Box.Right);
    RequireFinite(Candidate.Box.Bottom);
    Candidate.Score := BestScore;
    Candidate.ClassID := BestClass;
    Candidate.SourceIndex := I;
    SetLength(Candidate.Extra, Options.ExtraCount);
    for C := 0 to Options.ExtraCount - 1 do
    begin
      Candidate.Extra[C] := Feature(I, ExtraStart + C);
      if Options.ApplyObjectnessToExtra then
        Candidate.Extra[C] := Candidate.Extra[C] * Objectness;
      RequireFinite(Candidate.Extra[C]);
    end;
    Result[Count] := Candidate;
    { Each result owns a distinct extra array; do not mutate a previous row. }
    Candidate.Extra := nil;
    Inc(Count);
  end;
  SetLength(Result, Count);
  if Options.Layout <> yblEndToEndRows then SortCandidates(Result);
  if Length(Result) > Options.MaxNmsCandidates then SetLength(Result, Options.MaxNmsCandidates);
end;

function PositiveDifference(const A, B: Double): Double; inline;
begin
  if A > B then
    Result := A - B
  else
    Result := 0;
end;

function BoxIoU(const A, B: TRectF): Double;
var
  Width, Height, Intersection, AreaA, AreaB, Union: Double;
begin
  Width := PositiveDifference(Min(A.Right, B.Right), Max(A.Left, B.Left));
  Height := PositiveDifference(Min(A.Bottom, B.Bottom), Max(A.Top, B.Top));
  Intersection := Width * Height;
  AreaA := PositiveDifference(A.Right, A.Left) * PositiveDifference(A.Bottom, A.Top);
  AreaB := PositiveDifference(B.Right, B.Left) * PositiveDifference(B.Bottom, B.Top);
  Union := AreaA + AreaB - Intersection;
  if Union > 0 then
    Result := Intersection / Union
  else
    Result := 0;
end;

function DecodeYoloDetections(const Data: TArray<Single>; const BoxCount, FeatureCount: Integer;
  const Options: TYoloDetectionOptions): TYoloRawDetections;
var
  Candidates: TYoloRawDetections;
  Suppressed: TArray<Boolean>;
  I, J, Count: Integer;
begin
  Candidates := DecodeYoloCandidates(Data, BoxCount, FeatureCount, Options);
  SetLength(Result, Min(Length(Candidates), Options.MaxDetections));
  SetLength(Suppressed, Length(Candidates));
  Count := 0;
  for I := 0 to High(Candidates) do
  begin
    if Suppressed[I] then Continue;
    Result[Count] := Candidates[I];
    Inc(Count);
    if Count = Length(Result) then Break;
    if Options.Layout = yblEndToEndRows then Continue;
    for J := I + 1 to High(Candidates) do
      if not Suppressed[J] and (Candidates[I].ClassID = Candidates[J].ClassID) and
        (BoxIoU(Candidates[I].Box, Candidates[J].Box) > Options.NmsIoU) then
          Suppressed[J] := True;
  end;
  SetLength(Result, Count);
end;

end.
