{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Classification                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Classification;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

type
  TYoloClassScoreKind = (ycsProbabilities, ycsLogits);
  TYoloClassification = record
    ClassID: Integer;
    Score: Single;
  end;
  TYoloClassifications = TArray<TYoloClassification>;

function DecodeYoloClassification(const Scores: TArray<Single>; const Kind: TYoloClassScoreKind;
  const TopK: Integer): TYoloClassifications;

implementation

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF};

function DecodeYoloClassification(const Scores: TArray<Single>; const Kind: TYoloClassScoreKind;
  const TopK: Integer): TYoloClassifications;
var
  Probabilities: TArray<Single>;
  Exponentials: TArray<Double>;
  Used: TArray<Boolean>;
  I, K, Best, Count: Integer;
  Maximum: Single;
  Value, Total: Double;
begin
  if (Length(Scores) = 0) or (Length(Scores) > High(Integer)) or (TopK <= 0) or
    not (Kind in [ycsProbabilities, ycsLogits]) then
      raise EArgumentException.Create('Invalid YOLO classification profile');
  Maximum := -Infinity;
  for I := 0 to High(Scores) do
  begin
    if IsNan(Scores[I]) or IsInfinite(Scores[I]) then
      raise EArgumentException.Create('Nonfinite YOLO classification score');
    if (Kind = ycsProbabilities) and ((Scores[I] < 0) or (Scores[I] > 1)) then
      raise EArgumentException.Create('YOLO probability lies outside zero to one');
    Maximum := Max(Maximum, Scores[I]);
  end;
  if Kind = ycsProbabilities then
    Probabilities := Copy(Scores)
  else begin
    SetLength(Exponentials, Length(Scores));
    Total := 0;
    for I := 0 to High(Scores) do
    begin
      Value := Scores[I];
      Value := Value - Maximum;
      Exponentials[I] := Exp(Value);
      Total := Total + Exponentials[I];
    end;
    SetLength(Probabilities, Length(Scores));
    for I := 0 to High(Scores) do Probabilities[I] := Exponentials[I] / Total;
  end;
  Count := Min(TopK, Length(Scores));
  SetLength(Result, Count);
  SetLength(Used, Length(Scores));
  for K := 0 to Count - 1 do
  begin
    Best := -1;
    for I := 0 to High(Scores) do
      if not Used[I] and ((Best < 0) or (Probabilities[I] > Probabilities[Best])) then Best := I;
    Result[K].ClassID := Best;
    Result[K].Score := Probabilities[Best];
    Used[Best] := True;
  end;
end;

end.
