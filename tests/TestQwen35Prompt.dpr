{ ************************************************************************** }
{                                                                            }
{ TestQwen35Prompt                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

program TestQwen35Prompt;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$APPTYPE CONSOLE}

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF},
  Qwen35Prompt, Qwen35Native;

procedure Check(const Value: Boolean; const MessageText: string);
begin
  if not Value then raise Exception.Create(MessageText);
end;

function ExamplePrompt: TQwen35PreparedPrompt;
var
  Positions: TArray<TQwen35RopePosition>;
  Features: TArray<Single>;
  Row, Column, I: Integer;
begin
  Result := Default(TQwen35PreparedPrompt);
  Result.AppendText([2, 3]);
  SetLength(Positions, 6);
  for Row := 0 to 1 do
    for Column := 0 to 2 do
    begin
      I := Row * 3 + Column;
      Positions[I][0] := 2;
      Positions[I][1] := 2 + Row;
      Positions[I][2] := 2 + Column;
    end;
  SetLength(Features, 24);
  for I := 0 to High(Features) do Features[I] := (I + 1) / 32;
  Result.AppendVisualRows(7, Positions, Features, 4);
  Result.AppendText([9]);
end;

procedure InvalidPrompt(const CaseNumber: Integer);
var
  Prompt: TQwen35PreparedPrompt;
begin
  Prompt := ExamplePrompt;
  case CaseNumber of
    0: Prompt.Rows[2].Visual := 4;
    1: Prompt.VisualCount := 5;
    2: Prompt.Rows[0].Token := 16;
    3: Prompt.NextRope := 9;
    4: Prompt.Embeddings[0] := NaN;
    5: Prompt.Rows[3].Rope[1] := -1;
    6:
      begin
        Prompt.NextRope := High(Int64);
        Prompt.AppendText([1]);
      end;
  end;
  ValidateQwen35Prompt(Prompt, 4, 16, 16);
end;

procedure Run;
var
  Prompt: TQwen35PreparedPrompt;
  I, J: Integer;
  Rejected: Boolean;
begin
  Prompt := Default(TQwen35PreparedPrompt);
  Prompt.AppendText([2, 3, 9]);
  ValidateQwen35Prompt(Prompt, 4, 16, 16);
  Check((Prompt.NextRope = 3) and (Qwen35RopeDelta(Prompt) = 0), 'Text rotary delta');
  for I := 0 to High(Prompt.Rows) do
  begin
    Check(Prompt.Rows[I].Visual = -1, 'Text embedding selection');
    for J := 0 to 2 do Check(Prompt.Rows[I].Rope[J] = I, 'Text position');
  end;
  Prompt := ExamplePrompt;
  ValidateQwen35Prompt(Prompt, 4, 16, 16);
  Check((Length(Prompt.Rows) = 9) and (Prompt.VisualCount = 6), 'Merged grid row count');
  Check((Prompt.NextRope = 6) and (Qwen35RopeDelta(Prompt) = -3), 'Merged grid rotary delta');
  for I := 0 to 5 do
  begin
    Check(Prompt.Rows[I + 2].Visual = I, 'Visual embedding row order');
    Check((Prompt.Rows[I + 2].Rope[0] = 2) and
      (Prompt.Rows[I + 2].Rope[1] = 2 + I div 3) and
      (Prompt.Rows[I + 2].Rope[2] = 2 + I mod 3), 'Merged grid axis order');
  end;
  for J := 0 to 2 do Check(Prompt.Rows[8].Rope[J] = 5, 'Text after the image');
  for I := 0 to 6 do
  begin
    Rejected := False;
    try
      InvalidPrompt(I);
    except
      on E: Exception do Rejected := True;
    end;
    Check(Rejected, 'Invalid prepared prompt accepted: ' + IntToStr(I));
  end;
  Rejected := False;
  try
    ValidateQwen35Prompt(Prompt, 4, 16, 8);
  except
    on E: Exception do Rejected := True;
  end;
  Check(Rejected, 'Compiled context limit');
  WriteLn('{"status":"PASS","scenarios":10,"language_model_inference":false}');
end;

begin
  try
    Run;
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
