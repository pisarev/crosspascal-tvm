{ ************************************************************************** }
{                                                                            }
{ LanguageReport                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit LanguageReport;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

function GenerationSummary(const Report: string; const ContextTokens: Integer = 0): string;

implementation

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.Json;

function CountField(const Obj: TJSONObject; const Name, IDs: string): Int64;
var
  Value: TJSONValue;
begin
  Result := -1;
  Value := Obj.GetValue(Name);
  if (Value <> nil) and TryStrToInt64(JSONValueText(Value), Result) and (Result >= 0) then Exit;
  Result := -1;
  Value := Obj.GetValue(IDs);
  if Value is TJSONArray then Result := TJSONArray(Value).Count;
end;

function GenerationSummary(const Report: string; const ContextTokens: Integer): string;
var
  Value, StopValue: TJSONValue;
  Obj: TJSONObject;
  Reason, Counts: string;
  PromptCount, OutputCount: Int64;
begin
  Result := 'Generation finished.';
  if Trim(Report) = '' then Exit;
  Value := nil;
  try
    try
      Value := ParseCrossJSON(Report);
      if not (Value is TJSONObject) then
      begin
        Result := 'Generation returned an invalid report.';
        Exit;
      end;
      Obj := TJSONObject(Value);
      StopValue := Obj.GetValue('stop');
      if StopValue = nil then StopValue := Obj.GetValue('stop_reason');
      Reason := LowerCase(JSONValueText(StopValue));
      if (Reason = 'eos') or (Reason = 'end_of_turn') then
        Result := 'Model ended its answer.'
      else if Reason = 'context_limit' then
        Result := 'Context limit reached. The answer may be incomplete.'
      else if (Reason = 'manual_limit') or (Reason = 'max_new_tokens') then
        Result := 'New token limit reached. The answer may be incomplete.'
      else if Reason = 'effort_limit' then
        Result := 'Answer budget reached. The answer may be incomplete.'
      else if (Reason = 'user_stop') or (Reason = 'cancelled') then
        Result := 'Generation stopped by the user.'
      else if Reason <> '' then
        Result := 'Generation stopped: ' + Reason + '.';
      PromptCount := CountField(Obj, 'prompt_tokens', 'prompt_token_ids');
      OutputCount := CountField(Obj, 'generated_tokens', 'generated_token_ids');
      if OutputCount < 0 then OutputCount := CountField(Obj, 'generated_tokens', 'token_ids');
      Counts := '';
      if PromptCount >= 0 then Counts := 'Input ' + IntToStr(PromptCount);
      if OutputCount >= 0 then
      begin
        if Counts <> '' then Counts := Counts + ', ';
        Counts := Counts + 'output ' + IntToStr(OutputCount);
      end;
      if ContextTokens > 0 then
      begin
        if Counts <> '' then Counts := Counts + ', ';
        Counts := Counts + 'context ' + IntToStr(ContextTokens);
      end;
      if Counts <> '' then Result := Result + ' ' + Counts + ' tokens.';
    except
      on E: Exception do Result := 'Generation returned an invalid report.';
    end;
  finally
    Value.Free;
  end;
end;

end.
