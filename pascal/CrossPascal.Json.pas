{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Json                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Json;

{$IFDEF FPC}
{$MODE DELPHIUNICODE}
{$ENDIF}

interface

uses
  {$IFDEF FPC}FPJSON, SysUtils{$ELSE}System.JSON, System.SysUtils{$ENDIF};

type
  {$IFDEF FPC}
  TJSONValue = FPJSON.TJSONData;
  TJSONObject = FPJSON.TJSONObject;
  TJSONArray = FPJSON.TJSONArray;
  TJSONString = FPJSON.TJSONString;
  TJSONNumber = FPJSON.TJSONNumber;
  TJSONNull = FPJSON.TJSONNull;

  TCrossJSONObjectHelper = class helper for TJSONObject
    function GetValue(const Name: UnicodeString): TJSONValue;
    procedure AddPair(const Name, Value: UnicodeString); overload;
    procedure AddPair(const Name: UnicodeString; const Value: TJSONValue); overload;
    function ToJSON: UnicodeString;
  end;

  TCrossJSONArrayHelper = class helper for TJSONArray
    procedure AddElement(const Value: TJSONValue);
  end;
  {$ELSE}
  TJSONValue = System.JSON.TJSONValue;
  TJSONObject = System.JSON.TJSONObject;
  TJSONArray = System.JSON.TJSONArray;
  TJSONString = System.JSON.TJSONString;
  TJSONNumber = System.JSON.TJSONNumber;
  TJSONNull = System.JSON.TJSONNull;
  {$ENDIF}

function ParseCrossJSON(const Text: UnicodeString): TJSONValue;
function JSONValueText(const Value: TJSONValue): UnicodeString;
function JSONObjectName(const Obj: TJSONObject; const Index: Integer): UnicodeString;
function JSONObjectValue(const Obj: TJSONObject; const Index: Integer): TJSONValue;
function JSONIsBoolean(const Value: TJSONValue): Boolean;
function JSONIsTrue(const Value: TJSONValue): Boolean;
function JSONInteger(const Value: Int64): TJSONValue;
function JSONUnsigned(const Value: UInt64): TJSONValue;
function JSONFloat(const Value: Double): TJSONValue;
function JSONBoolean(const Value: Boolean): TJSONValue;
function JSONInvariantFormat: TFormatSettings;

implementation

{$IFDEF FPC}
uses
  JSONParser, JSONScanner;

function TCrossJSONObjectHelper.GetValue(const Name: UnicodeString): TJSONValue;
begin
  Result := Find(UTF8Encode(Name));
end;

procedure TCrossJSONObjectHelper.AddPair(const Name, Value: UnicodeString);
begin
  Add(UTF8Encode(Name), Value);
end;

procedure TCrossJSONObjectHelper.AddPair(const Name: UnicodeString; const Value: TJSONValue);
begin
  Add(UTF8Encode(Name), Value);
end;

function TCrossJSONObjectHelper.ToJSON: UnicodeString;
begin
  Result := UTF8Decode(AsJSON);
end;

procedure TCrossJSONArrayHelper.AddElement(const Value: TJSONValue);
begin
  Add(Value);
end;
{$ENDIF}

function ParseCrossJSON(const Text: UnicodeString): TJSONValue;
{$IFDEF FPC}
var
  Parser: TJSONParser;
{$ENDIF}
begin
  {$IFDEF FPC}
  Parser := TJSONParser.Create(UTF8Encode(Text), [joUTF8, joStrict]);
  try
    Result := Parser.Parse;
  finally
    Parser.Free;
  end;
  {$ELSE}
  Result := TJSONObject.ParseJSONValue(Text);
  {$ENDIF}
end;

function JSONValueText(const Value: TJSONValue): UnicodeString;
begin
  {$IFDEF FPC}
  if Value is TJSONNumber then
    Result := UTF8Decode(Value.AsJSON)
  else
    Result := UTF8Decode(Value.AsString);
  {$ELSE}
  Result := Value.Value;
  {$ENDIF}
end;

function JSONObjectName(const Obj: TJSONObject; const Index: Integer): UnicodeString;
begin
  {$IFDEF FPC}
  Result := UTF8Decode(Obj.Names[Index]);
  {$ELSE}
  Result := Obj.Pairs[Index].JsonString.Value;
  {$ENDIF}
end;

function JSONObjectValue(const Obj: TJSONObject; const Index: Integer): TJSONValue;
begin
  {$IFDEF FPC}
  Result := Obj.Items[Index];
  {$ELSE}
  Result := Obj.Pairs[Index].JsonValue;
  {$ENDIF}
end;

function JSONIsBoolean(const Value: TJSONValue): Boolean;
begin
  {$IFDEF FPC}
  Result := Value is TJSONBoolean;
  {$ELSE}
  Result := (Value is TJSONBool) or (Value is TJSONTrue) or (Value is TJSONFalse);
  {$ENDIF}
end;

function JSONIsTrue(const Value: TJSONValue): Boolean;
begin
  {$IFDEF FPC}
  Result := (Value is TJSONBoolean) and Value.AsBoolean;
  {$ELSE}
  Result := (Value is TJSONTrue) or ((Value is TJSONBool) and (Value.Value = 'true'));
  {$ENDIF}
end;

function JSONInteger(const Value: Int64): TJSONValue;
begin
  {$IFDEF FPC}
  Result := TJSONInt64Number.Create(Value);
  {$ELSE}
  Result := TJSONNumber.Create(Value);
  {$ENDIF}
end;

function JSONUnsigned(const Value: UInt64): TJSONValue;
begin
  {$IFDEF FPC}
  Result := TJSONQWordNumber.Create(Value);
  {$ELSE}
  Result := TJSONNumber.Create(Value);
  {$ENDIF}
end;

function JSONFloat(const Value: Double): TJSONValue;
begin
  {$IFDEF FPC}
  Result := TJSONFloatNumber.Create(Value);
  {$ELSE}
  Result := TJSONNumber.Create(Value);
  {$ENDIF}
end;

function JSONBoolean(const Value: Boolean): TJSONValue;
begin
  {$IFDEF FPC}
  Result := TJSONBoolean.Create(Value);
  {$ELSE}
  Result := TJSONBool.Create(Value);
  {$ENDIF}
end;

function JSONInvariantFormat: TFormatSettings;
begin
  {$IFDEF FPC}
  Result := DefaultFormatSettings;
  Result.DecimalSeparator := '.';
  Result.ThousandSeparator := #0;
  {$ELSE}
  Result := TFormatSettings.Invariant;
  {$ENDIF}
end;

end.
