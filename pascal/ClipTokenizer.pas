{ ************************************************************************** }
{                                                                            }
{ ClipTokenizer                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit ClipTokenizer;

{$IFDEF FPC}{$MODE DELPHI}{$ENDIF}

interface

uses
  CrossPascal.NativeIO, {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, BPETokenizer;

const
  ClipVocabularySize = 49408;
  ClipSequenceLength = 77;
  ClipStartToken = 49406;
  ClipEndToken = 49407;

type
  TClipSequence = record
    Ids: array[0..ClipSequenceLength - 1] of Int64;
    AttentionMask: array[0..ClipSequenceLength - 1] of Int64;
    TokenCount: Integer;
    Truncated: Boolean;
  end;

  TClipTokenizer = class(TCustomBPETokenizer)
  private
    FTokens: TArray<UnicodeString>;
    FSingle: array[0..255] of Integer;
    FFinal: array[0..255] of Integer;
    FByteCode: array[0..255] of WideChar;
    FCodeByte: array[0..511] of Integer;
    FLoaded: Boolean;
    procedure Clear;
    procedure AppendWord(const Text: UnicodeString; var Ids: TArray<Integer>; var Count: Integer);
    procedure AppendNormalized(const Text: UnicodeString; var Ids: TArray<Integer>; var Count: Integer);
  public
    function Load(const Directory: UnicodeString; const Error: PString): Boolean; override;
    function Loaded: Boolean; override;
    function TokenCount: Integer; override;
    function Encode(const Text: UnicodeString): TArray<Integer>; overload; override;
    function EncodeSequence(const Text: UnicodeString; const Truncate: Boolean = False): TClipSequence;
    function Decode(const Ids: array of Integer): UnicodeString; override;
    function SpecialId(const Text: UnicodeString): Integer; override;
  end;

implementation

uses
  {$IFDEF FPC}Classes, Character, Generics.Collections{$ELSE}System.Classes, System.Character, System.Generics.Collections{$ENDIF},
  CrossPascal.Json {$IFDEF MSWINDOWS}, {$IFDEF FPC}Windows{$ELSE}Winapi.Windows{$ENDIF}{$ENDIF};

const
  ClipMergeCount = 48894;
  ClipMaximumTextLength = 4096;
  ClipMaximumPieceBytes = 4096;
  ClipMaximumSourceBytes = 4 * 1024 * 1024;
  ClipStartText: UnicodeString = '<|startoftext|>';
  ClipEndText: UnicodeString = '<|endoftext|>';
  ClipWordSuffix: UnicodeString = '</w>';

{$IFDEF MSWINDOWS}
function NormalizeString(NormForm: Integer; Source: PWideChar; SourceLength: Integer; Destination: PWideChar;
  DestinationLength: Integer): Integer; stdcall; external 'Normaliz.dll' name 'NormalizeString';
{$ENDIF}

function NormalizeClipText(const Text: UnicodeString): UnicodeString;
{$IFDEF MSWINDOWS}
const
  NormalizationC = 1;
var
  Normalized: UnicodeString;
  Needed, At, I, Count, Taken: Integer;
  PreviousSpace: Boolean;
begin
  if Text = '' then Exit('');
  Needed := NormalizeString(NormalizationC, PWideChar(Text), Length(Text), nil, 0);
  if Needed <= 0 then
    raise Exception.CreateFmt('CLIP NFC normalization failed: %d', [GetLastError]);
  SetLength(Normalized, Needed);
  Needed := NormalizeString(NormalizationC, PWideChar(Text), Length(Text), PWideChar(Normalized), Needed);
  if Needed <= 0 then
    raise Exception.CreateFmt('CLIP NFC normalization failed: %d', [GetLastError]);
  SetLength(Normalized, Needed);
  SetLength(Result, Needed * 2);
  Count := 0;
  PreviousSpace := False;
  At := 1;
  while At <= Needed do
  begin
    Taken := 1;
    if TCharacter.IsHighSurrogate(Normalized[At]) then Taken := 2;
    if TCharacter.IsWhiteSpace(Normalized, At) then
    begin
      if not PreviousSpace then
      begin
        Inc(Count);
        Result[Count] := ' ';
      end;
      PreviousSpace := True;
    end
    else if Normalized[At] = WideChar($0130) then
    begin
      Inc(Count);
      Result[Count] := 'i';
      Inc(Count);
      Result[Count] := WideChar($0307);
      PreviousSpace := False;
    end
    else begin
      for I := 0 to Taken - 1 do
      begin
        Inc(Count);
        Result[Count] := Normalized[At + I];
      end;
      PreviousSpace := False;
    end;
    Inc(At, Taken);
  end;
  SetLength(Result, Count);
  Result := TCharacter.ToLower(Result);
end;
{$ELSE}
begin
  raise Exception.Create('CLIP NFC normalization requires Windows');
end;
{$ENDIF}

function ReadClipSource(const FileName: UnicodeString): UnicodeString;
var
  Stream: TStream;
  Bytes: UTF8String;
begin
  Stream := TNativeReadStream.Create(FileName);
  try
    if (Stream.Size <= 0) or (Stream.Size > ClipMaximumSourceBytes) then
      raise Exception.Create('Invalid CLIP tokenizer file size: ' + FileName);
    SetLength(Bytes, Stream.Size);
    Stream.ReadBuffer(Bytes[1], Length(Bytes));
    if Copy(Bytes, 1, 3) = #$EF#$BB#$BF then Delete(Bytes, 1, 3);
    Result := UTF8Decode(Bytes);
  finally
    Stream.Free;
  end;
end;

function ReadClipLine(const Text: UnicodeString; var At: Integer): UnicodeString;
var
  Start: Integer;
begin
  Start := At;
  while (At <= Length(Text)) and (Text[At] <> #10) do Inc(At);
  Result := Copy(Text, Start, At - Start);
  if (Result <> '') and (Result[Length(Result)] = #13) then
    SetLength(Result, Length(Result) - 1);
  Inc(At);
end;

procedure ValidateClipText(const Text: UnicodeString);
var
  At: Integer;
begin
  At := 1;
  while At <= Length(Text) do
    if TCharacter.IsHighSurrogate(Text[At]) then
    begin
      if (At = Length(Text)) or not TCharacter.IsLowSurrogate(Text[At + 1]) then
        raise Exception.Create('CLIP text has an unpaired UTF-16 high surrogate');
      Inc(At, 2);
    end
    else begin
      if TCharacter.IsLowSurrogate(Text[At]) then
        raise Exception.Create('CLIP text has an unpaired UTF-16 low surrogate');
      Inc(At);
    end;
end;

procedure AppendClipId(const Id: Integer; var Ids: TArray<Integer>; var Count: Integer);
begin
  if Count = Length(Ids) then SetLength(Ids, Count * 2 + 16);
  Ids[Count] := Id;
  Inc(Count);
end;

function ClipCodeLength(const Text: UnicodeString; const At: Integer): Integer;
begin
  if (At < Length(Text)) and TCharacter.IsHighSurrogate(Text[At]) and
    TCharacter.IsLowSurrogate(Text[At + 1]) then
      Exit(2);
  Result := 1;
end;

function ClipTextAt(const Text, Value: UnicodeString; const At: Integer): Boolean;
begin
  Result := Copy(Text, At, Length(Value)) = Value;
end;

function ClipSpanLength(const Text: UnicodeString; const At: Integer): Integer;
const
  Contractions: array[0..6] of UnicodeString = ('''s', '''t', '''re', '''ve', '''m', '''ll', '''d');
var
  I, Next: Integer;
begin
  if Text[At] = '''' then
    for I := Low(Contractions) to High(Contractions) do
      if ClipTextAt(Text, Contractions[I], At) then Exit(Length(Contractions[I]));
  Next := At + ClipCodeLength(Text, At);
  if TCharacter.IsLetter(Text, At) then
    while (Next <= Length(Text)) and TCharacter.IsLetter(Text, Next) do
      Inc(Next, ClipCodeLength(Text, Next))
  else if not TCharacter.IsNumber(Text, At) then
    while (Next <= Length(Text)) and not TCharacter.IsWhiteSpace(Text, Next) and
      not TCharacter.IsLetter(Text, Next) and not TCharacter.IsNumber(Text, Next) do
        Inc(Next, ClipCodeLength(Text, Next));
  Result := Next - At;
end;

procedure TClipTokenizer.Clear;
var
  I: Integer;
begin
  FLoaded := False;
  ClearMerges;
  SetLength(FTokens, 0);
  for I := 0 to 255 do
  begin
    FSingle[I] := -1;
    FFinal[I] := -1;
  end;
end;

function TClipTokenizer.Load(const Directory: UnicodeString; const Error: PString): Boolean;
var
  Root: TJSONValue;
  Vocab: TJSONObject;
  Names: TDictionary<UnicodeString, Integer>;
  Used: TArray<Boolean>;
  Dir: UnicodeString;
  Name, A, B, Merges: UnicodeString;
  I, Id, Next, Split, Left, Right, Joined, At: Integer;
begin
  Clear;
  if Assigned(Error) then Error^ := '';
  Dir := IncludeTrailingPathDelimiter(Directory);
  try
    Root := ParseCrossJSON(ReadClipSource(Dir + 'vocab.json'));
    try
      if not (Root is TJSONObject) then raise Exception.Create('Invalid CLIP vocabulary object');
      Vocab := TJSONObject(Root);
      if Vocab.Count <> ClipVocabularySize then
        raise Exception.CreateFmt('Expected %d CLIP vocabulary entries', [ClipVocabularySize]);
      SetLength(FTokens, ClipVocabularySize);
      SetLength(Used, ClipVocabularySize);
      Names := TDictionary<UnicodeString, Integer>.Create;
      try
        for I := 0 to Vocab.Count - 1 do
        begin
          Name := JSONObjectName(Vocab, I);
          if not (JSONObjectValue(Vocab, I) is TJSONNumber) or
            not TryStrToInt(JSONValueText(JSONObjectValue(Vocab, I)), Id) or
            (Id < 0) or (Id >= ClipVocabularySize) then
              raise Exception.Create('Invalid CLIP vocabulary token ID');
          if Used[Id] or (Name = '') then
            raise Exception.Create('Duplicate or empty CLIP vocabulary entry');
          Used[Id] := True;
          FTokens[Id] := Name;
          Names.Add(Name, Id);
        end;
        if (FTokens[ClipStartToken] <> ClipStartText) or (FTokens[ClipEndToken] <> ClipEndText) then
          raise Exception.Create('CLIP special token IDs differ');
        for I := Low(FCodeByte) to High(FCodeByte) do FCodeByte[I] := -1;
        Next := 256;
        for I := 0 to 255 do
        begin
          if ((I >= 33) and (I <= 126)) or ((I >= 161) and (I <= 172)) or (I >= 174) then
            FByteCode[I] := WideChar(I)
          else begin
            FByteCode[I] := WideChar(Next);
            Inc(Next);
          end;
          FCodeByte[Ord(FByteCode[I])] := I;
          if not Names.TryGetValue(UnicodeString(FByteCode[I]), FSingle[I]) or
            not Names.TryGetValue(UnicodeString(FByteCode[I]) + ClipWordSuffix, FFinal[I]) then
              raise Exception.CreateFmt('CLIP vocabulary has no byte token for %d', [I]);
        end;
        Merges := ReadClipSource(Dir + 'merges.txt');
        At := 1;
        if ReadClipLine(Merges, At) <> '#version: 0.2' then
          raise Exception.Create('CLIP merge version differs');
        ReserveMerges(ClipMergeCount);
        for I := 0 to ClipMergeCount - 1 do
        begin
          Name := ReadClipLine(Merges, At);
          Split := Pos(' ', Name);
          if (Split <= 1) or (Split = Length(Name)) then
            raise Exception.CreateFmt('Invalid CLIP merge at rank %d', [I]);
          A := Copy(Name, 1, Split - 1);
          B := Copy(Name, Split + 1, Length(Name));
          if not Names.TryGetValue(A, Left) or not Names.TryGetValue(B, Right) or
            not Names.TryGetValue(A + B, Joined) then
              raise Exception.CreateFmt('Unknown CLIP merge token at rank %d', [I]);
          Put(Left, Right, Joined, I);
        end;
        if At <= Length(Merges) then raise Exception.Create('CLIP merge count differs');
      finally
        Names.Free;
      end;
    finally
      Root.Free;
    end;
    FLoaded := True;
  except
    on E: Exception do
    begin
      Clear;
      if Assigned(Error) then Error^ := E.Message;
      Exit(False);
    end;
  end;
  Result := True;
end;

function TClipTokenizer.Loaded: Boolean;
begin
  Result := FLoaded;
end;

function TClipTokenizer.TokenCount: Integer;
begin
  Result := Length(FTokens);
end;

procedure TClipTokenizer.AppendWord(const Text: UnicodeString; var Ids: TArray<Integer>;
  var Count: Integer);
var
  Bytes: UTF8String;
  Work: TArray<Integer>;
  I, Taken: Integer;
begin
  Bytes := UTF8Encode(Text);
  Taken := Length(Bytes);
  if Taken = 0 then Exit;
  if Taken > ClipMaximumPieceBytes then
    raise Exception.CreateFmt('CLIP text piece exceeds %d UTF-8 bytes', [ClipMaximumPieceBytes]);
  SetLength(Work, Taken);
  for I := 0 to Taken - 2 do Work[I] := FSingle[Byte(Bytes[I + 1])];
  Work[Taken - 1] := FFinal[Byte(Bytes[Taken])];
  Taken := MergeBPE(Work, Taken);
  for I := 0 to Taken - 1 do AppendClipId(Work[I], Ids, Count);
end;

procedure TClipTokenizer.AppendNormalized(const Text: UnicodeString; var Ids: TArray<Integer>;
  var Count: Integer);
var
  Normalized, Part: UnicodeString;
  At, Next, Taken, Start: Integer;
begin
  Normalized := NormalizeClipText(Text);
  Start := 1;
  while Start <= Length(Normalized) do
  begin
    Next := Start;
    while (Next <= Length(Normalized)) and not ClipTextAt(Normalized, ClipStartText, Next) do
      Inc(Next);
    Part := Copy(Normalized, Start, Next - Start);
    At := 1;
    while At <= Length(Part) do
      if TCharacter.IsWhiteSpace(Part, At) then
        Inc(At, ClipCodeLength(Part, At))
      else begin
        Taken := ClipSpanLength(Part, At);
        AppendWord(Copy(Part, At, Taken), Ids, Count);
        Inc(At, Taken);
      end;
    if Next > Length(Normalized) then Break;
    AppendClipId(ClipStartToken, Ids, Count);
    Start := Next + Length(ClipStartText);
  end;
end;

function TClipTokenizer.Encode(const Text: UnicodeString): TArray<Integer>;
var
  At, Next, Count: Integer;
begin
  RequireLoaded;
  Result := nil;
  if Length(Text) > ClipMaximumTextLength then
    raise Exception.CreateFmt('CLIP text exceeds %d UTF-16 characters', [ClipMaximumTextLength]);
  ValidateClipText(Text);
  SetLength(Result, Length(Text) * 2 + 4);
  Count := 0;
  AppendClipId(ClipStartToken, Result, Count);
  At := 1;
  while At <= Length(Text) do
  begin
    Next := At;
    while (Next <= Length(Text)) and not ClipTextAt(Text, ClipEndText, Next) do Inc(Next);
    AppendNormalized(Copy(Text, At, Next - At), Result, Count);
    if Next > Length(Text) then Break;
    AppendClipId(ClipEndToken, Result, Count);
    At := Next + Length(ClipEndText);
  end;
  AppendClipId(ClipEndToken, Result, Count);
  SetLength(Result, Count);
end;

function TClipTokenizer.EncodeSequence(const Text: UnicodeString; const Truncate: Boolean): TClipSequence;
var
  Ids: TArray<Integer>;
  I, Count: Integer;
begin
  Ids := Encode(Text);
  Count := Length(Ids);
  if (Count > ClipSequenceLength) and not Truncate then
    raise Exception.CreateFmt('CLIP text needs %d tokens; the limit is %d', [Count, ClipSequenceLength]);
  FillChar(Result, SizeOf(TClipSequence), 0);
  Result.Truncated := Count > ClipSequenceLength;
  if Count > ClipSequenceLength then Count := ClipSequenceLength;
  Result.TokenCount := Count;
  for I := 0 to ClipSequenceLength - 1 do
  begin
    Result.Ids[I] := ClipEndToken;
    if I < Count then
    begin
      Result.Ids[I] := Ids[I];
      Result.AttentionMask[I] := 1;
    end;
  end;
  Result.Ids[Count - 1] := ClipEndToken;
end;

function TClipTokenizer.Decode(const Ids: array of Integer): UnicodeString;
var
  Bytes: UTF8String;
  Name: UnicodeString;
  I, J, Code, Count, Taken: Integer;
  WordEnd: Boolean;
begin
  RequireLoaded;
  Bytes := '';
  Count := 0;
  for I := Low(Ids) to High(Ids) do
  begin
    if (Ids[I] < 0) or (Ids[I] >= ClipVocabularySize) then
      raise Exception.CreateFmt('Invalid CLIP decode token ID: %d', [Ids[I]]);
    if (Ids[I] = ClipStartToken) or (Ids[I] = ClipEndToken) then Continue;
    Name := FTokens[Ids[I]];
    Taken := Length(Name);
    WordEnd := Copy(Name, Taken - Length(ClipWordSuffix) + 1, Length(ClipWordSuffix)) =
      ClipWordSuffix;
    if WordEnd then Dec(Taken, Length(ClipWordSuffix));
    SetLength(Bytes, Count + Taken + Ord(WordEnd));
    for J := 1 to Taken do
    begin
      Code := Ord(Name[J]);
      if (Code > High(FCodeByte)) or (FCodeByte[Code] < 0) then
        raise Exception.Create('Invalid CLIP byte-level token character');
      Inc(Count);
      Bytes[Count] := AnsiChar(FCodeByte[Code]);
    end;
    if WordEnd then
    begin
      Inc(Count);
      Bytes[Count] := ' ';
    end;
  end;
  Result := Trim(UTF8Decode(Bytes));
end;

function TClipTokenizer.SpecialId(const Text: UnicodeString): Integer;
begin
  RequireLoaded;
  if Text = ClipStartText then Exit(ClipStartToken);
  if Text = ClipEndText then Exit(ClipEndToken);
  Result := -1;
end;

end.
