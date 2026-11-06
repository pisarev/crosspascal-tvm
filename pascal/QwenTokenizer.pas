{ ************************************************************************** }
{                                                                            }
{ QwenTokenizer                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit QwenTokenizer;

{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Character{$ELSE}System.SysUtils, System.Character{$ENDIF},
  BPETokenizer;

type
  TQwenTokenizer = class(TCustomBPETokenizer)
  private
    type
      TSpan = record
        Offset: Cardinal;
        Length: Cardinal;
      end;
      TSpecial = record
        Id: Integer;
        Text: UnicodeString;
      end;
    var
      FTokenData: TBytes;
      FTokenAt: array of TSpan;
      FSingle: array[0..255] of Integer;
      FSpecials: array of TSpecial;
      FLoaded: Boolean;
    function PieceToIds(const Piece: TBytes): TArray<Integer>;
    procedure SplitPiece(const Text: UnicodeString; const From, Upto: Integer;
      var Ids: TArray<Integer>; var Count: Integer);
    procedure Clear;
  public
    function Load(const Directory: string; const Error: PString): Boolean; override;
    function Loaded: Boolean; override;
    function TokenCount: Integer; override;
    function Encode(const Text: UnicodeString): TArray<Integer>; overload; override;
    function Decode(const Ids: array of Integer): UnicodeString; override;
    function SpecialId(const Text: UnicodeString): Integer; override;
  end;

function NormalizeNFCWide(const Text: UnicodeString): UnicodeString;

implementation

{$IFDEF MSWINDOWS}
uses
  {$IFDEF FPC}Windows{$ELSE}Winapi.Windows{$ENDIF};

const
  NormalizationC = 1;

function NormalizeString(NormForm: Integer; lpSrcString: PWideChar; cwSrcLength: Integer;
  lpDstString: PWideChar; cwDstLength: Integer): Integer;
  stdcall; external 'Normaliz.dll' name 'NormalizeString';

function NormalizeNFCWide(const Text: UnicodeString): UnicodeString;
var
  Composed: UnicodeString;
  Needed: Integer;
begin
  if Text = '' then
    Exit('');
  Needed := NormalizeString(NormalizationC, PWideChar(Text), Length(Text), nil, 0);
  if Needed <= 0 then
    raise Exception.CreateFmt('NormalizeNFC: could not determine output size; error %d', [GetLastError]);
  SetLength(Composed, Needed);
  Needed := NormalizeString(NormalizationC, PWideChar(Text), Length(Text),
    PWideChar(Composed), Needed);
  if Needed <= 0 then
    raise Exception.CreateFmt('NormalizeNFC: normalization failed; error %d', [GetLastError]);
  SetLength(Composed, Needed);
  Result := Composed;
end;
{$ELSE}
function NormalizeNFCWide(const Text: UnicodeString): UnicodeString;
begin
  Result := Text;
end;
{$ENDIF}

function StepAt(const Text: UnicodeString; const Index: Integer): Integer;
begin
  if (Index < Length(Text)) and TCharacter.IsHighSurrogate(Text[Index]) and
    TCharacter.IsLowSurrogate(Text[Index + 1]) then
      Result := 2
  else
    Result := 1;
end;

procedure TQwenTokenizer.Clear;
var
  I: Integer;
begin
  FLoaded := False;
  ClearMerges;
  SetLength(FTokenData, 0);
  SetLength(FTokenAt, 0);
  SetLength(FSpecials, 0);
  for I := 0 to 255 do
    FSingle[I] := -1;
end;

function TQwenTokenizer.Load(const Directory: string; const Error: PString): Boolean;
var
  Dir: string;
  Index, Merges: TBytes;
  Spans: PCardinal;
  Triples: PInteger;
  Count, I: Integer;
  F: TextFile;
  Line: string;
  Position: Integer;
begin
  Result := False;
  Clear;
  Dir := IncludeTrailingPathDelimiter(Directory);
  if not ReadWholeFile(Dir + 'tokenizer-tokens.bin', FTokenData) then
  begin
    if Assigned(Error) then Error^ := 'Missing tokenizer-tokens.bin';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-tokens.index', Index) then
  begin
    if Assigned(Error) then Error^ := 'Missing tokenizer-tokens.index';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-merges.bin', Merges) then
  begin
    if Assigned(Error) then Error^ := 'Missing tokenizer-merges.bin';
    Exit;
  end;
  Count := Length(Index) div 8;
  SetLength(FTokenAt, Count);
  Spans := PCardinal(@Index[0]);
  for I := 0 to Count - 1 do
  begin
    FTokenAt[I].Offset := Spans[I * 2];
    FTokenAt[I].Length := Spans[I * 2 + 1];
  end;
  for I := 0 to 255 do
    FSingle[I] := -1;
  for I := 0 to Count - 1 do
    if (FTokenAt[I].Length = 1) and (FSingle[FTokenData[FTokenAt[I].Offset]] < 0) then
      FSingle[FTokenData[FTokenAt[I].Offset]] := I;
  for I := 0 to 255 do
    if FSingle[I] < 0 then
    begin
      if Assigned(Error) then
        Error^ := Format('Vocabulary has no single-byte token for byte %d', [I]);
      Exit;
    end;
  ReserveMerges(Length(Merges) div 12);
  Triples := PInteger(@Merges[0]);
  for I := 0 to (Length(Merges) div 12) - 1 do
    Put(Triples[I * 3], Triples[I * 3 + 1], Triples[I * 3 + 2], I);
  SetLength(FSpecials, 0);
  if FileExists(Dir + 'tokenizer-specials.txt') then
  begin
    AssignFile(F, Dir + 'tokenizer-specials.txt');
    {$I-}
    Reset(F);
    {$I+}
    if IOResult = 0 then
      try
        Readln(F, Line);
        SetLength(FSpecials, StrToIntDef(Trim(Line), 0));
        I := 0;
        while (I <= High(FSpecials)) and not Eof(F) do
        begin
          Readln(F, Line);
          Position := Pos(' ', Line);
          if Position < 2 then
            Continue;
          FSpecials[I].Id := StrToIntDef(Copy(Line, 1, Position - 1), -1);
          FSpecials[I].Text := UnicodeString(Copy(Line, Position + 1, Length(Line)));
          Inc(I);
        end;
        SetLength(FSpecials, I);
      finally
        CloseFile(F);
      end;
  end;
  FLoaded := True;
  Result := True;
end;

function TQwenTokenizer.Loaded: Boolean;
begin
  Result := FLoaded;
end;

function TQwenTokenizer.TokenCount: Integer;
begin
  Result := Length(FTokenAt);
end;

function TQwenTokenizer.PieceToIds(const Piece: TBytes): TArray<Integer>;
var
  Work: TArray<Integer>;
  Count, I: Integer;
begin
  Count := Length(Piece);
  SetLength(Result, 0);
  if Count = 0 then
    Exit;
  SetLength(Work, Count);
  for I := 0 to Count - 1 do
    Work[I] := FSingle[Piece[I]];
  Count := MergeBPE(Work, Count);
  SetLength(Result, Count);
  for I := 0 to Count - 1 do
    Result[I] := Work[I];
end;

function MatchLength(const Text: UnicodeString; const Start, Upto: Integer): Integer;

  function Letter(const I: Integer): Boolean;
  begin
    Result := (I >= 1) and (I <= Upto) and TCharacter.IsLetter(Text, I);
  end;

  function Number(const I: Integer): Boolean;
  begin
    Result := (I >= 1) and (I <= Upto) and TCharacter.IsNumber(Text, I);
  end;

  function Space(const I: Integer): Boolean;
  begin
    Result := (I >= 1) and (I <= Upto) and TCharacter.IsWhiteSpace(Text, I);
  end;

  function Break_(const I: Integer): Boolean;
  begin
    Result := (I >= 1) and (I <= Upto) and ((Text[I] = #13) or (Text[I] = #10));
  end;

  function Step(const I: Integer): Integer;
  begin
    if (I < Upto) and TCharacter.IsHighSurrogate(Text[I]) and TCharacter.IsLowSurrogate(Text[I + 1]) then
      Result := 2
    else
      Result := 1;
  end;

var
  At, RunEnd, Save: Integer;
  Ch: WideChar;
begin
  Result := 0;
  if Start > Upto then
    Exit;
  if Text[Start] = '''' then
  begin
    if Start + 1 <= Upto then
    begin
      Ch := TCharacter.ToLower(Text[Start + 1]);
      if CharInSet(Ch, ['s', 't', 'm', 'd']) then
        Exit(2);
      if (Start + 2 <= Upto) and (((Ch = 'r') and (TCharacter.ToLower(Text[Start + 2]) = 'e')) or
        ((Ch = 'v') and (TCharacter.ToLower(Text[Start + 2]) = 'e')) or ((Ch = 'l') and
        (TCharacter.ToLower(Text[Start + 2]) = 'l'))) then
          Exit(3);
    end;
  end;
  At := Start;
  if not Letter(At) and not Number(At) and not Break_(At) then
    At := At + Step(At);
  if Letter(At) then
  begin
    while Letter(At) do
      At := At + Step(At);
    Exit(At - Start);
  end;
  if Number(Start) then
    Exit(Step(Start));
  At := Start;
  if Text[At] = ' ' then
    Inc(At);
  Save := At;
  while (At <= Upto) and not Space(At) and not Letter(At) and not Number(At) do
    At := At + Step(At);
  if At > Save then
  begin
    while Break_(At) do
      Inc(At);
    Exit(At - Start);
  end;
  At := Start;
  while Space(At) and not Break_(At) do
    At := At + Step(At);
  if Break_(At) then
  begin
    while Break_(At) do
      Inc(At);
    Exit(At - Start);
  end;
  At := Start;
  if Space(At) then
  begin
    RunEnd := At;
    while Space(RunEnd) do
      RunEnd := RunEnd + Step(RunEnd);
    if RunEnd > Upto then
      Exit(RunEnd - Start);
    if RunEnd - 1 > Start then
      Exit(RunEnd - 1 - Start);
    Exit(RunEnd - Start);
  end;
end;

procedure TQwenTokenizer.SplitPiece(const Text: UnicodeString; const From, Upto: Integer;
  var Ids: TArray<Integer>; var Count: Integer);
var
  At, Taken, I: Integer;
  Chunk: UnicodeString;
  Raw: TBytes;
  Part: TArray<Integer>;
begin
  At := From;
  while At <= Upto do
  begin
    Taken := MatchLength(Text, At, Upto);
    if Taken <= 0 then
      Taken := StepAt(Text, At);
    Chunk := Copy(Text, At, Taken);
    Raw := TEncoding.UTF8.GetBytes(Chunk);
    Part := PieceToIds(Raw);
    if Count + Length(Part) > Length(Ids) then
      SetLength(Ids, (Count + Length(Part)) * 2 + 16);
    for I := 0 to High(Part) do
    begin
      Ids[Count] := Part[I];
      Inc(Count);
    end;
    Inc(At, Taken);
  end;
end;

function TQwenTokenizer.Encode(const Text: UnicodeString): TArray<Integer>;
var
  Normalized: UnicodeString;
  At, Best, BestAt, I, Position, Count: Integer;
begin
  RequireLoaded;
  SetLength(Result, 0);
  Count := 0;
  if Text = '' then
    Exit;
  Normalized := NormalizeNFCWide(Text);
  SetLength(Result, Length(Normalized) + 16);
  At := 1;
  while At <= Length(Normalized) do
  begin
    Best := -1;
    BestAt := MaxInt;
    for I := 0 to High(FSpecials) do
    begin
      if FSpecials[I].Text = '' then
        Continue;
      Position := Pos(FSpecials[I].Text, Normalized, At);
      if Position <= 0 then
        Continue;
      if (Position < BestAt) or ((Position = BestAt) and (Best >= 0) and (Length(FSpecials[I].Text) > Length(FSpecials[Best].Text))) then
      begin
        BestAt := Position;
        Best := I;
      end;
    end;
    if Best < 0 then
    begin
      SplitPiece(Normalized, At, Length(Normalized), Result, Count);
      Break;
    end;
    if BestAt > At then
      SplitPiece(Normalized, At, BestAt - 1, Result, Count);
    if Count >= Length(Result) then
      SetLength(Result, Count * 2 + 16);
    Result[Count] := FSpecials[Best].Id;
    Inc(Count);
    At := BestAt + Length(FSpecials[Best].Text);
  end;
  SetLength(Result, Count);
end;

function TQwenTokenizer.SpecialId(const Text: UnicodeString): Integer;
var
  I: Integer;
begin
  RequireLoaded;
  for I := 0 to High(FSpecials) do
    if FSpecials[I].Text = Text then
      Exit(FSpecials[I].Id);
  Result := -1;
end;

function TQwenTokenizer.Decode(const Ids: array of Integer): UnicodeString;
var
  Raw: TBytes;
  Total, At, I: Integer;
begin
  RequireLoaded;
  Total := 0;
  for I := 0 to High(Ids) do
    if (Ids[I] >= 0) and (Ids[I] <= High(FTokenAt)) then
      Inc(Total, FTokenAt[Ids[I]].Length);
  SetLength(Raw, Total);
  At := 0;
  for I := 0 to High(Ids) do
    if (Ids[I] >= 0) and (Ids[I] <= High(FTokenAt)) then
    begin
      if FTokenAt[Ids[I]].Length > 0 then
        Move(FTokenData[FTokenAt[Ids[I]].Offset], Raw[At], FTokenAt[Ids[I]].Length);
      Inc(At, FTokenAt[Ids[I]].Length);
    end;
  Result := TEncoding.UTF8.GetString(Raw);
end;

end.
