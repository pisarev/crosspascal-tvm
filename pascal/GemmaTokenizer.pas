{ ************************************************************************** }
{                                                                            }
{ GemmaTokenizer                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit GemmaTokenizer;

{$POINTERMATH ON}

{ Gemma SentencePiece BPE tokenizer. Split text into Unicode characters,
  use byte fallback for missing characters, and apply ranked merges. }

interface

uses
  {$IFDEF FPC}SysUtils, Character{$ELSE}System.SysUtils, System.Character{$ENDIF},
  BPETokenizer;

type
  TGemmaTokenizer = class(TCustomBPETokenizer)
  private
    type
      TSpan = record
        Offset: Cardinal;
        Length: Cardinal;
      end;
      TSpecial = record
        Id: Integer;
        Control: Boolean;   // Excluded when decoding.
        Text: UnicodeString;
      end;
    var
      FTokenData: TBytes;
      FTokenAt: array of TSpan;
      FCharCp: array of Cardinal;      // Sorted code points for single-character tokens.
      FCharId: array of Integer;       // Token IDs in the same order as FCharCp.
      FByteId: array[0..255] of Integer;
      FIsSpecial: array of Boolean;
      FSpecials: array of TSpecial;
      FBos: Integer;
      FLoaded: Boolean;
    function CharToId(const Cp: Cardinal): Integer;
    procedure Clear;
  public
    function Load(const Directory: string; const Error: PString): Boolean; override;
    function Loaded: Boolean; override;
    function TokenCount: Integer; override;
    function BosId: Integer;
    { Encode without a beginning-of-sequence token. The overload below adds it explicitly. }
    function Encode(const Text: UnicodeString): TArray<Integer>; overload; override;
    function Encode(const Text: UnicodeString;
      const PrependBos: Boolean): TArray<Integer>; overload;
    function Decode(const Ids: array of Integer): UnicodeString; override;
    function SpecialId(const Text: UnicodeString): Integer; override;
  end;

implementation

const
  SpaceMeta = Word($2581);   // SentencePiece space marker.

{ Find the token for a Unicode code point in the sorted table, or return -1. }
function TGemmaTokenizer.CharToId(const Cp: Cardinal): Integer;
var
  Lo, Hi, Mid: Integer;
begin
  Lo := 0;
  Hi := High(FCharCp);
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) shr 1;
    if FCharCp[Mid] = Cp then
      Exit(FCharId[Mid])
    else if FCharCp[Mid] < Cp then
        Lo := Mid + 1
      else
        Hi := Mid - 1;
  end;
  Result := -1;
end;

{ Reset the tables before loading so a failed reload cannot expose old tokens. }
procedure TGemmaTokenizer.Clear;
var
  I: Integer;
begin
  FLoaded := False;
  FBos := -1;
  ClearMerges;
  SetLength(FTokenData, 0);
  SetLength(FTokenAt, 0);
  SetLength(FCharCp, 0);
  SetLength(FCharId, 0);
  SetLength(FIsSpecial, 0);
  SetLength(FSpecials, 0);
  for I := 0 to 255 do
    FByteId[I] := -1;
end;

function TGemmaTokenizer.Load(const Directory: string; const Error: PString): Boolean;
var
  Dir: string;
  Index, Merges, Chars, Bytes: TBytes;
  Spans: PCardinal;
  Triples: PInteger;
  CharPairs, BytePtr: PInteger;
  Count, I: Integer;
  F: TextFile;
  Line, Key, Val, Rest: string;
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
  if not ReadWholeFile(Dir + 'tokenizer-chars.bin', Chars) then
  begin
    if Assigned(Error) then Error^ := 'Missing tokenizer-chars.bin';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-bytes.bin', Bytes) then
  begin
    if Assigned(Error) then Error^ := 'Missing tokenizer-bytes.bin';
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

  // Single-character entries are sorted by code point.
  Count := Length(Chars) div 8;
  SetLength(FCharCp, Count);
  SetLength(FCharId, Count);
  CharPairs := PInteger(@Chars[0]);
  for I := 0 to Count - 1 do
  begin
    FCharCp[I] := Cardinal(CharPairs[I * 2]);
    FCharId[I] := CharPairs[I * 2 + 1];
  end;

  // One fallback token for each byte value.
  if Length(Bytes) < 256 * 4 then
  begin
    if Assigned(Error) then Error^ := 'tokenizer-bytes.bin contains fewer than 256 entries';
    Exit;
  end;
  BytePtr := PInteger(@Bytes[0]);
  for I := 0 to 255 do
    FByteId[I] := BytePtr[I];

  // Read the merge triples into the inherited table.
  ReserveMerges(Length(Merges) div 12);
  Triples := PInteger(@Merges[0]);
  for I := 0 to (Length(Merges) div 12) - 1 do
    Put(Triples[I * 3], Triples[I * 3 + 1], Triples[I * 3 + 2], I);

  // Special tokens.
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
          Position := Pos(' ', Line);          // Skip the token ID.
          if Position < 2 then
            Continue;
          FSpecials[I].Id := StrToIntDef(Copy(Line, 1, Position - 1), -1);
          Rest := Copy(Line, Position + 1, Length(Line));
          Position := Pos(' ', Rest);          // Skip the flag.
          if Position < 1 then
            Continue;
          FSpecials[I].Control := StrToIntDef(Copy(Rest, 1, Position - 1), 0) = 1;
          FSpecials[I].Text := UnicodeString(
            StringReplace(
              Copy(
                Rest,
                Position + 1,
                Length(Rest)
              ),
              '\n',
              #10,
              [rfReplaceAll]
            )
          );
          Inc(I);
        end;
        SetLength(FSpecials, I);
      finally
        CloseFile(F);
      end;
  end;

  // Drop control tokens and retain unused and whitespace tokens.
  SetLength(FIsSpecial, Length(FTokenAt));
  for I := 0 to High(FSpecials) do
    if FSpecials[I].Control and (FSpecials[I].Id >= 0) and
      (FSpecials[I].Id <= High(FIsSpecial)) then
        FIsSpecial[FSpecials[I].Id] := True;

  // Beginning-of-sequence token from tokenizer.index.
  if FileExists(Dir + 'tokenizer.index') then
  begin
    AssignFile(F, Dir + 'tokenizer.index');
    {$I-}
    Reset(F);
    {$I+}
    if IOResult = 0 then
      try
        while not Eof(F) do
        begin
          Readln(F, Line);
          Position := Pos(' ', Line);
          if Position < 2 then
            Continue;
          Key := Copy(Line, 1, Position - 1);
          Val := Copy(Line, Position + 1, Length(Line));
          if Key = 'bos' then
            FBos := StrToIntDef(Trim(Val), -1);
        end;
      finally
        CloseFile(F);
      end;
  end;
  FLoaded := True;
  Result := True;
end;

function TGemmaTokenizer.Loaded: Boolean;
begin
  Result := FLoaded;
end;

function TGemmaTokenizer.TokenCount: Integer;
begin
  Result := Length(FTokenAt);
end;

function TGemmaTokenizer.BosId: Integer;
begin
  Result := FBos;
end;

function TGemmaTokenizer.SpecialId(const Text: UnicodeString): Integer;
var
  I: Integer;
begin
  RequireLoaded;
  for I := 0 to High(FSpecials) do
    if FSpecials[I].Text = Text then
      Exit(FSpecials[I].Id);
  Result := -1;
end;

{ Encode one Unicode code point into Buf and return its UTF-8 byte count. }
function CodepointToUtf8(const Cp: Cardinal; out Buf: array of Byte): Integer;
begin
  if Cp < $80 then
  begin
    Buf[0] := Byte(Cp);
    Result := 1;
  end
  else if Cp < $800 then
    begin
      Buf[0] := Byte($C0 or (Cp shr 6));
      Buf[1] := Byte($80 or (Cp and $3F));
      Result := 2;
    end
    else if Cp < $10000 then
      begin
        Buf[0] := Byte($E0 or (Cp shr 12));
        Buf[1] := Byte($80 or ((Cp shr 6) and $3F));
        Buf[2] := Byte($80 or (Cp and $3F));
        Result := 3;
      end
      else begin
        Buf[0] := Byte($F0 or (Cp shr 18));
        Buf[1] := Byte($80 or ((Cp shr 12) and $3F));
        Buf[2] := Byte($80 or ((Cp shr 6) and $3F));
        Buf[3] := Byte($80 or (Cp and $3F));
        Result := 4;
      end;
end;

function TGemmaTokenizer.Encode(const Text: UnicodeString): TArray<Integer>;
begin
  Result := Encode(Text, False);
end;

function TGemmaTokenizer.Encode(const Text: UnicodeString;
  const PrependBos: Boolean): TArray<Integer>;
var
  Norm: UnicodeString;
  Work: TArray<Integer>;
  Count, I, J, Cp, Id, Consumed, N: Integer;
  Utf8: array[0..3] of Byte;
begin
  RequireLoaded;
  SetLength(Result, 0);
  if (Text = '') and not PrependBos then
    Exit;

  { Use UTF-16 code units here so U+2581 and non-ASCII code points survive normalization. }
  Norm := Text;
  // Replace spaces with the SentencePiece space marker.
  for I := 1 to Length(Norm) do
    if Norm[I] = ' ' then
      Norm[I] := WideChar(SpaceMeta);
  { Read Unicode code points, combining surrogate pairs, then use a single
    token or fall back to UTF-8 bytes. }
  { Reserve room for UTF-8 byte fallback without growing the work array. }
  SetLength(Work, Length(Norm) * 4 + 4);
  Count := 0;
  I := 1;
  while I <= Length(Norm) do
  begin
    if TCharacter.IsHighSurrogate(Norm[I]) and (I < Length(Norm)) and TCharacter.IsLowSurrogate(Norm[I + 1]) then
    begin
      Cp := $10000 + ((Ord(Norm[I]) - $D800) shl 10) + (Ord(Norm[I + 1]) - $DC00);
      Consumed := 2;
    end
    else begin
      Cp := Ord(Norm[I]);
      Consumed := 1;
    end;
    Inc(I, Consumed);
    Id := CharToId(Cardinal(Cp));
    if Id >= 0 then
    begin
      Work[Count] := Id;
      Inc(Count);
    end
    else begin
      N := CodepointToUtf8(Cardinal(Cp), Utf8);
      for J := 0 to N - 1 do
      begin
        Work[Count] := FByteId[Utf8[J]];
        Inc(Count);
      end;
    end;
  end;

  { Merge only the occupied entries. The optional beginning-of-sequence token is added afterward. }
  Count := MergeBPE(Work, Count);
  if PrependBos and (FBos >= 0) then
  begin
    SetLength(Result, Count + 1);
    Result[0] := FBos;
    for I := 0 to Count - 1 do
      Result[I + 1] := Work[I];
  end
  else begin
    SetLength(Result, Count);
    for I := 0 to Count - 1 do
      Result[I] := Work[I];
  end;
end;

{ Decode token bytes, skip control tokens, and restore spaces. }
function TGemmaTokenizer.Decode(const Ids: array of Integer): UnicodeString;
var
  Raw: TBytes;
  Encoded: UTF8String;
  Wide: UnicodeString;
  Total, At, I: Integer;
  Keep: Boolean;
begin
  RequireLoaded;
  Total := 0;
  for I := 0 to High(Ids) do
    if (Ids[I] >= 0) and (Ids[I] <= High(FTokenAt)) and not FIsSpecial[Ids[I]] then
      Inc(Total, FTokenAt[Ids[I]].Length);
  SetLength(Raw, Total);
  At := 0;
  for I := 0 to High(Ids) do
  begin
    if (Ids[I] < 0) or (Ids[I] > High(FTokenAt)) then
      Continue;
    Keep := not FIsSpecial[Ids[I]];
    if Keep and (FTokenAt[Ids[I]].Length > 0) then
    begin
      Move(FTokenData[FTokenAt[Ids[I]].Offset], Raw[At], FTokenAt[Ids[I]].Length);
      Inc(At, FTokenAt[Ids[I]].Length);
    end;
  end;
  { Decode the concatenated bytes once. A token may contain only part of a UTF-8 character. }
  SetLength(Encoded, Length(Raw));
  if Length(Raw) > 0 then Move(Raw[0], Encoded[1], Length(Raw));
  Wide := UTF8Decode(Encoded);
  for I := 1 to Length(Wide) do
    if Wide[I] = WideChar(SpaceMeta) then
      Wide[I] := ' ';
  Result := Wide;
end;

end.
