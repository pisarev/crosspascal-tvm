{ ************************************************************************** }
{                                                                            }
{ GemmaTokenizer                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit GemmaTokenizer;

{$POINTERMATH ON}

// Токенизатор Gemma (SentencePiece-BPE) со стороны Паскаля. Шаг Ш5 лестницы.
//
// Отличие от QwenTokenizer (побайтовый BPE): начальная разбивка ПОСИМВОЛЬНАЯ.
// Пробел заменяется на U+2581, затем каждый символ берется как свой
// односимвольный токен, если он есть в словаре; иначе byte-fallback - каждый
// UTF-8 байт символа дает токен <0xXX>. Дальше те же слияния-триплеты по
// рангу, что и у Qwen. Порядок совпал с HF 7 из 7.
//
// Таблицы готовит export_tokenizer_gemma.py. QwenTokenizer не трогается -
// это инструмент жемчужины.

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
        Control: Boolean;   // special=true: декод отбрасывает
        Text: UnicodeString;
      end;
    var
      FTokenData: TBytes;
      FTokenAt: array of TSpan;
      FCharCp: array of Cardinal;      // односимвольные токены: точки, по возр.
      FCharId: array of Integer;       // параллельно - их номера
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
    { Работа, обещанная предком: разбор БЕЗ затравочного токена, то самое
      поведение, с которым сверяется корпус эталона. Затравку ставит второй вид
      ниже, и зовут его явно - молчаливой затравки здесь нет. }
    function Encode(const Text: UnicodeString): TArray<Integer>; overload; override;
    function Encode(const Text: UnicodeString;
      const PrependBos: Boolean): TArray<Integer>; overload;
    function Decode(const Ids: array of Integer): UnicodeString; override;
    function SpecialId(const Text: UnicodeString): Integer; override;
  end;

implementation

const
  SpaceMeta = Word($2581);   // ▁ - метапробел SentencePiece

{ Номер односимвольного токена по точке Юникода, или -1. Двоичный поиск:
  таблица отсортирована по возрастанию точки при выгрузке. }
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

{ ---- загрузка ---- }

{
  Сброс таблиц в начало загрузки.

  Записи этого не требовали: каждый заход брал свежую переменную, и прошлого
  состояния взяться было неоткуда. У объекта оно берется. Загрузка выходит
  досрочно на каждом из ПЯТИ обязательных файлов, и без сброса неуспешная
  загрузка вслед за успешной оставила бы объект со СТАРЫМИ таблицами при
  Loaded, равном False: отказ виден, а данные прежние.
}
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
    if Assigned(Error) then Error^ := 'нет tokenizer-tokens.bin';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-tokens.index', Index) then
  begin
    if Assigned(Error) then Error^ := 'нет tokenizer-tokens.index';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-merges.bin', Merges) then
  begin
    if Assigned(Error) then Error^ := 'нет tokenizer-merges.bin';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-chars.bin', Chars) then
  begin
    if Assigned(Error) then Error^ := 'нет tokenizer-chars.bin';
    Exit;
  end;
  if not ReadWholeFile(Dir + 'tokenizer-bytes.bin', Bytes) then
  begin
    if Assigned(Error) then Error^ := 'нет tokenizer-bytes.bin';
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

  // односимвольные токены: пары (точка, номер), уже по возрастанию точки
  Count := Length(Chars) div 8;
  SetLength(FCharCp, Count);
  SetLength(FCharId, Count);
  CharPairs := PInteger(@Chars[0]);
  for I := 0 to Count - 1 do
  begin
    FCharCp[I] := Cardinal(CharPairs[I * 2]);
    FCharId[I] := CharPairs[I * 2 + 1];
  end;

  // byte-fallback: 256 номеров, по байту
  if Length(Bytes) < 256 * 4 then
  begin
    if Assigned(Error) then Error^ := 'tokenizer-bytes.bin короче 256 записей';
    Exit;
  end;
  BytePtr := PInteger(@Bytes[0]);
  for I := 0 to 255 do
    FByteId[I] := BytePtr[I];

  // сама таблица живет в предке; здесь только разбор файла на тройки чисел
  ReserveMerges(Length(Merges) div 12);
  Triples := PInteger(@Merges[0]);
  for I := 0 to (Length(Merges) div 12) - 1 do
    Put(Triples[I * 3], Triples[I * 3 + 1], Triples[I * 3 + 2], I);

  // особые токены
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
        Position := Pos(' ', Line);          // после id
        if Position < 2 then
          Continue;
        FSpecials[I].Id := StrToIntDef(Copy(Line, 1, Position - 1), -1);
        Rest := Copy(Line, Position + 1, Length(Line));
        Position := Pos(' ', Rest);          // после флага
        if Position < 1 then
          Continue;
        FSpecials[I].Control := StrToIntDef(Copy(Rest, 1, Position - 1), 0) = 1;
        FSpecials[I].Text := UnicodeString(
          StringReplace(Copy(Rest, Position + 1, Length(Rest)),
            '\n', #10, [rfReplaceAll]));
        Inc(I);
      end;
      SetLength(FSpecials, I);
    finally
      CloseFile(F);
    end;
  end;

  // декод отбрасывает ТОЛЬКО управляющие (special=true), а содержимое
  // (▁▁, <unusedN>) сохраняет - как эталон skip_special_tokens
  SetLength(FIsSpecial, Length(FTokenAt));
  for I := 0 to High(FSpecials) do
    if FSpecials[I].Control and (FSpecials[I].Id >= 0)
      and (FSpecials[I].Id <= High(FIsSpecial)) then
      FIsSpecial[FSpecials[I].Id] := True;

  // bos из tokenizer.index
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

{ ---- кодирование ---- }

{ Точка Юникода -> её UTF-8 байты в Buf, возвращает число байтов. }
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
  else
  begin
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

  {
    Norm широкий, и весь посимвольный обход ниже идет по нему. С 21.09.2026
    широким приходит и сам довод, так что перевода здесь больше нет вовсе;
    местная переменная осталась потому, что обход ее ПРАВИТ - подставляет
    метапробел, - а довод объявлен const.

    Ниже записано, почему широта здесь была обязательна еще ДО смены договора.
    Корень, общий для обоих мест этого модуля и для QwenTokenizer: под FPC
    string есть последовательность БАЙТОВ в кодировке системы, а не
    последовательность точек Юникода. Отсюда две беды разом, и обе тихие.

    Первая - метапробел. U+2581 в однобайтовой кодировке непредставим, и
    подстановка давала вопросительный знак вместо него. Бьет по КАЖДОЙ строке
    с пробелом, то есть почти по всем.

    Вторая - чтение. Ниже стоит Cp := Ord(Norm[I]); от однобайтовой строки это
    дало бы БАЙТ кодовой страницы вместо точки Юникода: у «п» вышло бы 239
    вместо 1055, и точка ушла бы в латиницу. Потери при этом нет вовсе - строка
    держит верную кириллицу, круговой ход байт в байт сходится, - портится
    именно ЧТЕНИЕ.

    Замер 20.09.2026 под FPC до правки: сошлось 65 строк корпуса из 1074.
    Уцелевали ровно те, где нет пробела И каждый знак имеет в кодовой странице
    байт, равный своей точке Юникода (семибитные и еще пятнадцать, среди них
    неразрывный пробел). Предсказано по этому правилу 65, измерено 65.

    Чинить одну беду без другой бесполезно: почини метапробел - останутся сотни
    строк, падающих на Ord, и батарея поднимется настолько, что поиск закроют.
  }
  Norm := Text;
  // нормализация: пробел -> ▁ (единственная у Gemma, NFC нет)
  for I := 1 to Length(Norm) do
    if Norm[I] = ' ' then
      Norm[I] := WideChar(SpaceMeta);

  // посимвольная разбивка: символ -> его токен, иначе байты -> byte-fallback.
  // Точку Юникода собираем сами: суррогатную пару в одну, иначе один код.
  {
    Оценка емкости есть ПРЕДУСЛОВИЕ: роста по нужде ниже нет вовсе. Пересчитана
    заново после смены договора, потому что вход впервые несет знаки вне
    кодовой страницы. Граница прежняя: знак основной плоскости дает не более
    трех байтов UTF-8, то есть не более трех элементов при байтовом откате;
    суррогатная пара - четыре байта на ДВА знака, то есть два элемента на знак.
    Отведено четыре на знак, и запас не сократился.
  }
  SetLength(Work, Length(Norm) * 4 + 4);
  Count := 0;
  I := 1;
  while I <= Length(Norm) do
  begin
    if TCharacter.IsHighSurrogate(Norm[I]) and (I < Length(Norm))
      and TCharacter.IsLowSurrogate(Norm[I + 1]) then
    begin
      Cp := $10000 + ((Ord(Norm[I]) - $D800) shl 10) + (Ord(Norm[I + 1]) - $DC00);
      Consumed := 2;
    end
    else
    begin
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
    else
    begin
      N := CodepointToUtf8(Cardinal(Cp), Utf8);
      for J := 0 to N - 1 do
      begin
        Work[Count] := FByteId[Utf8[J]];
        Inc(Count);
      end;
    end;
  end;

  {
    Слияние живет в предке. Массив здесь отведен С ЗАПАСОМ ВЧЕТВЕРО - под
    худший случай байтового отката, - а действительных элементов в нем много
    меньше. Поэтому число элементов передается отдельным доводом: возьми общая
    работа Length(Work), она сливала бы мусор за границей. У Qwen обе величины
    совпадают, и такая ошибка осталась бы там невидимой.

    Затравка в слияниях НЕ УЧАСТВУЕТ, и это не оплошность, а нынешнее
    поведение: она приписывается ниже, уже к готовому разбору.
  }
  Count := MergeBPE(Work, Count);

  if PrependBos and (FBos >= 0) then
  begin
    SetLength(Result, Count + 1);
    Result[0] := FBos;
    for I := 0 to Count - 1 do
      Result[I + 1] := Work[I];
  end
  else
  begin
    SetLength(Result, Count);
    for I := 0 to Count - 1 do
      Result[I] := Work[I];
  end;
end;

{ ---- декодирование ---- }

{ Байты выданных токенов обратно в текст. Особые номера пропускаются (эталон
  при разборке их тоже отбрасывает). Метапробел ▁ возвращается пробелом. }
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
    if (Ids[I] >= 0) and (Ids[I] <= High(FTokenAt))
      and not FIsSpecial[Ids[I]] then
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
  {
    Байты собраны ЦЕЛИКОМ и лишь потом переведены в текст, один раз. Переводить
    каждый отрезок отдельно нельзя: слово словаря вправе нести ЧАСТЬ
    многобайтового знака, и такой отрезок сам по себе не есть текст.

    Обратный ход метапробела идет по широкой строке и до возврата. Прежде
    перевод в простую строку шел ПЕРВЫМ: TEncoding отдает широкую строку,
    присвоение в однобайтовую роняло U+2581 в вопросительный знак, и
    подстановка ниже искала метапробел, которого уже не было. Пробелы не
    восстанавливались вовсе.

    С 21.09.2026 сужения на выходе НЕТ: подпись широкая. Прежняя последняя
    строка - перевод Wide в простую строку - не просто стала лишней, она была
    потолком разбора у этой работы.
  }
  SetLength(Encoded, Length(Raw));
  if Length(Raw) > 0 then Move(Raw[0], Encoded[1], Length(Raw));
  Wide := UTF8Decode(Encoded);
  for I := 1 to Length(Wide) do
    if Wide[I] = WideChar(SpaceMeta) then
      Wide[I] := ' ';
  Result := Wide;
end;

end.
