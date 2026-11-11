{ ************************************************************************** }
{                                                                            }
{ QwenModel                                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit QwenModel;

{$IFDEF FPC}{$MODE DELPHI}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF},
  DLPack,
  TVM;

type
  TQwenParamEntry = record
    Name: string;
    Offset: Int64;
    ByteCount: Int64;
    Dims: TArray<Int64>;
  end;

  TQwenModel = record
  public
    Directory: string;
    Entries: TArray<TQwenParamEntry>;
    ParamHandles: TArray<TTVMFFIObjectHandle>;
    ParamsArray: TTVMFFIObjectHandle;
    DeviceType: Integer;
    DataBits: Integer;
    ModuleHandle: TTVMFFIObjectHandle;
    VMHandle: TTVMFFIObjectHandle;
    CacheHandle: TTVMFFIObjectHandle;
    EmbedFunc: TTVMFFIObjectHandle;
    PrefillFunc: TTVMFFIObjectHandle;
    DecodeFunc: TTVMFFIObjectHandle;
    function Load(const ADirectory: string; const Error: PString;
      const LibraryName: string = ''): Boolean;
    procedure Unload;
    function CreateCache(const MaxBatch, MaxTotal, Chunk, Page: Int64;
      const Error: PString): Boolean;
    function AddSequence(const SeqId: Int64; const Error: PString): Boolean;
    function RemoveSequence(const SeqId: Int64; const Error: PString): Boolean;
    function ClearCache(const Error: PString): Boolean;
    function Embed(const Tokens: array of Integer; out TensorHandle: TTVMFFIObjectHandle;
      const Error: PString): Boolean;
    function Run(const Which: string; const EmbedTensor: TTVMFFIObjectHandle;
      const SeqId, Count: Int64; out LogitsHandle: TTVMFFIObjectHandle;
      const Error: PString): Boolean;
    function IndexOfParam(const Name: string): Integer;
    function ReadFloats(const TensorHandle: TTVMFFIObjectHandle; out Values: TArray<Single>;
      const Error: PString): Boolean;
  end;

function MakeShape(const Values: array of Int64; out Handle: TTVMFFIObjectHandle;
  const Error: PString): Boolean;

type
  TQwenWords = TArray<string>;

function SplitWords(const Line: string): TQwenWords;

function ReadChunk(const Path: string; const Offset, Count: Int64; const Target: Pointer): Boolean;

function ArgMaxOf(const Values: PSingle; const Count: Int64): Int64;

implementation

uses
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF};

function SplitWords(const Line: string): TQwenWords;
var
  I, Start, Count: Integer;
begin
  SetLength(Result, 0);
  Count := 0;
  I := 1;
  while I <= Length(Line) do
  begin
    while (I <= Length(Line)) and (Line[I] = ' ') do
      Inc(I);
    if I > Length(Line) then
      Break;
    Start := I;
    while (I <= Length(Line)) and (Line[I] <> ' ') do
      Inc(I);
    Inc(Count);
    SetLength(Result, Count);
    Result[Count - 1] := Copy(Line, Start, I - Start);
  end;
end;

function ReadChunk(const Path: string; const Offset, Count: Int64; const Target: Pointer): Boolean;
var
  Stream: TFileStream;
  ReadBytes: Integer;
  Remaining: Int64;
  Dest: PByte;
begin
  Result := False;
  if (Offset < 0) or (Count < 0) or ((Count > 0) and not Assigned(Target)) then
    Exit;
  try
    Stream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
    try
      if (Offset > Stream.Size) or (Count > Stream.Size - Offset) then
        Exit;
      Stream.Position := Offset;
      Dest := PByte(Target);
      Remaining := Count;
      while Remaining > 0 do
      begin
        ReadBytes := 64 * 1024 * 1024;
        if Remaining < ReadBytes then ReadBytes := Integer(Remaining);
        if Stream.Read(Dest^, ReadBytes) <> ReadBytes then Exit;
        Inc(Dest, ReadBytes);
        Dec(Remaining, ReadBytes);
      end;
      Result := True;
    finally
      Stream.Free;
    end;
  except
    on E: EStreamError do Result := False;
    on E: EOSError do Result := False;
  end;
end;

function MakeShape(const Values: array of Int64; out Handle: TTVMFFIObjectHandle;
  const Error: PString): Boolean;
var
  Args: array of TTVMFFIAny;
  Res: TTVMFFIAny;
  I: Integer;
begin
  Handle := nil;
  SetLength(Args, Length(Values));
  for I := 0 to High(Values) do
    Args[I] := TTVMRuntime.AnyInt(Values[I]);
  if not TTVMRuntime.CallGlobal('ffi.Shape', Args, Res, Error) then
    Exit(False);
  Handle := Res.VObj;
  Result := Res.TypeIndex = kTVMFFIShape;
end;

function ArgMaxOf(const Values: PSingle; const Count: Int64): Int64;
var
  I: Int64;
  Best: Single;
begin
  Result := 0;
  if Count <= 0 then
    Exit;
  Best := Values[0];
  for I := 1 to Count - 1 do
    if Values[I] > Best then
    begin
      Best := Values[I];
      Result := I;
    end;
end;

function ReadParamIndex(const Path: string; out Entries: TArray<TQwenParamEntry>;
  out DataType, Target: string): Boolean;
var
  F: TextFile;
  Line: string;
  Words: TArray<string>;
  Total, Filled, D: Integer;
begin
  Result := False;
  SetLength(Entries, 0);
  DataType := '';
  Target := 'cpu';
  AssignFile(F, Path);
  {$I-}
  Reset(F);
  {$I+}
  if IOResult <> 0 then
    Exit;
  try
    Readln(F, Line);
    Readln(F, Line);
    Words := SplitWords(Line);
    if Length(Words) < 2 then
      Exit;
    Total := StrToIntDef(Words[0], 0);
    DataType := Words[1];
    if Length(Words) >= 3 then
      Target := Words[2];
    if Total <= 0 then
      Exit;
    SetLength(Entries, Total);
    Filled := 0;
    while (Filled < Total) and not Eof(F) do
    begin
      Readln(F, Line);
      Words := SplitWords(Line);
      if Length(Words) < 4 then
        Continue;
      Entries[Filled].Name := Words[0];
      Entries[Filled].Offset := StrToInt64Def(Words[1], -1);
      Entries[Filled].ByteCount := StrToInt64Def(Words[2], -1);
      SetLength(Entries[Filled].Dims, StrToIntDef(Words[3], 0));
      for D := 0 to High(Entries[Filled].Dims) do
        Entries[Filled].Dims[D] := StrToInt64Def(Words[4 + D], 0);
      Inc(Filled);
    end;
    Result := Filled = Total;
  finally
    CloseFile(F);
  end;
end;

function TQwenModel.IndexOfParam(const Name: string): Integer;
var
  I: Integer;
begin
  for I := 0 to High(Entries) do
    if Entries[I].Name = Name then
      Exit(I);
  Result := -1;
end;

function TQwenModel.ReadFloats(const TensorHandle: TTVMFFIObjectHandle; out Values: TArray<Single>;
  const Error: PString): Boolean;
var
  Raw: PDLTensor;
  Count: Int64;
  I: Integer;
begin
  Result := False;
  SetLength(Values, 0);
  Raw := TTVMRuntime.TensorDLTensor(TensorHandle);
  if not Assigned(Raw) then
  begin
    if Assigned(Error) then Error^ := 'ReadFloats: tensor is missing';
    Exit;
  end;
  Count := 1;
  for I := 0 to Raw.ndim - 1 do
    Count := Count * Raw.shape[I];
  if Count <= 0 then
  begin
    if Assigned(Error) then Error^ := 'ReadFloats: tensor is empty';
    Exit;
  end;
  SetLength(Values, Count);
  Result := TTVMRuntime.CopyBytesFromTensor(TensorHandle, @Values[0],
    Count * SizeOf(Single), Error);
end;

function TQwenModel.Load(const ADirectory: string; const Error: PString;
  const LibraryName: string): Boolean;
var
  DataType, Target, ModuleFile: string;
  Raw: PDLTensor;
  Staging: Pointer;
  Once: TTVMFFIObjectHandle;
  Args: array of TTVMFFIAny;
  Res: TTVMFFIAny;
  I: Integer;
begin
  Result := False;
  Once := nil;
  Staging := nil;
  Directory := IncludeTrailingPathDelimiter(ADirectory);
  ParamsArray := nil;
  ModuleHandle := nil;
  VMHandle := nil;
  CacheHandle := nil;
  EmbedFunc := nil;
  PrefillFunc := nil;
  DecodeFunc := nil;
  SetLength(ParamHandles, 0);
  if not ReadParamIndex(Directory + 'params.index', Entries, DataType, Target) then
  begin
    if Assigned(Error) then Error^ := 'Could not read params.index';
    Exit;
  end;
  if DataType = 'float32' then
    DataBits := 32
  else if DataType = 'float16' then
      DataBits := 16
    else begin
      if Assigned(Error) then
        Error^ := 'Weights declare dtype ' + DataType +
          '; supported dtypes are float32 and float16';
      Exit;
    end;
  if Target = 'cuda' then
    DeviceType := Ord(kDLCUDA)
  else if Target = 'cpu' then
      DeviceType := Ord(kDLCPU)
    else begin
      if Assigned(Error) then
        Error^ := 'Parameter index declares an unknown target: ' + Target;
      Exit;
    end;
  SetLength(ParamHandles, Length(Entries));
  for I := 0 to High(Entries) do
  begin
    if not TTVMRuntime.AllocDeviceTensor(Entries[I].Dims,
      Ord(kDLFloat), DataBits, 1, DeviceType, 0, ParamHandles[I], Error) then
        Exit;
    GetMem(Staging, Entries[I].ByteCount);
    try
      if not ReadChunk(Directory + 'params.bin', Entries[I].Offset,
        Entries[I].ByteCount, Staging) then
        begin
          if Assigned(Error) then
            Error^ := 'Could not read parameter ' + Entries[I].Name;
          Exit;
        end;
      if not TTVMRuntime.CopyBytesToTensor(ParamHandles[I], Staging,
        Entries[I].ByteCount, Error) then
          Exit;
    finally
      FreeMem(Staging);
      Staging := nil;
    end;
  end;
  SetLength(Args, Length(ParamHandles));
  for I := 0 to High(ParamHandles) do
    Args[I] := TTVMRuntime.AnyObject(ParamHandles[I]);
  if not TTVMRuntime.CallGlobal('ffi.Array', Args, Res, Error) then
    Exit;
  ParamsArray := Res.VObj;
  ModuleFile := LibraryName;
  if ModuleFile = '' then ModuleFile := 'qwen2-' + Target + '.dll';
  if not TTVMRuntime.ModuleLoadFromFile(Directory + ModuleFile,
    'dll', ModuleHandle, Error) then
      Exit;
  if not TTVMRuntime.ModuleGetFunction(ModuleHandle, 'vm_load_executable', True, Once, Error) then
    Exit;
  try
    if not TTVMRuntime.CallFunc(Once, [], Res, Error) then
      Exit;
  finally
    TTVMRuntime.ObjectRelease(Once);
  end;
  VMHandle := Res.VObj;
  if not TTVMRuntime.ModuleGetFunction(VMHandle, 'vm_initialization', True, Once, Error) then
    Exit;
  try
    if DeviceType = Ord(kDLCPU) then
    begin
      if not TTVMRuntime.CallFunc(Once,
        [TTVMRuntime.AnyInt(DeviceType), TTVMRuntime.AnyInt(0),
         TTVMRuntime.AnyInt(kTVMVMAllocPooled)], Res, Error) then
          Exit;
    end
    else begin
      if not TTVMRuntime.CallFunc(Once,
        [TTVMRuntime.AnyInt(DeviceType), TTVMRuntime.AnyInt(0),
         TTVMRuntime.AnyInt(kTVMVMAllocNaive),
         TTVMRuntime.AnyInt(Ord(kDLCPU)), TTVMRuntime.AnyInt(0),
         TTVMRuntime.AnyInt(kTVMVMAllocPooled)], Res, Error) then
          Exit;
    end;
  finally
    TTVMRuntime.ObjectRelease(Once);
  end;
  if not TTVMRuntime.ModuleGetFunction(VMHandle, 'embed', True, EmbedFunc, Error) or
    not TTVMRuntime.ModuleGetFunction(VMHandle, 'prefill', True, PrefillFunc, Error) or
    not TTVMRuntime.ModuleGetFunction(VMHandle, 'decode', True, DecodeFunc, Error) then
      Exit;
  Result := True;
end;

procedure TQwenModel.Unload;
var
  I: Integer;
begin
  if Assigned(DecodeFunc) then TTVMRuntime.ObjectRelease(DecodeFunc);
  if Assigned(PrefillFunc) then TTVMRuntime.ObjectRelease(PrefillFunc);
  if Assigned(EmbedFunc) then TTVMRuntime.ObjectRelease(EmbedFunc);
  if Assigned(CacheHandle) then TTVMRuntime.ObjectRelease(CacheHandle);
  if Assigned(VMHandle) then TTVMRuntime.ObjectRelease(VMHandle);
  if Assigned(ModuleHandle) then TTVMRuntime.ObjectRelease(ModuleHandle);
  if Assigned(ParamsArray) then TTVMRuntime.ObjectRelease(ParamsArray);
  for I := 0 to High(ParamHandles) do
    if Assigned(ParamHandles[I]) then
      TTVMRuntime.ObjectRelease(ParamHandles[I]);
  SetLength(ParamHandles, 0);
end;

function TQwenModel.CreateCache(const MaxBatch, MaxTotal, Chunk, Page: Int64;
  const Error: PString): Boolean;
var
  ShapeBatch, ShapeTotal, ShapeChunk, ShapePage, CreateFunc: TTVMFFIObjectHandle;
  Res: TTVMFFIAny;
begin
  Result := False;
  ShapeBatch := nil;
  ShapeTotal := nil;
  ShapeChunk := nil;
  ShapePage := nil;
  CreateFunc := nil;
  try
    if not MakeShape([MaxBatch], ShapeBatch, Error) or not MakeShape([MaxTotal], ShapeTotal, Error) or
      not MakeShape([Chunk], ShapeChunk, Error) or not MakeShape([Page], ShapePage, Error) then
        Exit;
    if not TTVMRuntime.ModuleGetFunction(VMHandle, 'create_tir_paged_kv_cache',
      True, CreateFunc, Error) then
        Exit;
    if not TTVMRuntime.CallFunc(CreateFunc,
      [TTVMRuntime.AnyObject(ShapeBatch), TTVMRuntime.AnyObject(ShapeTotal),
       TTVMRuntime.AnyObject(ShapeChunk), TTVMRuntime.AnyObject(ShapePage)],
      Res, Error) then
        Exit;
    if Res.TypeIndex < kTVMFFIStaticObjectBegin then
    begin
      if Assigned(Error) then
        Error^ := Format('Cache result is not an object; type_index %d', [Res.TypeIndex]);
      Exit;
    end;
    CacheHandle := Res.VObj;
    Result := True;
  finally
    if Assigned(CreateFunc) then TTVMRuntime.ObjectRelease(CreateFunc);
    if Assigned(ShapePage) then TTVMRuntime.ObjectRelease(ShapePage);
    if Assigned(ShapeChunk) then TTVMRuntime.ObjectRelease(ShapeChunk);
    if Assigned(ShapeTotal) then TTVMRuntime.ObjectRelease(ShapeTotal);
    if Assigned(ShapeBatch) then TTVMRuntime.ObjectRelease(ShapeBatch);
  end;
end;

function TQwenModel.AddSequence(const SeqId: Int64; const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
begin
  Result := TTVMRuntime.CallGlobal('vm.builtin.kv_state_add_sequence',
    [TTVMRuntime.AnyObject(CacheHandle), TTVMRuntime.AnyInt(SeqId)], Res, Error);
end;

function TQwenModel.RemoveSequence(const SeqId: Int64; const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
begin
  Result := TTVMRuntime.CallGlobal('vm.builtin.kv_state_remove_sequence',
    [TTVMRuntime.AnyObject(CacheHandle), TTVMRuntime.AnyInt(SeqId)], Res, Error);
end;

function TQwenModel.ClearCache(const Error: PString): Boolean;
var
  Res: TTVMFFIAny;
begin
  Result := TTVMRuntime.CallGlobal('vm.builtin.kv_state_clear',
    [TTVMRuntime.AnyObject(CacheHandle)], Res, Error);
end;

function TQwenModel.Embed(const Tokens: array of Integer; out TensorHandle: TTVMFFIObjectHandle;
  const Error: PString): Boolean;
var
  Shape: array[0..1] of Int64;
  Cells: TArray<Integer>;
  Res: TTVMFFIAny;
  Token: TTVMFFIObjectHandle;
  I: Integer;
begin
  Result := False;
  TensorHandle := nil;
  Token := nil;
  Shape[0] := 1;
  Shape[1] := Length(Tokens);
  SetLength(Cells, Length(Tokens));
  for I := 0 to High(Tokens) do
    Cells[I] := Tokens[I];
  if not TTVMRuntime.AllocDeviceTensor(Shape, Ord(kDLInt), 32, 1, DeviceType, 0, Token, Error) then
    Exit;
  if not TTVMRuntime.CopyBytesToTensor(Token, @Cells[0],
    Length(Tokens) * SizeOf(Integer), Error) then
    begin
      TTVMRuntime.ObjectRelease(Token);
      Exit;
    end;
  try
    if not TTVMRuntime.CallFunc(EmbedFunc,
      [TTVMRuntime.AnyObject(Token), TTVMRuntime.AnyObject(ParamsArray)],
      Res, Error) then
        Exit;
    if Res.TypeIndex <> kTVMFFITensor then
    begin
      if Assigned(Error) then
        Error^ := Format('embed returned type_index %d instead of a tensor', [Res.TypeIndex]);
      Exit;
    end;
    TensorHandle := Res.VObj;
    Result := True;
  finally
    TTVMRuntime.ObjectRelease(Token);
  end;
end;

function TQwenModel.Run(const Which: string; const EmbedTensor: TTVMFFIObjectHandle;
  const SeqId, Count: Int64; out LogitsHandle: TTVMFFIObjectHandle; const Error: PString): Boolean;
var
  SeqIds, Lengths: TTVMFFIObjectHandle;
  Func: TTVMFFIObjectHandle;
  Res, Answer: TTVMFFIAny;
begin
  Result := False;
  LogitsHandle := nil;
  SeqIds := nil;
  Lengths := nil;
  FillChar(Answer, SizeOf(Answer), 0);
  if Which = 'prefill' then
    Func := PrefillFunc
  else
    Func := DecodeFunc;
  try
    if not MakeShape([SeqId], SeqIds, Error) or not MakeShape([Count], Lengths, Error) then
      Exit;
    if not TTVMRuntime.CallGlobal('vm.builtin.kv_state_begin_forward',
      [TTVMRuntime.AnyObject(CacheHandle), TTVMRuntime.AnyObject(SeqIds),
       TTVMRuntime.AnyObject(Lengths)], Res, Error) then
        Exit;
    if not TTVMRuntime.CallFunc(Func,
      [TTVMRuntime.AnyObject(EmbedTensor), TTVMRuntime.AnyObject(CacheHandle),
       TTVMRuntime.AnyObject(ParamsArray)], Answer, Error) then
        Exit;
    if not TTVMRuntime.CallGlobal('vm.builtin.kv_state_end_forward',
      [TTVMRuntime.AnyObject(CacheHandle)], Res, Error) then
        Exit;
    if Answer.TypeIndex = kTVMFFITensor then
    begin
      LogitsHandle := Answer.VObj;
      Answer.VObj := nil;
    end
    else if Answer.TypeIndex >= kTVMFFIStaticObjectBegin then
      begin
        if not TTVMRuntime.CallGlobal('ffi.ArrayGetItem',
          [TTVMRuntime.AnyObject(Answer.VObj), TTVMRuntime.AnyInt(0)],
          Res, Error) then
            Exit;
        if Res.TypeIndex <> kTVMFFITensor then
        begin
          if Assigned(Error) then
            Error^ := Format('First tuple element has type_index %d instead of a tensor',
              [Res.TypeIndex]);
          Exit;
        end;
        LogitsHandle := Res.VObj;
      end
      else begin
        if Assigned(Error) then
          Error^ := Format('%s returned type_index %d', [Which, Answer.TypeIndex]);
        Exit;
      end;
    Result := True;
  finally
    if Answer.TypeIndex >= kTVMFFIStaticObjectBegin then
      TTVMRuntime.ObjectRelease(Answer.VObj);
    if Assigned(Lengths) then TTVMRuntime.ObjectRelease(Lengths);
    if Assigned(SeqIds) then TTVMRuntime.ObjectRelease(SeqIds);
  end;
end;

end.
