{ Copyright © 2026 Yuriy Pisarev. MIT license; see LICENSE. }

unit CrossPascal.SentencePiece;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils, Windows{$ELSE}System.SysUtils, Winapi.Windows{$ENDIF};

type
  TSentencePiece = class
  private type
    TLoad = function(Bytes: Pointer; Size: Integer): Pointer; cdecl;
    TFree = procedure(Processor: Pointer); cdecl;
    TID = function(Processor: Pointer; Piece: PAnsiChar): Integer; cdecl;
    TEncode = function(Processor: Pointer; Text: PAnsiChar; Size: Integer;
      Output: PInteger; Capacity: Integer): Integer; cdecl;
    TDecode = function(Processor: Pointer; IDs: PInteger; Count: Integer;
      Output: PAnsiChar; Capacity: Integer): Integer; cdecl;
  private
    FLibrary: HMODULE;
    FProcessor: Pointer;
    FFree: TFree;
    FID: TID;
    FEncode: TEncode;
    FDecode: TDecode;
    function Entry(const Name: AnsiString): Pointer;
  public
    constructor Create(const LibraryPath, ModelPath: string);
    destructor Destroy; override;
    function ID(const Piece: UTF8String): Integer;
    function Encode(const Text: string): TArray<Integer>;
    function Decode(const IDs: array of Integer): string;
  end;

implementation

uses
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF};

function TSentencePiece.Entry(const Name: AnsiString): Pointer;
begin
  Result := GetProcAddress(FLibrary, PAnsiChar(Name));
  if Result = nil then raise Exception.Create('Missing SentencePiece export: ' + string(Name));
end;

constructor TSentencePiece.Create(const LibraryPath, ModelPath: string);
var
  Bytes: TBytes;
  Stream: TFileStream;
  Load: TLoad;
begin
  inherited Create;
  FLibrary := LoadLibraryExW(PWideChar(ExpandFileName(LibraryPath)), 0, $1100);
  if FLibrary = 0 then RaiseLastOSError;
  Load := TLoad(Entry('cp_sp_load'));
  FFree := TFree(Entry('cp_sp_free'));
  FID := TID(Entry('cp_sp_id'));
  FEncode := TEncode(Entry('cp_sp_encode'));
  FDecode := TDecode(Entry('cp_sp_decode'));
  Stream := TFileStream.Create(ModelPath, fmOpenRead or fmShareDenyWrite);
  try
    if (Stream.Size <= 0) or (Stream.Size > 64 * 1024 * 1024) then
      raise Exception.Create('Invalid SentencePiece model size');
    SetLength(Bytes, Stream.Size);
    Stream.ReadBuffer(Bytes[0], Length(Bytes));
  finally
    Stream.Free;
  end;
  FProcessor := Load(Pointer(Bytes), Length(Bytes));
  if FProcessor = nil then raise Exception.Create('Could not read SentencePiece model: ' + ModelPath);
end;

destructor TSentencePiece.Destroy;
begin
  if FProcessor <> nil then FFree(FProcessor);
  if FLibrary <> 0 then FreeLibrary(FLibrary);
  inherited;
end;

function TSentencePiece.ID(const Piece: UTF8String): Integer;
begin
  Result := FID(FProcessor, PAnsiChar(Piece));
end;

function TSentencePiece.Encode(const Text: string): TArray<Integer>;
var
  Count: Integer;
  Bytes: UTF8String;
begin
  if Text = '' then Exit(nil);
  Bytes := UTF8Encode(Text);
  Count := FEncode(FProcessor, PAnsiChar(Bytes), Length(Bytes), nil, 0);
  if Count < 0 then raise Exception.Create('SentencePiece could not encode the prompt');
  SetLength(Result, Count);
  if (Count > 0) and (FEncode(FProcessor, PAnsiChar(Bytes), Length(Bytes),
    Pointer(Result), Count) <> Count) then
    raise Exception.Create('SentencePiece token count changed');
end;

function TSentencePiece.Decode(const IDs: array of Integer): string;
var
  Count: Integer;
  Bytes: UTF8String;
begin
  if Length(IDs) = 0 then Exit('');
  Count := FDecode(FProcessor, @IDs[0], Length(IDs), nil, 0);
  if Count < 0 then raise Exception.Create('SentencePiece could not decode the response');
  SetLength(Bytes, Count);
  if (Count > 0) and (FDecode(FProcessor, @IDs[0], Length(IDs), PAnsiChar(Bytes), Count) <> Count) then
    raise Exception.Create('SentencePiece response length changed');
  Result := UTF8Decode(Bytes);
end;

end.
