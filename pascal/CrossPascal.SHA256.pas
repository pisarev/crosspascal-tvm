{ ************************************************************************** }
{                                                                            }
{ CrossPascal.SHA256                                                         }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.SHA256;

{$IFDEF FPC}
{$MODE DELPHIUNICODE}
{$ENDIF}

interface

uses
  {$IFDEF FPC}Classes, SysUtils{$ELSE}System.Classes, System.SysUtils{$ENDIF};

function SHA256Stream(const Stream: TStream): UnicodeString;
function SHA256Bytes(const Bytes: TBytes): UnicodeString;

implementation

{$IFDEF MSWINDOWS}
function BCryptOpenAlgorithmProvider(out Algorithm: Pointer; AlgorithmID, ProviderName: PWideChar;
  Flags: Cardinal): LongInt; stdcall; external 'bcrypt.dll';
function BCryptGetProperty(Handle: Pointer; PropertyName: PWideChar; Output: PByte;
  OutputSize: Cardinal; out ResultSize: Cardinal; Flags: Cardinal): LongInt; stdcall;
  external 'bcrypt.dll';
function BCryptCreateHash(Algorithm: Pointer; out Hash: Pointer; ObjectBuffer: PByte;
  ObjectSize: Cardinal; Secret: PByte; SecretSize, Flags: Cardinal): LongInt; stdcall;
  external 'bcrypt.dll';
function BCryptHashData(Hash: Pointer; Input: PByte; Size, Flags: Cardinal): LongInt; stdcall;
  external 'bcrypt.dll';
function BCryptFinishHash(Hash: Pointer; Output: PByte; Size, Flags: Cardinal): LongInt; stdcall;
  external 'bcrypt.dll';
function BCryptDestroyHash(Hash: Pointer): LongInt; stdcall; external 'bcrypt.dll';
function BCryptCloseAlgorithmProvider(Algorithm: Pointer; Flags: Cardinal): LongInt; stdcall;
  external 'bcrypt.dll';

procedure Check(const Status: LongInt; const Operation: UnicodeString);
begin
  if Status < 0 then
    raise Exception.Create('SHA-256 ' + Operation + ' failed (0x' + IntToHex(Cardinal(Status), 8) + ').');
end;
{$ELSE}
{$IFDEF FPC}
uses
  Dynlibs;

function SHA256UnixStream(const Stream: TStream): UnicodeString;
type
  TNewContext = function: Pointer; cdecl;
  TFreeContext = procedure(Context: Pointer); cdecl;
  TAlgorithm = function: Pointer; cdecl;
  TInitialize = function(Context, Algorithm, Engine: Pointer): Integer; cdecl;
  TUpdate = function(Context, Data: Pointer; Size: NativeUInt): Integer; cdecl;
  TFinish = function(Context, Digest: Pointer; out Size: Cardinal): Integer; cdecl;
const
  Hex: UnicodeString = '0123456789abcdef';
var
  LibraryHandle: TLibHandle;
  NewContext: TNewContext;
  FreeContext: TFreeContext;
  Algorithm: TAlgorithm;
  Initialize: TInitialize;
  Update: TUpdate;
  Finish: TFinish;
  Context: Pointer;
  Buffer: TBytes;
  Digest: array[0..31] of Byte;
  Count, I: Integer;
  DigestSize: Cardinal;
begin
  LibraryHandle := LoadLibrary('libcrypto.so.3');
  if LibraryHandle = NilHandle then LibraryHandle := LoadLibrary('libcrypto.so.1.1');
  if LibraryHandle = NilHandle then
    raise Exception.Create('SHA-256 requires the system OpenSSL crypto library');
  try
    NewContext := TNewContext(GetProcedureAddress(LibraryHandle, 'EVP_MD_CTX_new'));
    FreeContext := TFreeContext(GetProcedureAddress(LibraryHandle, 'EVP_MD_CTX_free'));
    Algorithm := TAlgorithm(GetProcedureAddress(LibraryHandle, 'EVP_sha256'));
    Initialize := TInitialize(GetProcedureAddress(LibraryHandle, 'EVP_DigestInit_ex'));
    Update := TUpdate(GetProcedureAddress(LibraryHandle, 'EVP_DigestUpdate'));
    Finish := TFinish(GetProcedureAddress(LibraryHandle, 'EVP_DigestFinal_ex'));
    if not (Assigned(NewContext) and Assigned(FreeContext) and Assigned(Algorithm) and
      Assigned(Initialize) and Assigned(Update) and Assigned(Finish)) then
        raise Exception.Create('SHA-256 OpenSSL entry points are unavailable');
    Context := NewContext();
    if Context = nil then raise Exception.Create('Cannot allocate SHA-256 context');
    try
      if Initialize(Context, Algorithm(), nil) <> 1 then
        raise Exception.Create('Cannot initialize SHA-256 context');
      SetLength(Buffer, 1024 * 1024);
      repeat
        Count := Stream.Read(Buffer[0], Length(Buffer));
        if Count > 0 then
          if Update(Context, @Buffer[0], Count) <> 1 then
            raise Exception.Create('SHA-256 stream update failed');
      until Count = 0;
      if (Finish(Context, @Digest[0], DigestSize) <> 1) or (DigestSize <> SizeOf(Digest)) then
        raise Exception.Create('SHA-256 finalization failed');
      SetLength(Result, 64);
      for I := Low(Digest) to High(Digest) do
      begin
        Result[I * 2 + 1] := Hex[(Digest[I] shr 4) + 1];
        Result[I * 2 + 2] := Hex[(Digest[I] and 15) + 1];
      end;
    finally
      FreeContext(Context);
    end;
  finally
    UnloadLibrary(LibraryHandle);
  end;
end;
{$ELSE}
uses
  System.Hash;
{$ENDIF}
{$ENDIF}

function SHA256Stream(const Stream: TStream): UnicodeString;
{$IFDEF MSWINDOWS}
const
  Hex: UnicodeString = '0123456789abcdef';
var
  Algorithm, Hash: Pointer;
  Storage, Buffer: TBytes;
  Digest: array[0..31] of Byte;
  ObjectSize, ResultSize: Cardinal;
  Count, I: Integer;
{$ENDIF}
begin
  {$IFDEF MSWINDOWS}
  Check(BCryptOpenAlgorithmProvider(Algorithm, 'SHA256', nil, 0), 'provider initialization');
  try
    Check(BCryptGetProperty(Algorithm, 'ObjectLength', PByte(@ObjectSize), SizeOf(ObjectSize),
      ResultSize, 0), 'object length query');
    if (ResultSize <> SizeOf(ObjectSize)) or (ObjectSize = 0) then
      raise Exception.Create('Invalid SHA-256 object length.');
    SetLength(Storage, ObjectSize);
    SetLength(Buffer, 1024 * 1024);
    Check(BCryptCreateHash(Algorithm, Hash, @Storage[0], ObjectSize, nil, 0, 0), 'hash creation');
    try
      repeat
        Count := Stream.Read(Buffer[0], Length(Buffer));
        if Count > 0 then
          Check(BCryptHashData(Hash, @Buffer[0], Count, 0), 'stream update');
      until Count = 0;
      Check(BCryptFinishHash(Hash, @Digest[0], SizeOf(Digest), 0), 'finalization');
      SetLength(Result, 64);
      for I := Low(Digest) to High(Digest) do
      begin
        Result[I * 2 + 1] := Hex[(Digest[I] shr 4) + 1];
        Result[I * 2 + 2] := Hex[(Digest[I] and 15) + 1];
      end;
    finally
      BCryptDestroyHash(Hash);
    end;
  finally
    BCryptCloseAlgorithmProvider(Algorithm, 0);
  end;
  {$ELSE}
  {$IFDEF FPC}
  Result := SHA256UnixStream(Stream);
  {$ELSE}
  Result := LowerCase(THashSHA2.GetHashString(Stream));
  {$ENDIF}
  {$ENDIF}
end;

function SHA256Bytes(const Bytes: TBytes): UnicodeString;
var
  Stream: TBytesStream;
begin
  Stream := TBytesStream.Create(Bytes);
  try
    Result := SHA256Stream(Stream);
  finally
    Stream.Free;
  end;
end;

end.
