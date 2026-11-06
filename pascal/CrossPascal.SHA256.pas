{ ************************************************************************** }
{                                                                            }
{ CrossPascal.SHA256                                                         }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
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

{$IFDEF FPC}
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
uses
  System.Hash;
{$ENDIF}

function SHA256Stream(const Stream: TStream): UnicodeString;
{$IFDEF FPC}
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
  {$IFDEF FPC}
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
  Result := LowerCase(THashSHA2.GetHashString(Stream));
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
