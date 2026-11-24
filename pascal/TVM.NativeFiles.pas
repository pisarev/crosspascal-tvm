{ ************************************************************************** }
{                                                                            }
{ TVM.NativeFiles                                                            }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.NativeFiles;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.Json;

type
  ENativePackage = class(EArgumentException);
  INativeFileLease = interface
    ['{3658918E-7104-4C61-BF46-84183C178C4E}']
    function FileName: string;
    function SHA256: string;
    function ByteCount: Int64;
    function ReadBytes(const Offset: Int64; const Count: Integer): TBytes;
  end;

procedure NativeRequire(const Condition: Boolean; const MessageText: string);
procedure NativeRequireSHA256(const Value: string);
function NativeMember(const Directory, Member: string): string;
function HoldNativeFile(const Directory, Member, ExpectedSHA256: string;
  const ExpectedBytes: Int64): INativeFileLease;
function ReadNativeObject(const Lease: INativeFileLease;
  const MaximumBytes: Integer = 1024 * 1024): TJSONObject;
procedure NativeUniqueKeys(const Value: TJSONValue);
function NativeObject(const Value: TJSONValue): TJSONObject;
function NativeArray(const Value: TJSONValue): TJSONArray;
function NativeField(const Obj: TJSONObject; const Name: string): TJSONValue;
function NativeText(const Obj: TJSONObject; const Name: string): string;
function NativeInteger(const Obj: TJSONObject; const Name: string): Int64;
function NativeHostOS: string;
procedure RequireNativeX64Library(const Lease: INativeFileLease);
procedure RequireNativeX64LibraryForOS(const Lease: INativeFileLease; const TargetOS: string);

implementation

uses
  {$IFDEF FPC}Classes, SyncObjs{$ELSE}System.Classes, System.SyncObjs{$ENDIF},
  CrossPascal.SHA256, CrossPascal.NativeIO;

type
  TNativeFileLease = class(TInterfacedObject, INativeFileLease)
  private
    FStream: TStream;
    FFileName: string;
    FSHA256: string;
    FByteCount: Int64;
    FLock: TCriticalSection;
  public
    constructor Create(const Name, ExpectedSHA256: string; const ExpectedBytes: Int64);
    destructor Destroy; override;
    function FileName: string;
    function SHA256: string;
    function ByteCount: Int64;
    function ReadBytes(const Offset: Int64; const Count: Integer): TBytes;
  end;

procedure NativeRequire(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise ENativePackage.Create(MessageText);
end;

procedure NativeRequireSHA256(const Value: string);
var
  C: Char;
begin
  NativeRequire(Length(Value) = 64, 'Invalid native package SHA256');
  for C in Value do
    NativeRequire(CharInSet(C, ['0'..'9', 'a'..'f']), 'Invalid native package SHA256');
end;

function NativeMember(const Directory, Member: string): string;
begin
  NativeRequire((Member <> '') and (Member <> '.') and (Member <> '..') and
    (Member = Trim(Member)) and (Pos('/', Member) = 0) and (Pos('\', Member) = 0) and
    (Pos(':', Member) = 0) and (Pos(#0, Member) = 0) and (Pos(#9, Member) = 0) and
    (Pos(#10, Member) = 0) and (Pos(#13, Member) = 0), 'Expected a native package leaf filename');
  NativeRequire(Member[Length(Member)] <> '.', 'Invalid native package filename suffix');
  NativeRequire(DirectoryExists(Directory), 'Native package directory does not exist');
  Result := IncludeTrailingPathDelimiter(ExpandFileName(Directory)) + Member;
end;

constructor TNativeFileLease.Create(const Name, ExpectedSHA256: string; const ExpectedBytes: Int64);
begin
  inherited Create;
  NativeRequireSHA256(ExpectedSHA256);
  NativeRequire(ExpectedBytes > 0, 'Invalid native package file size');
  FLock := TCriticalSection.Create;
  FFileName := Name;
  FStream := TNativeReadStream.Create(Name);
  FByteCount := FStream.Size;
  NativeRequire(FByteCount = ExpectedBytes, 'Native package file size mismatch: ' + ExtractFileName(Name));
  FSHA256 := SHA256Stream(FStream);
  NativeRequire(FSHA256 = ExpectedSHA256, 'Native package SHA256 mismatch: ' + ExtractFileName(Name));
end;

destructor TNativeFileLease.Destroy;
begin
  FStream.Free; FLock.Free; inherited;
end;

function TNativeFileLease.FileName: string;
begin
  Result := FFileName;
end;

function TNativeFileLease.SHA256: string;
begin
  Result := FSHA256;
end;

function TNativeFileLease.ByteCount: Int64;
begin
  Result := FByteCount;
end;

function TNativeFileLease.ReadBytes(const Offset: Int64; const Count: Integer): TBytes;
begin
  NativeRequire((Offset >= 0) and (Count >= 0) and (Offset <= FByteCount) and
    (Count <= FByteCount - Offset), 'Native package read exceeds its file');
  FLock.Enter;
  try
    SetLength(Result, Count);
    FStream.Position := Offset;
    if Count > 0 then FStream.ReadBuffer(Result[0], Count);
  finally
    FLock.Leave;
  end;
end;

function HoldNativeFile(const Directory, Member, ExpectedSHA256: string;
  const ExpectedBytes: Int64): INativeFileLease;
begin
  Result := TNativeFileLease.Create(NativeMember(Directory, Member), ExpectedSHA256, ExpectedBytes);
end;

procedure NativeUniqueKeys(const Value: TJSONValue);
var
  Obj: TJSONObject;
  Rows: TJSONArray;
  I, J: Integer;
begin
  if Value is TJSONObject then
  begin
    Obj := TJSONObject(Value);
    for I := 0 to Obj.Count - 1 do
    begin
      for J := 0 to I - 1 do
        NativeRequire(JSONObjectName(Obj, I) <> JSONObjectName(Obj, J),
          'Duplicate native package JSON field');
      NativeUniqueKeys(JSONObjectValue(Obj, I));
    end;
  end
  else if Value is TJSONArray then
  begin
    Rows := TJSONArray(Value);
    for I := 0 to Rows.Count - 1 do NativeUniqueKeys(Rows.Items[I]);
  end;
end;

function NativeObject(const Value: TJSONValue): TJSONObject;
begin
  NativeRequire(Value is TJSONObject, 'Expected a native package object');
  Result := TJSONObject(Value);
end;

function NativeArray(const Value: TJSONValue): TJSONArray;
begin
  NativeRequire(Value is TJSONArray, 'Expected a native package array');
  Result := TJSONArray(Value);
end;

function NativeField(const Obj: TJSONObject; const Name: string): TJSONValue;
begin
  Result := Obj.GetValue(Name); NativeRequire(Result <> nil, 'Missing native package field: ' + Name);
end;

function NativeText(const Obj: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := NativeField(Obj, Name);
  NativeRequire(Value is TJSONString, 'Expected a native package string: ' + Name);
  Result := JSONValueText(Value);
end;

function NativeInteger(const Obj: TJSONObject; const Name: string): Int64;
var
  Value: TJSONValue;
begin
  Value := NativeField(Obj, Name);
  NativeRequire((Value is TJSONNumber) and TryStrToInt64(JSONValueText(Value), Result),
    'Expected a native package integer: ' + Name);
end;

function ReadNativeObject(const Lease: INativeFileLease; const MaximumBytes: Integer): TJSONObject;
var
  Bytes: TBytes;
  Encoded: UTF8String;
  Value: TJSONValue;
begin
  NativeRequire((Lease <> nil) and (MaximumBytes > 0) and (Lease.ByteCount <= MaximumBytes),
    'Native package JSON exceeds its byte limit');
  Bytes := Lease.ReadBytes(0, Lease.ByteCount); SetLength(Encoded, Length(Bytes));
  if Length(Bytes) > 0 then Move(Bytes[0], Encoded[1], Length(Bytes));
  Value := ParseCrossJSON(UTF8ToString(Encoded));
  try
    Result := NativeObject(Value);
    NativeUniqueKeys(Result);
    Value := nil;
  finally
    Value.Free;
  end;
end;

function NativeHostOS: string;
begin
  Result := {$IFDEF MSWINDOWS}'windows'{$ELSE}'linux'{$ENDIF};
end;

procedure RequireNativeX64Library(const Lease: INativeFileLease);
begin
  RequireNativeX64LibraryForOS(Lease, NativeHostOS);
end;

procedure RequireNativeX64LibraryForOS(const Lease: INativeFileLease; const TargetOS: string);
var
  Header: TBytes;
  Offset: Int64;
begin
  NativeRequire(SizeOf(Pointer) = 8, 'Native model libraries require a 64-bit application');
  NativeRequire((TargetOS = 'windows') or (TargetOS = 'linux'), 'Unsupported native library target OS');
  Header := Lease.ReadBytes(0, 64);
  if TargetOS = 'windows' then
  begin
    NativeRequire((Header[0] = $4D) and (Header[1] = $5A), 'Expected a Windows model library');
    Offset := Int64(Header[$3C]) or (Int64(Header[$3D]) shl 8) or
      (Int64(Header[$3E]) shl 16) or (Int64(Header[$3F]) shl 24);
    Header := Lease.ReadBytes(Offset, 26);
    NativeRequire(
      (Header[0] = $50) and (Header[1] = $45) and (Header[2] = 0) and (Header[3] = 0) and (Header[4] = $64) and (Header[5] = $86) and ((Header[23] and $20) <> 0) and (Header[24] = $0B) and (Header[25] = $02),
      'Expected an x64 PE32+ DLL'
    );
  end
  else begin
    NativeRequire(
      (Header[0] = $7F) and (Header[1] = $45) and (Header[2] = $4C) and (Header[3] = $46) and (Header[4] = 2) and (Header[5] = 1) and (Header[16] = 3) and (Header[17] = 0) and (Header[18] = 62) and (Header[19] = 0),
      'Expected an x64 little-endian ELF shared library'
    );
  end;
end;

end.
