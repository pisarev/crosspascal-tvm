{ ************************************************************************** }
{                                                                            }
{ TVM.WindowsPaths                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.WindowsPaths;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}Windows, SysUtils{$ELSE}Winapi.Windows, System.SysUtils{$ENDIF};

type
  TWindowsModelDirectory = class
  private
    FHandle: THandle;
    FPath: string;
    FOriginal: string;
  public
    constructor Create(const Directory: string);
    destructor Destroy; override;
    property Path: string read FPath;
    property Original: string read FOriginal;
  end;

function ModelRuntimeDirectory(const Directory: string): string;

implementation

type
  TNativeUnicodeString = record
    Length: Word;
    MaximumLength: Word;
    Buffer: PWideChar;
  end;
  PNativeUnicodeString = ^TNativeUnicodeString;
  TNativeObjectAttributes = record
    Length: Cardinal;
    RootDirectory: THandle;
    ObjectName: PNativeUnicodeString;
    Attributes: Cardinal;
    SecurityDescriptor: Pointer;
    SecurityQualityOfService: Pointer;
  end;

function NtCreateSymbolicLinkObject(out Handle: THandle; Access: Cardinal;
  var Attributes: TNativeObjectAttributes; var Target: TNativeUnicodeString): LongInt;
  stdcall; external 'ntdll.dll';
function CompareStringOrdinal(First: PWideChar; FirstLength: Integer; Second: PWideChar;
  SecondLength, IgnoreCase: Integer): Integer;
  stdcall; external 'kernel32.dll';
function NativeFullPath(FileName: PWideChar; BufferLength: Cardinal; Buffer: PWideChar;
  FilePart: Pointer): Cardinal;
  stdcall; external 'kernel32.dll' name 'GetFullPathNameW';

const
  OrdinalStringsEqual = 2;

procedure CheckPath(const Condition: Boolean; const MessageText: string);
begin
  if not Condition then raise Exception.Create(MessageText);
end;

function AbsoluteDirectory(const Directory: string): string;
var
  Required, Count, Attributes: Cardinal;
begin
  CheckPath((Directory <> '') and (Pos(#0, Directory) = 0), 'Choose a model or runtime folder.');
  Required := NativeFullPath(PWideChar(Directory), 0, nil, nil);
  CheckPath((Required > 0) and (Required <= 32767), 'The folder path could not be resolved.');
  SetLength(Result, Required);
  Count := NativeFullPath(PWideChar(Directory), Required, PWideChar(Result), nil);
  CheckPath((Count > 0) and (Count < Required), 'The folder path changed while resolving it.');
  SetLength(Result, Count);
  Attributes := GetFileAttributesW(PWideChar(Result));
  CheckPath((Attributes <> INVALID_FILE_ATTRIBUTES) and
    ((Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0), 'The model or runtime folder was not found.');
end;

function NeedsAlias(const Path: string): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to Length(Path) do if Ord(Path[I]) > 127 then Exit(True);
end;

function NativeString(const Value: string): TNativeUnicodeString;
begin
  CheckPath(Length(Value) <= 32766, 'The folder path is too long.');
  Result.Length := Length(Value) * SizeOf(WideChar);
  Result.MaximumLength := Result.Length + SizeOf(WideChar);
  Result.Buffer := PWideChar(Value);
end;

constructor TWindowsModelDirectory.Create(const Directory: string);
var
  ID: TGUID;
  Name, Target: string;
  NativeName, NativeTarget: TNativeUnicodeString;
  Attributes: TNativeObjectAttributes;
  Status: LongInt;
begin
  inherited Create;
  FOriginal := AbsoluteDirectory(Directory);
  FPath := FOriginal;
  if not NeedsAlias(FOriginal) then Exit;
  CheckPath((SizeOf(TNativeObjectAttributes) = 48) and
    (SizeOf(TNativeUnicodeString) = 16), 'The Windows path interface requires an x64 build.');
  CheckPath(CreateGUID(ID) = 0, 'A temporary model path could not be created.');
  Name := '\??\CrossPascal-' + GUIDToString(ID);
  if Copy(FOriginal, 1, 4) = '\\?\' then
    Target := '\??\' + Copy(FOriginal, 5, MaxInt)
  else if Copy(FOriginal, 1, 2) = '\\' then
      Target := '\??\UNC\' + Copy(FOriginal, 3, MaxInt)
    else
      Target := '\??\' + FOriginal;
  NativeName := NativeString(Name);
  NativeTarget := NativeString(Target);
  FillChar(Attributes, SizeOf(Attributes), 0);
  Attributes.Length := SizeOf(Attributes);
  Attributes.ObjectName := @NativeName;
  Attributes.Attributes := $40;
  Status := NtCreateSymbolicLinkObject(FHandle, 1, Attributes, NativeTarget);
  CheckPath(Status >= 0,
    'This folder cannot be used by the model loader. Choose a folder with an ASCII path. Windows status ' + IntToHex(Cardinal(Status), 8));
  FPath := '\\?\CrossPascal-' + GUIDToString(ID);
end;

destructor TWindowsModelDirectory.Destroy;
begin
  if FHandle <> 0 then CloseHandle(FHandle);
  inherited;
end;

function ModelRuntimeDirectory(const Directory: string): string;
const
  Libraries: array[0..3] of string = ('tvm_ffi.dll', 'tvm_runtime.dll',
    'tvm_runtime_extra.dll', 'tvm_runtime_cuda.dll');
var
  Actual, Expected: string;
  Handle: THandle;
  Count: Cardinal;
  I: Integer;
begin
  Result := AbsoluteDirectory(Directory);
  if (Result[Length(Result)] <> '\') and (Result[Length(Result)] <> '/') then
    Result := Result + '\';
  for I := Low(Libraries) to High(Libraries) do
  begin
    Handle := GetModuleHandleW(PWideChar(Libraries[I]));
    if Handle = 0 then Continue;
    SetLength(Actual, 32768);
    Count := GetModuleFileNameW(Handle, PWideChar(Actual), Length(Actual));
    CheckPath((Count > 0) and (Count < Cardinal(Length(Actual))),
      'The loaded runtime path could not be checked.');
    SetLength(Actual, Count);
    Expected := Result + Libraries[I];
    CheckPath(CompareStringOrdinal(PWideChar(Actual), Length(Actual),
      PWideChar(Expected), Length(Expected), 1) = OrdinalStringsEqual,
      'Restart the application before selecting a different runtime folder.');
  end;
end;

end.
