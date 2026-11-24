unit CrossPascal.NativeIO;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}Classes, SysUtils{$ELSE}System.Classes, System.SysUtils{$ENDIF};

type
  { No process-wide code-page changes. Windows filenames/arguments are UTF-16;
    Unix filenames are UTF-8. Streams own their handles, including on errors. }
  TNativeHandleStream = class(THandleStream)
  private
    FHasHandle: Boolean;
  public
    destructor Destroy; override;
  end;
  TNativeReadStream = class(TNativeHandleStream)
  public
    constructor Create(const FileName: string);
  end;
  TNativeWriteStream = class(TNativeHandleStream)
  public
    constructor Create(const FileName: string; const Exclusive: Boolean = True);
  end;

function NativeParamStr(const Index: Integer): string;
function NativeParamCount: Integer;
function NativeEnvironmentVariable(const Name: string): string;
{ Writes exact UTF-8 bytes to a standard stream, including redirected pipes.
  The process locale and console code page are not changed. }
procedure NativeWriteUTF8(const Value: string; const ErrorOutput: Boolean = False);

implementation

uses
  {$IFDEF UNIX}BaseUnix{$ELSE}{$IFDEF FPC}Windows{$ELSE}Winapi.Windows{$ENDIF}{$ENDIF};

{$IFNDEF UNIX}
type
  TWideArguments = array[0..65535] of PWideChar;
  PWideArguments = ^TWideArguments;

function ParseWideArguments(Line: PWideChar; out Count: Integer): PWideArguments;
  stdcall; external 'shell32.dll' name 'CommandLineToArgvW';
{$ENDIF}

destructor TNativeHandleStream.Destroy;
begin
  if FHasHandle then
  begin
  {$IFDEF UNIX}
  if Handle >= 0 then fpClose(Handle);
  {$ELSE}
  if (Handle <> 0) and (Handle <> INVALID_HANDLE_VALUE) then CloseHandle(Handle);
  {$ENDIF}
  end;
  inherited;
end;

constructor TNativeReadStream.Create(const FileName: string);
var
  FileHandle: THandle;
  {$IFDEF UNIX}
  Info: Stat;{$ENDIF}
begin
  if (FileName = '') or (Pos(#0, FileName) <> 0) then
    raise EArgumentException.Create('Invalid disk filename.');
  {$IFDEF UNIX}
  FileHandle := fpOpen(PAnsiChar(UTF8Encode(FileName)), O_RDONLY);
  inherited Create(FileHandle);
  FHasHandle := FileHandle >= 0;
  if FileHandle < 0 then RaiseLastOSError;
  if fpFcntl(FileHandle, F_SETFD, 1) < 0 then RaiseLastOSError;
  if fpFStat(FileHandle, Info) <> 0 then RaiseLastOSError;
  if not FPS_ISREG(Info.st_mode) then raise EFOpenError.Create('Expected a disk file: ' + FileName);
  {$ELSE}
  FileHandle := CreateFileW(PWideChar(FileName), GENERIC_READ, FILE_SHARE_READ, nil,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  inherited Create(FileHandle);
  FHasHandle := FileHandle <> INVALID_HANDLE_VALUE;
  if FileHandle = INVALID_HANDLE_VALUE then RaiseLastOSError;
  if GetFileType(FileHandle) <> FILE_TYPE_DISK then
    raise EFOpenError.CreateFmt('Expected a disk file: %s', [FileName]);
  {$ENDIF}
end;

constructor TNativeWriteStream.Create(const FileName: string; const Exclusive: Boolean);
var
  FileHandle: THandle;
  {$IFDEF UNIX}
  Flags: Integer;{$ELSE}
  Disposition: DWORD;{$ENDIF}
begin
  if (FileName = '') or (Pos(#0, FileName) <> 0) then
    raise EArgumentException.Create('Invalid disk filename.');
  {$IFDEF UNIX}
  Flags := O_WRONLY or O_CREAT;
  if Exclusive then
    Flags := Flags or O_EXCL
  else
    Flags := Flags or O_TRUNC;
  FileHandle := fpOpen(PAnsiChar(UTF8Encode(FileName)), Flags, &666);
  inherited Create(FileHandle);
  FHasHandle := FileHandle >= 0;
  if FileHandle < 0 then RaiseLastOSError;
  if fpFcntl(FileHandle, F_SETFD, 1) < 0 then RaiseLastOSError;
  {$ELSE}
  Disposition := CREATE_ALWAYS;
  if Exclusive then Disposition := CREATE_NEW;
  FileHandle := CreateFileW(PWideChar(FileName), GENERIC_WRITE, 0, nil,
    Disposition, FILE_ATTRIBUTE_NORMAL, 0);
  inherited Create(FileHandle);
  FHasHandle := FileHandle <> INVALID_HANDLE_VALUE;
  if FileHandle = INVALID_HANDLE_VALUE then RaiseLastOSError;
  {$ENDIF}
end;

function NativeParamStr(const Index: Integer): string;
{$IFNDEF UNIX}
var
  Arguments: PWideArguments;
  Count: Integer;
{$ENDIF}
begin
  {$IFDEF UNIX}
  Result := UTF8Decode(System.ParamStr(Index));
  {$ELSE}
  Arguments := ParseWideArguments(GetCommandLineW, Count);
  if Arguments = nil then RaiseLastOSError;
  try
    Result := '';
    if (Index >= 0) and (Index < Count) and (Index <= High(TWideArguments)) then
      Result := Arguments^[Index];
  finally
    LocalFree(HLOCAL(Arguments));
  end;
  {$ENDIF}
end;

function NativeParamCount: Integer;
{$IFNDEF UNIX}
var
  Arguments: PWideArguments;
  Count: Integer;
{$ENDIF}
begin
  {$IFDEF UNIX}
  Result := System.ParamCount;
  {$ELSE}
  Arguments := ParseWideArguments(GetCommandLineW, Count);
  if Arguments = nil then RaiseLastOSError;
  try
    Result := Count - 1;
  finally
    LocalFree(HLOCAL(Arguments));
  end;
  {$ENDIF}
end;

function NativeEnvironmentVariable(const Name: string): string;
{$IFNDEF UNIX}
var
  Count, Written: DWORD;
{$ENDIF}
begin
  {$IFDEF UNIX}
  Result := UTF8Decode(GetEnvironmentVariable(UTF8Encode(Name)));
  {$ELSE}
  Count := GetEnvironmentVariableW(PWideChar(Name), nil, 0);
  Result := '';
  if Count = 0 then Exit;
  SetLength(Result, Count);
  Written := GetEnvironmentVariableW(PWideChar(Name), PWideChar(Result), Count);
  if Written >= Count then
    raise EArgumentException.Create('The environment variable changed while reading.');
  SetLength(Result, Written);
  {$ENDIF}
end;

procedure NativeWriteUTF8(const Value: string; const ErrorOutput: Boolean);
var
  Bytes: UTF8String;
  FileHandle: THandle;
  At, Count, Written: NativeInt;
  {$IFNDEF UNIX}
  ChunkWritten: DWORD;{$ENDIF}
begin
  Bytes := UTF8Encode(Value);
  {$IFDEF UNIX}
  if ErrorOutput then
    FileHandle := 2
  else
    FileHandle := 1;
  {$ELSE}
  if ErrorOutput then
    FileHandle := GetStdHandle(STD_ERROR_HANDLE)
  else
    FileHandle := GetStdHandle(STD_OUTPUT_HANDLE);
  if (FileHandle = 0) or (FileHandle = INVALID_HANDLE_VALUE) then RaiseLastOSError;
  {$ENDIF}
  At := 1;
  while At <= Length(Bytes) do
  begin
    Count := Length(Bytes) - At + 1;
    if Count > High(Integer) then Count := High(Integer);
    {$IFDEF UNIX}
    Written := fpWrite(FileHandle, Bytes[At], Count);
    if (Written < 0) and (fpGetErrNo = ESysEINTR) then Continue;
    if Written < 0 then RaiseLastOSError;
    {$ELSE}
    if not WriteFile(FileHandle, Bytes[At], DWORD(Count), ChunkWritten, nil) then RaiseLastOSError;
    Written := ChunkWritten;
    {$ENDIF}
    if Written = 0 then
      raise EWriteError.CreateFmt('%s', ['Cannot write the UTF-8 standard stream.']);
    Inc(At, Written);
  end;
end;

end.
