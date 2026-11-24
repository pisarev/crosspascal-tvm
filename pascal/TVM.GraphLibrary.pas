{ ************************************************************************** }
{                                                                            }
{ TVM.GraphLibrary                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.GraphLibrary;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM;

type
  EGraphLibrary = class(Exception);
  TGraphLibraryLease = record
    Module: TTVMFFIObjectHandle;
    Token: Pointer;
  end;

{ One FFI library context per physical loaded library. VMs remain separate.
  Acquire/release are paired and serialized; the final client releases both
  the FFI import tree and the native library pin. }
function AcquireGraphLibrary(const FileName: string): TGraphLibraryLease;
procedure ReleaseGraphLibrary(var Lease: TGraphLibraryLease);

implementation

uses
  {$IFDEF FPC}{$IFNDEF UNIX}Windows,{$ENDIF}SyncObjs
  {$ELSE}Winapi.Windows, System.SyncObjs{$ENDIF};

{$IFDEF UNIX}
function OpenLibrary(Name: PAnsiChar; Flags: Integer): Pointer; cdecl; external 'dl' name 'dlopen';
function CloseLibrary(Handle: Pointer): Integer; cdecl; external 'dl' name 'dlclose';
{$ENDIF}

type
  TNativeLibrary = {$IFDEF UNIX}Pointer{$ELSE}HMODULE{$ENDIF};
  TGraphLibrary = class
    Native: TNativeLibrary;
    Module: TTVMFFIObjectHandle;
    Clients: Integer;
    destructor Destroy; override;
  end;

var
  LibraryLock: TCriticalSection;
  Libraries: TArray<TGraphLibrary>;

procedure ReleaseNative(const Handle: TNativeLibrary);
begin
  if Handle <> {$IFDEF UNIX}nil{$ELSE}0{$ENDIF} then
    {$IFDEF UNIX}CloseLibrary(Handle){$ELSE}FreeLibrary(Handle){$ENDIF};
end;

destructor TGraphLibrary.Destroy;
begin
  TTVMRuntime.ObjectRelease(Module);
  ReleaseNative(Native);
  inherited;
end;

function AcquireGraphLibrary(const FileName: string): TGraphLibraryLease;
var
  Native: TNativeLibrary;
  Entry: TGraphLibrary;
  Error: {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
  I, At: Integer;
  FullName: string;
begin
  Result := Default(TGraphLibraryLease);
  FullName := ExpandFileName(FileName);
  LibraryLock.Enter;
  try
    Native := {$IFDEF UNIX}OpenLibrary(PAnsiChar(UTF8Encode(FullName)), 2)
      {$ELSE}LoadLibraryExW(PWideChar(FullName), 0, $900){$ENDIF};
    if Native = {$IFDEF UNIX}nil{$ELSE}0{$ENDIF} then
      raise EGraphLibrary.Create('Cannot load graph library: ' + FullName);
    try
      for I := 0 to High(Libraries) do
        if Libraries[I].Native = Native then
        begin
          Inc(Libraries[I].Clients);
          Result.Module := Libraries[I].Module;
          Result.Token := Libraries[I];
          Exit;
        end;
      Entry := TGraphLibrary.Create;
      try
        Entry.Native := Native;
        Native := {$IFDEF UNIX}nil{$ELSE}0{$ENDIF};
        if not TTVMRuntime.ModuleLoadFromFile({$IFDEF UNIX}UTF8Encode(FullName){$ELSE}FullName{$ENDIF},
          {$IFDEF UNIX}'so'{$ELSE}'dll'{$ENDIF}, Entry.Module, @Error) then
            raise EGraphLibrary.Create('Cannot initialize graph library: ' + string(Error));
        Entry.Clients := 1;
        At := Length(Libraries);
        SetLength(Libraries, At + 1);
        Libraries[At] := Entry;
        Result.Module := Entry.Module;
        Result.Token := Entry;
      except
        Entry.Free;
        raise;
      end;
    finally
      ReleaseNative(Native);
    end;
  finally
    LibraryLock.Leave;
  end;
end;

procedure ReleaseGraphLibrary(var Lease: TGraphLibraryLease);
var
  I, J: Integer;
  Entry: TGraphLibrary;
begin
  if Lease.Token = nil then Exit;
  LibraryLock.Enter;
  try
    for I := 0 to High(Libraries) do
      if Pointer(Libraries[I]) = Lease.Token then
      begin
        Entry := Libraries[I];
        Lease := Default(TGraphLibraryLease);
        Dec(Entry.Clients);
        if Entry.Clients = 0 then
        begin
          for J := I to High(Libraries) - 1 do Libraries[J] := Libraries[J + 1];
          SetLength(Libraries, Length(Libraries) - 1);
          Entry.Free;
        end;
        Exit;
      end;
    raise EGraphLibrary.Create('Unknown graph library lease');
  finally
    LibraryLock.Leave;
  end;
end;

initialization
  LibraryLock := TCriticalSection.Create;
finalization
  LibraryLock.Free;
end.
