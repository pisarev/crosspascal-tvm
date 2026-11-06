{ ************************************************************************** }
{                                                                            }
{ CrossPascal.WinMemory                                                      }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.WinMemory;

{$IFDEF FPC}
{$MODE DELPHI}
{$ENDIF}

interface

{$IFDEF FPC}
uses
  Windows;

type
  TMemoryStatusEx = record
    dwLength: DWORD;
    dwMemoryLoad: DWORD;
    ullTotalPhys: UInt64;
    ullAvailPhys: UInt64;
    ullTotalPageFile: UInt64;
    ullAvailPageFile: UInt64;
    ullTotalVirtual: UInt64;
    ullAvailVirtual: UInt64;
    ullAvailExtendedVirtual: UInt64;
  end;

function GlobalMemoryStatusEx(var Status: TMemoryStatusEx): BOOL; stdcall; external 'kernel32.dll';
{$ENDIF}

implementation

end.
