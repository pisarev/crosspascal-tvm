{ ************************************************************************** }
{                                                                            }
{ TVM.Threads                                                                }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Threads;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

function TVMThreadShutdownAvailable: Boolean;
function ShutdownTVMThread(out ErrorText: string): Boolean;
function FinishTVMThread(var OperationError: string): Boolean;

implementation

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM;

function TVMThreadShutdownAvailable: Boolean;
var
  Func: TTVMFFIObjectHandle;
begin
  Result := False;
  if (TTVMRuntime.TModule.RuntimeHandle = 0) or
    not Assigned(TTVMRuntime.TModule.TVMFFIFunctionCall) or
    not Assigned(TTVMRuntime.TModule.TVMFFIObjectDecRef) then Exit;
  Func := TTVMRuntime.TModule.GetGlobalFunc('runtime.ShutdownThreadPool');
  try
    Result := Func <> nil;
  finally
    TTVMRuntime.ObjectRelease(Func);
  end;
end;

function ShutdownTVMThread(out ErrorText: string): Boolean;
var
  Func: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
  CallError: {$IFDEF FPC}AnsiString{$ELSE}string{$ENDIF};
begin
  Result := False;
  ErrorText := '';
  if TTVMRuntime.TModule.RuntimeHandle = 0 then Exit(True);
  try
    if not Assigned(TTVMRuntime.TModule.TVMFFIFunctionCall) or
      not Assigned(TTVMRuntime.TModule.TVMFFIObjectDecRef) then
      begin
        ErrorText := 'TVM thread-pool shutdown functions are unavailable.';
        Exit;
      end;
    Func := TTVMRuntime.TModule.GetGlobalFunc('runtime.ShutdownThreadPool');
    try
      if Func = nil then
      begin
        ErrorText := 'The selected TVM runtime does not support safe thread shutdown.';
        Exit;
      end;
      CallError := '';
      if not TTVMRuntime.CallFunc(Func, [], Value, @CallError) then
      begin
        ErrorText := string(CallError);
        if ErrorText = '' then ErrorText := 'TVM thread-pool shutdown failed without a diagnostic.';
        Exit;
      end;
      if Value.TypeIndex <> kTVMFFINone then
      begin
        if Value.TypeIndex >= kTVMFFIStaticObjectBegin then TTVMRuntime.ObjectRelease(Value.VObj);
        ErrorText := 'TVM returned an invalid thread-pool shutdown result.';
        Exit;
      end;
      Result := True;
    finally
      TTVMRuntime.ObjectRelease(Func);
    end;
  except
    on E: Exception do ErrorText := E.ClassName + ': ' + E.Message;
  end;
end;

function FinishTVMThread(var OperationError: string): Boolean;
var
  Detail: string;
begin
  Result := ShutdownTVMThread(Detail);
  if Result then Exit;
  Detail := 'TVM thread cleanup failed: ' + Detail;
  if OperationError = '' then
    OperationError := Detail
  else
    OperationError := OperationError + sLineBreak + Detail;
end;

end.
