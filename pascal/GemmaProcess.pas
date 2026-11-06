{ ************************************************************************** }
{                                                                            }
{ GemmaProcess                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit GemmaProcess;

{$IFDEF FPC}
{$MODE DELPHIUNICODE}
{$ENDIF}

interface

uses
  {$IFDEF FPC}Windows, SysUtils{$ELSE}Winapi.Windows, System.SysUtils{$ENDIF},
  CrossPascal.Json, QwenCuda;

type
  TGemmaProcess = class
  private
    FProcess: THandle;
    FInput: THandle;
    FOutput: THandle;
    FPending: RawByteString;
    FRequestID: Int64;
    procedure Send(const Text: string);
  public
    destructor Destroy; override;
    procedure Start(const SDKRoot: string);
    procedure Generate(const Bundle, Prompt: string; const MaxNewTokens: Int64;
      const Plain: Boolean);
    procedure Request(const Command, Bundle, Prompt: string; const Options: TQwenGenerationOptions); overload;
    procedure Request(const Command, Bundle, Prompt: string; const Options: TQwenGenerationOptions;
      const Images: array of string); overload;
    procedure Stop;
    procedure RequestClose;
    function ReadLine(out Line: string): Boolean;
    function HasExited(out Code: Cardinal): Boolean;
    property RequestID: Int64 read FRequestID;
  end;

function FindGemmaSDK: string;

implementation

function FindGemmaSDK: string;
var
  Current, Parent: string;
  I: Integer;
begin
  Result := '';
  Current := ExcludeTrailingPathDelimiter(ExtractFilePath(ParamStr(0)));
  for I := 0 to 10 do
  begin
    if FileExists(Current + '\runtime\gemma-python\python.exe') and FileExists(Current + '\crosspascal-convert\source\pascalvision_convert\gemma_chat_worker.py') then
      Exit(Current);
    Parent := ExtractFileDir(Current);
    if SameText(Parent, Current) then Break;
    Current := Parent;
  end;
end;

procedure TGemmaProcess.Start(const SDKRoot: string);
var
  Security: TSecurityAttributes;
  Startup: TStartupInfoW;
  Info: TProcessInformation;
  ChildInput, ChildOutput: THandle;
  PythonPath, ScriptPath, Command, Root: string;
begin
  if FProcess <> 0 then Exit;
  Root := ExcludeTrailingPathDelimiter(ExpandFileName(SDKRoot));
  PythonPath := Root + '\runtime\gemma-python\python.exe';
  ScriptPath := Root + '\crosspascal-convert\source\pascalvision_convert\gemma_chat_worker.py';
  if not FileExists(PythonPath) then
    raise Exception.Create('Extract the Gemma runtime asset into the SDK root first.');
  if not FileExists(ScriptPath) then
    raise Exception.Create('The Gemma worker is missing from the SDK.');
  FillChar(Security, SizeOf(Security), 0);
  Security.nLength := SizeOf(Security);
  Security.bInheritHandle := True;
  ChildInput := 0;
  ChildOutput := 0;
  try
    if not CreatePipe(ChildInput, FInput, @Security, 0) then RaiseLastOSError;
    if not SetHandleInformation(FInput, HANDLE_FLAG_INHERIT, 0) then RaiseLastOSError;
    if not CreatePipe(FOutput, ChildOutput, @Security, 0) then RaiseLastOSError;
    if not SetHandleInformation(FOutput, HANDLE_FLAG_INHERIT, 0) then RaiseLastOSError;
    FillChar(Startup, SizeOf(Startup), 0);
    Startup.cb := SizeOf(Startup);
    Startup.dwFlags := STARTF_USESTDHANDLES or STARTF_USESHOWWINDOW;
    Startup.wShowWindow := SW_HIDE;
    Startup.hStdInput := ChildInput;
    Startup.hStdOutput := ChildOutput;
    Startup.hStdError := ChildOutput;
    FillChar(Info, SizeOf(Info), 0);
    Command := '"' + PythonPath + '" -I -B -u "' + ScriptPath + '"';
    UniqueString(Command);
    if not CreateProcessW(PWideChar(PythonPath), PWideChar(Command), nil, nil,
      True, CREATE_NO_WINDOW, nil, PWideChar(Root), Startup, Info) then RaiseLastOSError;
    FProcess := Info.hProcess;
    CloseHandle(Info.hThread);
  finally
    if ChildInput <> 0 then CloseHandle(ChildInput);
    if ChildOutput <> 0 then CloseHandle(ChildOutput);
    if FProcess = 0 then
    begin
      if FInput <> 0 then CloseHandle(FInput);
      if FOutput <> 0 then CloseHandle(FOutput);
      FInput := 0;
      FOutput := 0;
    end;
  end;
end;

procedure TGemmaProcess.Send(const Text: string);
var
  Bytes: UTF8String;
  Written: DWORD;
begin
  if FInput = 0 then raise Exception.Create('The Gemma runtime is not running.');
  Bytes := UTF8Encode(Text + #10);
  if Length(Bytes) > 65536 then raise Exception.Create('The request is too large.');
  if not WriteFile(FInput, Bytes[1], Length(Bytes), Written, nil) then RaiseLastOSError;
  if Written <> DWORD(Length(Bytes)) then raise Exception.Create('Incomplete runtime request.');
end;

procedure TGemmaProcess.Generate(const Bundle, Prompt: string; const MaxNewTokens: Int64;
  const Plain: Boolean);
var
  Request: TJSONObject;
begin
  if MaxNewTokens < 0 then raise Exception.Create('Use 0 or a positive token limit.');
  Inc(FRequestID);
  Request := TJSONObject.Create;
  try
    Request.AddPair('command', 'generate');
    Request.AddPair('id', JSONInteger(FRequestID));
    Request.AddPair('bundle', Bundle);
    Request.AddPair('prompt', Prompt);
    Request.AddPair('max_new_tokens', JSONInteger(MaxNewTokens));
    Request.AddPair('plain', JSONBoolean(Plain));
    Send(Request.ToJSON);
  finally
    Request.Free;
  end;
end;

procedure TGemmaProcess.Request(const Command, Bundle, Prompt: string;
  const Options: TQwenGenerationOptions);
begin
  Request(Command, Bundle, Prompt, Options, []);
end;

procedure TGemmaProcess.Request(const Command, Bundle, Prompt: string;
  const Options: TQwenGenerationOptions; const Images: array of string);
const
  Efforts: array[TQwenAnswerEffort] of string = ('full_context', 'brief', 'balanced', 'detailed');
var
  Data, Settings: TJSONObject;
  Sampling, Image: string;
  Attachments: TJSONArray;
begin
  ValidateQwenGenerationOptions(Options);
  Inc(FRequestID);
  Data := TJSONObject.Create;
  try
    Data.AddPair('command', Command);
    Data.AddPair('id', JSONInteger(FRequestID));
    Data.AddPair('bundle', Bundle);
    Data.AddPair('prompt', Prompt);
    Data.AddPair('system_prompt', Options.SystemPrompt);
    Data.AddPair('max_new_tokens', JSONInteger(Options.MaxNewTokens));
    Data.AddPair('plain', JSONBoolean(Options.PlainCompletion));
    Attachments := TJSONArray.Create;
    Data.AddPair('images', Attachments);
    for Image in Images do
      Attachments.AddElement(TJSONString.Create({$IFDEF FPC}UTF8Encode(Image){$ELSE}Image{$ENDIF}));
    Settings := TJSONObject.Create;
    Data.AddPair('options', Settings);
    Sampling := 'greedy';
    if (Options.Sampling = qsSample) or ((Options.Sampling = qsDefault) and
      (Options.MaxNewTokens = 0)) then
        Sampling := 'sample';
    Settings.AddPair('sampling', Sampling);
    Settings.AddPair('temperature', JSONFloat(Options.Temperature));
    Settings.AddPair('top_p', JSONFloat(Options.TopP));
    Settings.AddPair('top_k', JSONInteger(Options.TopK));
    Settings.AddPair('repetition_penalty', JSONFloat(Options.RepetitionPenalty));
    Settings.AddPair('seed', JSONInteger(Options.Seed));
    Settings.AddPair('effort', Efforts[Options.Effort]);
    Send(Data.ToJSON);
  finally
    Data.Free;
  end;
end;

procedure TGemmaProcess.Stop;
begin
  Send('{"command":"stop","id":' + IntToStr(FRequestID) + '}');
end;

procedure TGemmaProcess.RequestClose;
begin
  if FInput <> 0 then Send('{"command":"close"}');
end;

function TGemmaProcess.ReadLine(out Line: string): Boolean;
var
  Available, Count: DWORD;
  Buffer: array[0..4095] of AnsiChar;
  Chunk: RawByteString;
  Index: Integer;
begin
  Result := False;
  Line := '';
  Index := Pos(#10, FPending);
  if (Index = 0) and (FOutput <> 0) then
  begin
    while PeekNamedPipe(FOutput, nil, 0, nil, @Available, nil) and (Available > 0) do
    begin
      if Available > SizeOf(Buffer) then Available := SizeOf(Buffer);
      if not ReadFile(FOutput, Buffer, Available, Count, nil) then Break;
      SetString(Chunk, PAnsiChar(@Buffer[0]), Count);
      FPending := FPending + Chunk;
      if Length(FPending) > 16 * 1024 * 1024 then
        raise Exception.Create('Runtime response exceeds the protocol limit.');
      Index := Pos(#10, FPending);
      if Index > 0 then Break;
    end;
  end;
  if Index > 0 then
  begin
    Line := UTF8Decode(Copy(FPending, 1, Index - 1));
    Delete(FPending, 1, Index);
    Result := True;
  end;
end;

function TGemmaProcess.HasExited(out Code: Cardinal): Boolean;
begin
  Code := 0;
  Result := (FProcess <> 0) and (WaitForSingleObject(FProcess, 0) = WAIT_OBJECT_0);
  if Result and not GetExitCodeProcess(FProcess, Code) then RaiseLastOSError;
end;

destructor TGemmaProcess.Destroy;
begin
  if FInput <> 0 then CloseHandle(FInput);
  if FOutput <> 0 then CloseHandle(FOutput);
  if FProcess <> 0 then CloseHandle(FProcess);
  inherited;
end;

end.
