{ ************************************************************************** }
{                                                                            }
{ LanguageModel                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit LanguageModel;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF},
  QwenCuda, GemmaProcess;

type
  TLanguageOptions = TQwenGenerationOptions;
  TLanguageProgress = TQwenProgress;
  TLanguageStop = TQwenStopRequested;

  TLanguageCapabilities = record
    Architecture: string;
    Device: string;
    ContextTokens: Integer;
    SupportsImages: Boolean;
    ImageExplanation: string;
    Experimental: Boolean;
  end;

  TLanguageModel = class
  private
    FQwen: TQwenCudaSession;
    FGemma: TGemmaProcess;
    FBundle: string;
    FRuntime: string;
    FLoaded: Boolean;
    FInfo: TLanguageCapabilities;
    FOnText: TLanguageProgress;
    FOnStatus: TLanguageProgress;
    FShouldStop: TLanguageStop;
    function GemmaRequest(const Command, Prompt: string; const Options: TLanguageOptions;
      const Images: array of string): string;
    procedure CloseWorker;
  public
    constructor Create;
    destructor Destroy; override;
    class function Inspect(const Bundle: string): TLanguageCapabilities; static;
    procedure Load(const Bundle, RuntimeRoot: string);
    function Generate(const Prompt: string; const Options: TLanguageOptions;
      const Images: array of string): string;
    procedure Unload;
    property Loaded: Boolean read FLoaded;
    property Capabilities: TLanguageCapabilities read FInfo;
    property OnText: TLanguageProgress read FOnText write FOnText;
    property OnStatus: TLanguageProgress read FOnStatus write FOnStatus;
    property ShouldStop: TLanguageStop read FShouldStop write FShouldStop;
  end;

function DefaultLanguageOptions: TLanguageOptions;

implementation

uses
  {$IFDEF FPC}Classes, Windows{$ELSE}System.Classes, Winapi.Windows{$ENDIF},
  CrossPascal.Json;

function DefaultLanguageOptions: TLanguageOptions;
begin
  Result := DefaultQwenGenerationOptions;
end;

function TextField(const Obj: TJSONObject; const Name: string): string;
var
  V: TJSONValue;
begin
  V := Obj.GetValue(Name);
  if (V = nil) or (V is TJSONNull) then
    Result := ''
  else
    Result := JSONValueText(V);
end;

constructor TLanguageModel.Create;
begin
  inherited;
  FQwen := TQwenCudaSession.Create;
end;

class function TLanguageModel.Inspect(const Bundle: string): TLanguageCapabilities;
var
  Text: TStringList;
  Value: TJSONValue;
  Obj: TJSONObject;
  LibraryName: string;
begin
  Result := Default(TLanguageCapabilities);
  Text := TStringList.Create;
  try
    Text.LoadFromFile(IncludeTrailingPathDelimiter(Bundle) + 'model.json', TEncoding.UTF8);
    Value := ParseCrossJSON(Text.Text);
    try
      if not (Value is TJSONObject) then raise Exception.Create('Invalid model.json');
      Obj := TJSONObject(Value);
      Result.Architecture := TextField(Obj, 'architecture');
      if Result.Architecture = '' then Result.Architecture := TextField(Obj, 'model_type');
      LibraryName := TextField(Obj, 'library');
      if LibraryName = 'qwen2-cuda.dll' then Result.Architecture := 'qwen2';
      if (Result.Architecture <> 'qwen2') and (Result.Architecture <> 'gemma2') and
        (Result.Architecture <> 'gemma3n') and (Result.Architecture <> 'qwen2_5_vl') then
          raise Exception.Create('This model architecture needs a different runtime adapter.');
      Result.ContextTokens := StrToIntDef(TextField(Obj, 'context'), 0);
      if Result.ContextTokens = 0 then
        Result.ContextTokens := StrToIntDef(TextField(Obj, 'context_window_size'), 0);
      if Result.Architecture = 'qwen2' then Result.ContextTokens := QwenDefaultContextSize;
      Result.Device := 'CPU';
      if (Result.Architecture = 'qwen2') or (TextField(Obj, 'backend') = 'TVM CUDA W8A32') then
        Result.Device := 'CUDA';
      Result.Experimental := Result.Architecture <> 'qwen2';
      Result.SupportsImages := False;
      if (Result.Architecture = 'qwen2_5_vl') and
        (TextField(Obj, 'schema') = 'crosspascal.qwen2_5_vl.v1') and
        (TextField(Obj, 'backend') = 'TVM LLVM CPU FP32') then
          Result.SupportsImages := True;
      if (Result.Architecture = 'gemma3n') and
        ((TextField(Obj, 'backend') = 'TVM CUDA W8A32') or
        (TextField(Obj, 'backend') = 'TVM LLVM CPU FP32')) and
        (TextField(Obj, 'vision_manifest') = 'vision.json') then
          Result.SupportsImages := FileExists(IncludeTrailingPathDelimiter(Bundle) + 'vision.json');
      Result.ImageExplanation := 'This bundle accepts text only. Images need a vision encoder and a compatible multimodal runtime. Your image will not be sent or silently ignored.';
    finally
      Value.Free;
    end;
  finally
    Text.Free;
  end;
end;

function TLanguageModel.GemmaRequest(const Command, Prompt: string; const Options: TLanguageOptions;
  const Images: array of string): string;
var
  Line, Event, ErrorText, ID: string;
  Value, Payload: TJSONValue;
  Obj: TJSONObject;
  Code: Cardinal;
  Stopped, Done: Boolean;
begin
  Result := '';
  FGemma.Request(Command, FBundle, Prompt, Options, Images);
  ID := IntToStr(FGemma.RequestID);
  Stopped := False;
  Done := False;
  ErrorText := '';
  while not Done do
  begin
    if not Stopped and Assigned(FShouldStop) and FShouldStop() then
    begin
      FGemma.Stop;
      Stopped := True;
    end;
    if FGemma.ReadLine(Line) then
    begin
      if (Line = '') or (Line[1] <> '{') then Continue;
      Value := ParseCrossJSON(Line);
      try
        if not (Value is TJSONObject) then raise Exception.Create('Invalid worker event');
        Obj := TJSONObject(Value);
        Event := TextField(Obj, 'event');
        if Event = 'fatal' then raise Exception.Create(TextField(Obj, 'message'));
        if TextField(Obj, 'id') <> ID then Continue;
        if Event = 'text' then
        begin
          if Assigned(FOnText) then FOnText(TextField(Obj, 'text'));
        end
        else if Event = 'status' then
          begin
            if Assigned(FOnStatus) then FOnStatus(TextField(Obj, 'text'));
          end
          else if Event = 'loaded' then
            begin
              FInfo.ContextTokens := StrToIntDef(TextField(Obj, 'context'), FInfo.ContextTokens);
              if Assigned(FOnStatus) then FOnStatus('Model loaded on ' + TextField(Obj, 'gpu'));
            end
            else if Event = 'result' then
              begin
                Payload := Obj.GetValue('result');
                if not (Payload is TJSONObject) then
                  raise Exception.Create('Invalid generation result');
                Result := TJSONObject(Payload).ToJSON;
              end
              else if Event = 'error' then
                  ErrorText := TextField(Obj, 'message')
                else if Event = 'done' then
                    Done := True;
      finally
        Value.Free;
      end;
    end
    else begin
      if FGemma.HasExited(Code) then
        raise Exception.Create('Model runtime exited unexpectedly: ' + UIntToStr(Code));
      Sleep(10);
    end;
  end;
  if ErrorText <> '' then raise Exception.Create(ErrorText);
end;

procedure TLanguageModel.Load(const Bundle, RuntimeRoot: string);
var
  Info: TQwenCudaResult;
begin
  Unload;
  FInfo := Inspect(Bundle);
  FBundle := ExpandFileName(Bundle);
  FRuntime := ExpandFileName(RuntimeRoot);
  try
    if FInfo.Architecture = 'qwen2' then
    begin
      if not FileExists(IncludeTrailingPathDelimiter(FRuntime) + 'tvm_runtime_cuda.dll') then
        FRuntime := IncludeTrailingPathDelimiter(FRuntime) +
          'crosspascal-tvm\runtime\windows-x64-cuda';
      Info := FQwen.Load(FBundle, FRuntime, FOnStatus);
      FInfo.ContextTokens := Info.ContextTokens;
    end
    else begin
      FGemma := TGemmaProcess.Create;
      FGemma.Start(FRuntime);
      GemmaRequest('load', '', DefaultLanguageOptions, []);
    end;
    FLoaded := True;
  except
    Unload;
    raise;
  end;
end;

function TLanguageModel.Generate(const Prompt: string; const Options: TLanguageOptions;
  const Images: array of string): string;
var
  Answer: TQwenCudaResult;
begin
  if not FLoaded then raise Exception.Create('Load a model first.');
  if (Length(Images) > 0) and not FInfo.SupportsImages then
    raise Exception.Create(FInfo.ImageExplanation);
  if FInfo.Architecture = 'qwen2' then
  begin
    Answer := FQwen.Run(FBundle, FRuntime, Prompt, Options, FOnStatus, FShouldStop, FOnText);
    Result := QwenResultJSON(Answer);
  end
  else
    Result := GemmaRequest('generate', Prompt, Options, Images);
end;

procedure TLanguageModel.CloseWorker;
var
  Code: Cardinal;
  Line: string;
begin
  if FGemma = nil then Exit;
  if not FGemma.HasExited(Code) then
  begin
    FGemma.RequestClose;
    while not FGemma.HasExited(Code) do
    begin
      while FGemma.ReadLine(Line) do;
      Sleep(10);
    end;
  end;
  FreeAndNil(FGemma);
  if Code <> 0 then raise Exception.Create('Model runtime shutdown failed: ' + UIntToStr(Code));
end;

procedure TLanguageModel.Unload;
begin
  FLoaded := False;
  if FQwen <> nil then FQwen.Unload;
  CloseWorker;
end;

destructor TLanguageModel.Destroy;
begin
  try
    Unload;
  finally
    FQwen.Free;
    FGemma.Free;
    inherited;
  end;
end;

end.
