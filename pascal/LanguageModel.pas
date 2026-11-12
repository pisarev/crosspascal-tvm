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
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, QwenCuda, Gemma3nNative, Gemma2Cpu,
  QwenVLCpu, Qwen2Native;

type
  TLanguageOptions = TQwenGenerationOptions;
  TLanguageProgress = TQwenProgress;
  TLanguageStop = TQwenStopRequested;

  TLanguageCapabilities = record
    Architecture: string;
    Schema: string;
    Device: string;
    ContextTokens: Integer;
    SupportsImages: Boolean;
    ImageExplanation: string;
    Experimental: Boolean;
  end;

  TLanguageModel = class
  private
    FQwen: TQwenCudaSession;
    FQwen2Native: TQwen2Native;
    FGemma3n: TGemma3nNative;
    FGemma2Cpu: TGemma2Cpu;
    FQwenVLCpu: TQwenVLCpu;
    FBundle: string;
    FRuntime: string;
    FLoaded: Boolean;
    FInfo: TLanguageCapabilities;
    FOnText: TLanguageProgress;
    FOnStatus: TLanguageProgress;
    FShouldStop: TLanguageStop;
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
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF}, CrossPascal.Json, TVM.CpuGraph;

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
  LibraryName, VisionManifest: string;
begin
  Result := Default(TLanguageCapabilities);
  Text := TStringList.Create;
  try
    Text.LoadFromFile(IncludeTrailingPathDelimiter(Bundle) + 'model.json', TEncoding.UTF8);
    Value := ParseCrossJSON(Text.Text);
    try
      if not (Value is TJSONObject) then raise Exception.Create('Invalid model.json');
      Obj := TJSONObject(Value);
      Result.Schema := TextField(Obj, 'schema');
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
      if (Result.Architecture = 'qwen2') and (Result.Schema <> 'crosspascal.qwen2.text.v1') then
        Result.ContextTokens := QwenDefaultContextSize;
      Result.Device := 'CPU';
      if (Result.Architecture = 'qwen2') or (TextField(Obj, 'backend') = 'TVM CUDA W8A32') then
        Result.Device := 'CUDA';
      Result.Experimental := Result.Architecture <> 'qwen2';
      Result.SupportsImages := False;
      if (Result.Architecture = 'qwen2_5_vl') and
        (TextField(Obj, 'schema') = 'crosspascal.qwen2_5_vl.v1') and
        ((TextField(Obj, 'backend') = 'TVM LLVM CPU FP32') or
        (TextField(Obj, 'backend') = 'TVM CUDA W8A32')) then
          Result.SupportsImages := True;
      if (Result.Architecture = 'gemma3n') and
        ((TextField(Obj, 'backend') = 'TVM CUDA W8A32') or
        (TextField(Obj, 'backend') = 'TVM LLVM CPU FP32')) then
        begin
          VisionManifest := TextField(Obj, 'vision_manifest');
          Result.SupportsImages := (VisionManifest <> '') and
            FileExists(BundleMember(Bundle, VisionManifest));
        end;
      Result.ImageExplanation := 'This bundle accepts text only. Images need a vision encoder and a compatible multimodal runtime. Your image will not be sent or silently ignored.';
    finally
      Value.Free;
    end;
  finally
    Text.Free;
  end;
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
      if FInfo.Schema = 'crosspascal.qwen2.text.v1' then
      begin
        FQwen2Native := TQwen2Native.Create(FBundle, FRuntime, FOnStatus);
        FInfo.ContextTokens := FQwen2Native.ContextTokens;
      end
      else begin
        Info := FQwen.Load(FBundle, FRuntime, FOnStatus);
        FInfo.ContextTokens := Info.ContextTokens;
      end;
    end
    else if FInfo.Architecture = 'gemma3n' then
      begin
        FGemma3n := TGemma3nNative.Create(FBundle, FRuntime, FOnStatus);
        FInfo.ContextTokens := FGemma3n.ContextTokens;
      end
      else if FInfo.Architecture = 'gemma2' then
        begin
          FGemma2Cpu := TGemma2Cpu.Create(FBundle, FRuntime, FOnStatus);
          FInfo.ContextTokens := FGemma2Cpu.ContextTokens;
        end
        else if FInfo.Architecture = 'qwen2_5_vl' then
          begin
            FQwenVLCpu := TQwenVLCpu.Create(FBundle, FRuntime, FOnStatus);
            FInfo.ContextTokens := FQwenVLCpu.ContextTokens;
          end
          else
            raise Exception.CreateFmt('No native runtime is available for %s on %s.',
              [FInfo.Architecture, FInfo.Device]);
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
    if FQwen2Native <> nil then
      Answer := FQwen2Native.Generate(Prompt, Options, FOnStatus, FOnText, FShouldStop)
    else
      Answer := FQwen.Run(FBundle, FRuntime, Prompt, Options, FOnStatus, FShouldStop, FOnText);
    Result := QwenResultJSON(Answer);
  end
  else if FGemma3n <> nil then
      Result := FGemma3n.Generate(Prompt, Options, Images, FOnStatus, FOnText, FShouldStop)
    else if FGemma2Cpu <> nil then
        Result := FGemma2Cpu.Generate(Prompt, Options, FOnStatus, FOnText, FShouldStop)
      else if FQwenVLCpu <> nil then
          Result := FQwenVLCpu.Generate(Prompt, Options, Images, FOnStatus, FOnText, FShouldStop)
        else
          raise Exception.Create('The native model session is unavailable.');
end;

procedure TLanguageModel.Unload;
begin
  FLoaded := False;
  if FQwen <> nil then FQwen.Unload;
  FreeAndNil(FQwen2Native);
  FreeAndNil(FGemma3n);
  FreeAndNil(FGemma2Cpu);
  FreeAndNil(FQwenVLCpu);
end;

destructor TLanguageModel.Destroy;
begin
  try
    Unload;
  finally
    FQwen.Free;
    inherited;
  end;
end;

end.
