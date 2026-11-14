{ ************************************************************************** }
{                                                                            }
{ TVM.BiRefNet                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.BiRefNet;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  DLPack, TVM, TVM.CpuGraph, TVM.CudaExecution, TVM.WindowsPaths, CrossVision.Mat;

const
  BiRefNetSide = 800;
  BiRefNetSourceSHA256 = '99c984d124ee1b68aaccbdfc715c1a9b5025b769f8b49d296eaf2ba028d247e6';

type
  TBiRefNetModel = class
  private type
    TCloseState = (csOpen, csClosing, csClosed, csFailed, csQuarantined);
    TNativeResources = class
      Graph: TCpuGraph;
      Cuda: TCudaExecution;
      Workspace: TCudaWorkspace;
      Next: TNativeResources;
    end;
  private
    FNative: TNativeResources;
    FCloseState: TCloseState;
    FCloseError: string;
    FContextUncertain: Boolean;
    FLease: TFileStream;
    FDevice: TDLDeviceType;
    FDeviceName: string;
    FModelSHA256: string;
    FDirectory: TWindowsModelDirectory;
    procedure RememberError(const Operation, Detail: string);
    procedure ContextFailed(const Operation, Detail: string);
    procedure EnterContext;
    procedure LeaveContext;
    procedure FinishContext(const BodySucceeded: Boolean);
    procedure QuarantineNative;
    procedure Cleanup;
  public
    constructor Create(const Bundle, RuntimeDirectory: string);
    destructor Destroy; override;
    procedure Close;
    function Run(const Input: TArray<Single>;
      const ShouldStop: TCpuGraphStop = nil): TArray<Single>;
    procedure RemoveBackground(const Image: IMat; out Mask, Cutout: IMat;
      const ShouldStop: TCpuGraphStop = nil);
    property DeviceName: string read FDeviceName;
    property ModelSHA256: string read FModelSHA256;
  end;

function BiRefNetInput(const Image: IMat): TArray<Single>;
function BiRefNetMask(const Logits: TArray<Single>; const Width, Height: Integer): IMat;
function BiRefNetCutout(const Image, Mask: IMat): IMat;
function BiRefNetRequiresProcessRestart: Boolean;
function BiRefNetCleanupDiagnostic: string;

implementation

uses
  {$IFDEF FPC}Windows, Math{$ELSE}Winapi.Windows, System.Math{$ENDIF}, CrossPascal.Json,
  CrossPascal.SHA256, QwenCuda, OpenCVConsts, OpenCVTypes, TVM.ImageInput;

var
  CleanupLock: TRTLCriticalSection;
  RestartRequired: LongInt;
  CleanupDiagnostic: string;
  QuarantinedNative: TBiRefNetModel.TNativeResources;

function BiRefNetRequiresProcessRestart: Boolean;
begin
  Result := InterlockedCompareExchange(RestartRequired, 0, 0) <> 0;
end;

function BiRefNetCleanupDiagnostic: string;
begin
  EnterCriticalSection(CleanupLock);
  try
    Result := CleanupDiagnostic;
  finally
    LeaveCriticalSection(CleanupLock);
  end;
end;

procedure CheckImage(const Image: IMat);
begin
  RequireCpu((Image <> nil) and not Image.IsEmpty and (Image.Depth = CV_8U) and
    (Image.Channels = 3) and (Image.Width > 0) and (Image.Height > 0) and
    (Image.Width <= 4096) and (Image.Height <= 4096) and (Image.DataPtr <> nil) and
    (Image.Step >= NativeInt(Image.Width) * 3),
    'BiRefNet requires a BGR8 image up to 4096 pixels per side');
end;

function BiRefNetInput(const Image: IMat): TArray<Single>;
const
  Mean: array[0..2] of Single = (0.485, 0.456, 0.406);
  Std: array[0..2] of Single = (0.229, 0.224, 0.225);
var
  Pixels: TBytes;
  X, Y, C: Integer;
  Value: Single;
begin
  CheckImage(Image);
  Pixels := ResizeColorImage(Image, BiRefNetSide, BiRefNetSide, True);
  SetLength(Result, 3 * BiRefNetSide * BiRefNetSide);
  for Y := 0 to BiRefNetSide - 1 do
  begin
    for X := 0 to BiRefNetSide - 1 do
      for C := 0 to 2 do
      begin
        Value := Pixels[(Y * BiRefNetSide + X) * 3 + C] / 255;
        Value := Value - Mean[C];
        Result[(C * BiRefNetSide + Y) * BiRefNetSide + X] := Value / Std[C];
      end;
  end;
end;

function BiRefNetMask(const Logits: TArray<Single>; const Width, Height: Integer): IMat;
var
  Probabilities: TArray<Single>;
  Row: PSingle;
  X, Y, I, X0, X1, Y0, Y1: Integer;
  Value, Exponent, SX, SY, WX, WY, Top, Bottom: Double;
begin
  RequireCpu((Length(Logits) = BiRefNetSide * BiRefNetSide) and
    (Width > 0) and (Width <= 4096) and (Height > 0) and (Height <= 4096),
    'Invalid BiRefNet mask dimensions');
  SetLength(Probabilities, Length(Logits));
  I := 0;
  for Y := 0 to BiRefNetSide - 1 do
  begin
    for X := 0 to BiRefNetSide - 1 do
    begin
      Value := Logits[I];
      RequireCpu(not IsNan(Value) and not IsInfinite(Value), 'Non-finite BiRefNet logit');
      if Value >= 80 then
        Probabilities[I] := 1
      else if Value <= -104 then
          Probabilities[I] := 0
        else if Value >= 0 then
            Probabilities[I] := 1 / (1 + Exp(-Value))
          else begin
            Exponent := Exp(Value);
            Probabilities[I] := Exponent / (1 + Exponent);
          end;
      Inc(I);
    end;
  end;
  Result := NewMat(Width, Height, CV_MAKETYPE(CV_32F, 1));
  RequireCpu((Result <> nil) and not Result.IsEmpty and (Result.DataPtr <> nil),
    'BiRefNet mask allocation failed');
  for Y := 0 to Height - 1 do
  begin
    SY := Y;
    SY := (SY + 0.5) * BiRefNetSide / Height - 0.5;
    SY := Max(0.0, Min(BiRefNetSide - 1.0, SY));
    Y0 := Floor(SY);
    Y1 := Min(Y0 + 1, BiRefNetSide - 1);
    WY := SY - Y0;
    Row := PSingle(PByte(Result.DataPtr) + NativeInt(Y) * Result.Step);
    for X := 0 to Width - 1 do
    begin
      SX := X;
      SX := (SX + 0.5) * BiRefNetSide / Width - 0.5;
      SX := Max(0.0, Min(BiRefNetSide - 1.0, SX));
      X0 := Floor(SX);
      X1 := Min(X0 + 1, BiRefNetSide - 1);
      WX := SX - X0;
      Top := Probabilities[Y0 * BiRefNetSide + X0] * (1 - WX) +
        Probabilities[Y0 * BiRefNetSide + X1] * WX;
      Bottom := Probabilities[Y1 * BiRefNetSide + X0] * (1 - WX) +
        Probabilities[Y1 * BiRefNetSide + X1] * WX;
      Row[X] := Top * (1 - WY) + Bottom * WY;
    end;
  end;
end;

function BiRefNetCutout(const Image, Mask: IMat): IMat;
var
  Source, Target, ByteAlpha: PByte;
  Alpha: PSingle;
  ByteMask: IMat;
  X, Y: Integer;
begin
  CheckImage(Image);
  RequireCpu((Mask <> nil) and not Mask.IsEmpty and (Mask.Width = Image.Width) and
    (Mask.Height = Image.Height) and (Mask.Depth = CV_32F) and (Mask.Channels = 1) and
    (Mask.DataPtr <> nil) and (Mask.Step >= NativeInt(Mask.Width) * SizeOf(Single)),
    'BiRefNet requires a matching FP32 probability mask');
  for Y := 0 to Image.Height - 1 do
  begin
    Alpha := PSingle(PByte(Mask.DataPtr) + NativeInt(Y) * Mask.Step);
    for X := 0 to Image.Width - 1 do
      RequireCpu(not IsNan(Alpha[X]) and not IsInfinite(Alpha[X]) and
        (Alpha[X] >= 0) and (Alpha[X] <= 1), 'Invalid BiRefNet alpha probability');
  end;
  ByteMask := Mask.ConvertTo(CV_MAKETYPE(CV_8U, 1), 255, 0);
  RequireCpu((ByteMask <> nil) and not ByteMask.IsEmpty and
    (ByteMask.Width = Image.Width) and (ByteMask.Height = Image.Height) and
    (ByteMask.Depth = CV_8U) and (ByteMask.Channels = 1) and
    (ByteMask.DataPtr <> nil) and (ByteMask.Step >= Image.Width),
    'BiRefNet byte mask conversion failed');
  Result := NewMat(Image.Width, Image.Height, CV_MAKETYPE(CV_8U, 4));
  RequireCpu((Result <> nil) and not Result.IsEmpty and (Result.DataPtr <> nil),
    'BiRefNet cutout allocation failed');
  for Y := 0 to Image.Height - 1 do
  begin
    Source := PByte(Image.DataPtr) + NativeInt(Y) * Image.Step;
    Target := PByte(Result.DataPtr) + NativeInt(Y) * Result.Step;
    ByteAlpha := PByte(ByteMask.DataPtr) + NativeInt(Y) * ByteMask.Step;
    for X := 0 to Image.Width - 1 do
    begin
      Target[X * 4] := Source[X * 3];
      Target[X * 4 + 1] := Source[X * 3 + 1];
      Target[X * 4 + 2] := Source[X * 3 + 2];
      Target[X * 4 + 3] := ByteAlpha[X];
    end;
  end;
end;

constructor TBiRefNetModel.Create(const Bundle, RuntimeDirectory: string);
const
  WorkspaceReserve = UInt64(1024) * 1024 * 1024;
  MinimumFreeCUDA = UInt64(4) * 1024 * 1024 * 1024;
var
  Document: TJSONValue;
  Root, Artifact, IO, Settings, Provenance: TJSONObject;
  Shape: TArray<Int64>;
  Folder, Device, Architecture, LibraryName: string;
  BodySucceeded: Boolean;
begin
  inherited Create;
  FNative := TNativeResources.Create;
  RequireCpu(not BiRefNetRequiresProcessRestart,
    'Restart the application before loading another BiRefNet model.');
  FDirectory := TWindowsModelDirectory.Create(Bundle);
  Folder := FDirectory.Path;
  Document := ReadModelJSON(BundleMember(Folder, 'manifest.json'));
  try
    Root := ModelObject(Document, 'BiRefNet manifest');
    RequireCpu(
      (ModelInt(Root.GetValue('format_version'), 'format version') = 1) and (ModelText(Root.GetValue('adapter')) = 'birefnet') and (ModelText(Root.GetValue('task_kind')) = 'image_segmentation'),
      'Unsupported BiRefNet manifest'
    );
    IO := ModelObject(Root.GetValue('io'), 'BiRefNet input');
    Shape := ModelShape(IO.GetValue('shape'));
    RequireCpu((Length(Shape) = 4) and (Shape[0] = 1) and (Shape[1] = 3) and
      (Shape[2] = BiRefNetSide) and (Shape[3] = BiRefNetSide) and
      (ModelText(IO.GetValue('name')) = 'x') and (ModelText(IO.GetValue('layout')) = 'NCHW') and
      (ModelText(IO.GetValue('dtype')) = 'float32'), 'Invalid BiRefNet input contract');
    Settings := ModelObject(Root.GetValue('segmentation'), 'BiRefNet settings');
    RequireCpu((ModelText(Settings.GetValue('profile')) = 'birefnet-hr800') and
      (ModelText(Settings.GetValue('preprocessing')) = 'rgb-cubic-imagenet') and
      (ModelText(Settings.GetValue('output')) = 'logits-nchw') and
      (ModelText(Settings.GetValue('mask_resize')) = 'linear'), 'Unsupported BiRefNet profile');
    Provenance := ModelObject(Root.GetValue('provenance'), 'BiRefNet provenance');
    RequireCpu(SameText(ModelText(Provenance.GetValue('source_hash')), BiRefNetSourceSHA256),
      'BiRefNet source SHA-256 differs');
    Artifact := ModelObject(Root.GetValue('artifact'), 'BiRefNet artifact');
    RequireCpu((ModelText(Artifact.GetValue('os')) = 'windows') and
      (ModelText(Artifact.GetValue('arch')) = 'x86_64') and
      (ModelText(Artifact.GetValue('dtype')) = 'float32') and
      (ModelText(Artifact.GetValue('entry')) = 'main'), 'Unsupported BiRefNet artifact');
    Device := ModelText(Artifact.GetValue('device'));
    RequireCpu((Device = 'cpu') or (Device = 'cuda'), 'Unsupported BiRefNet device');
    FDevice := kDLCPU;
    FDeviceName := 'CPU';
    if Device = 'cuda' then
    begin
      FDevice := kDLCUDA;
      FNative.Cuda := TCudaExecution.Create;
      Architecture := ModelText(Artifact.GetValue('cuda_arch'));
      FNative.Cuda.RequireArchitecture(Architecture);
      FDeviceName := FNative.Cuda.DeviceName + ' (' + Architecture + ')';
    end;
    LibraryName := 'model.cpu.dll';
    if FDevice = kDLCUDA then LibraryName := 'model.cuda.dll';
    RequireCpu(ModelText(Artifact.GetValue('file')) = LibraryName,
      'Unsupported BiRefNet model library name');
    LibraryName := BundleMember(Folder, LibraryName);
    FLease := TFileStream.Create(LibraryName, fmOpenRead or fmShareDenyWrite);
    FModelSHA256 := SHA256Stream(FLease);
    RequireCpu(SameText(FModelSHA256, ModelText(Artifact.GetValue('hash'))),
      'BiRefNet model SHA-256 differs');
    InitializeRuntime(ModelRuntimeDirectory(RuntimeDirectory), FDevice = kDLCUDA);
    EnterContext;
    BodySucceeded := False;
    try
      if FNative.Cuda <> nil then
        RequireCpu(FNative.Cuda.FreeBytes >= MinimumFreeCUDA,
          'BiRefNet requires at least 4 GiB of free CUDA memory');
      FNative.Graph := TCpuGraph.Create(LibraryName, FDevice);
      if FNative.Cuda <> nil then
        FNative.Workspace := TCudaWorkspace.Create(FNative.Cuda, LibraryName, WorkspaceReserve);
      BodySucceeded := True;
    finally
      FinishContext(BodySucceeded);
    end;
  finally
    Document.Free;
  end;
end;

procedure TBiRefNetModel.RememberError(const Operation, Detail: string);
begin
  if FCloseError <> '' then FCloseError := FCloseError + ' ';
  FCloseError := FCloseError + Operation + ': ' + Detail;
end;

procedure TBiRefNetModel.ContextFailed(const Operation, Detail: string);
begin
  FContextUncertain := True;
  InterlockedExchange(RestartRequired, 1);
  RememberError(Operation, Detail);
  EnterCriticalSection(CleanupLock);
  try
    if CleanupDiagnostic = '' then
      CleanupDiagnostic := FCloseError
    else if Pos(Operation + ': ' + Detail, CleanupDiagnostic) = 0 then
        CleanupDiagnostic := CleanupDiagnostic + ' ' + Operation + ': ' + Detail;
  finally
    LeaveCriticalSection(CleanupLock);
  end;
end;

procedure TBiRefNetModel.EnterContext;
begin
  if FNative.Cuda = nil then Exit;
  try
    FNative.Cuda.Enter;
  except
    on E: Exception do
    begin
      ContextFailed('CUDA context entry', E.Message);
      raise;
    end;
  end;
end;

procedure TBiRefNetModel.LeaveContext;
begin
  if FNative.Cuda = nil then Exit;
  try
    FNative.Cuda.Leave;
  except
    on E: Exception do
    begin
      ContextFailed('CUDA context restoration', E.Message);
      raise;
    end;
  end;
end;

procedure TBiRefNetModel.FinishContext(const BodySucceeded: Boolean);
begin
  if BodySucceeded then
    LeaveContext
  else begin
    try
      LeaveContext;
    except
      // The original constructor or inference exception remains primary.
    end;
  end;
end;

procedure TBiRefNetModel.QuarantineNative;
begin
  if FNative = nil then Exit;
  InterlockedExchange(RestartRequired, 1);
  EnterCriticalSection(CleanupLock);
  try
    FNative.Next := QuarantinedNative;
    QuarantinedNative := FNative;
    FNative := nil;
  finally
    LeaveCriticalSection(CleanupLock);
  end;
end;

procedure TBiRefNetModel.Cleanup;
var
  Entered, Quarantined: Boolean;
  Workspace: TCudaWorkspace;
  Graph: TCpuGraph;
begin
  if FCloseState <> csOpen then Exit;
  FCloseState := csClosing;
  Entered := False;
  Quarantined := False;
  try
    try
      if FNative <> nil then
      begin
        if (FNative.Cuda <> nil) and BiRefNetRequiresProcessRestart then
          FContextUncertain := True;
        if not FContextUncertain then
        begin
          try
            EnterContext;
            Entered := FNative.Cuda <> nil;
          except
            // EnterContext records the failure. Never guess whether push occurred.
          end;
          if not FContextUncertain then
          begin
            Workspace := FNative.Workspace;
            FNative.Workspace := nil;
            try
              Workspace.Free;
            except
              on E: Exception do ContextFailed('CUDA workspace cleanup', E.Message);
            end;
            Graph := FNative.Graph;
            FNative.Graph := nil;
            try
              Graph.Free;
            except
              on E: Exception do ContextFailed('Model graph cleanup', E.Message);
            end;
          end;
          if Entered then
          begin
            try
              LeaveContext;
            except
              // Exactly one pop for the successful push; no retry after an error.
            end;
          end;
        end;
        if FContextUncertain then
        begin
          if FCloseError = '' then RememberError('CUDA cleanup', 'Context state is uncertain.');
          QuarantineNative;
          Quarantined := True;
        end
        else begin
          FreeAndNil(FNative.Cuda);
          FreeAndNil(FNative);
        end;
      end;
    except
      on E: Exception do
      begin
        ContextFailed('Native cleanup', E.Message);
        QuarantineNative;
        Quarantined := True;
      end;
    end;
  finally
    try
      FreeAndNil(FLease);
    except
      on E: Exception do RememberError('Model file cleanup', E.Message);
    end;
    try
      FreeAndNil(FDirectory);
    except
      on E: Exception do RememberError('Model path cleanup', E.Message);
    end;
    if Quarantined then
      FCloseState := csQuarantined
    else if (FCloseError <> '') or (FNative <> nil) or FContextUncertain then
        FCloseState := csFailed
      else
        FCloseState := csClosed;
  end;
end;

procedure TBiRefNetModel.Close;
begin
  Cleanup;
  RequireCpu(FCloseState = csClosed,
    'BiRefNet cleanup did not complete. ' + FCloseError +
    ' Restart the application before running another model.');
end;

destructor TBiRefNetModel.Destroy;
begin
  try
    Cleanup;
  except
    // Destruction must preserve an exception already being unwound.
    try
      QuarantineNative;
    except
    end;
    try
      FreeAndNil(FLease);
    except
    end;
    try
      FreeAndNil(FDirectory);
    except
    end;
  end;
  inherited;
end;

function TBiRefNetModel.Run(const Input: TArray<Single>;
  const ShouldStop: TCpuGraphStop): TArray<Single>;
var
  Argument, Returned, Output: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
  Shape: TArray<Int64>;
  I: Integer;
  BodySucceeded: Boolean;
begin
  RequireCpu((FCloseState = csOpen) and not BiRefNetRequiresProcessRestart,
    'Restart the application before running another BiRefNet model.');
  RequireCpu(Length(Input) = 3 * BiRefNetSide * BiRefNetSide, 'BiRefNet input element count differs');
  if Assigned(ShouldStop) and ShouldStop() then raise ECpuGraphStopped.Create('BiRefNet stopped');
  Argument := nil;
  Returned := nil;
  Output := nil;
  EnterContext;
  BodySucceeded := False;
  try
    Argument := DeviceTensor([1, 3, BiRefNetSide, BiRefNetSide], kDLFloat, 32, Pointer(Input), FDevice);
    if FNative.Workspace <> nil then
      Value := FNative.Workspace.Run(FNative.Graph, [Argument], ShouldStop)
    else
      Value := FNative.Graph.Run([Argument], ShouldStop);
    Returned := Value.VObj;
    if Value.TypeIndex = kTVMFFITensor then
      Shape := TTVMRuntime.GetTensorShape(TTVMRuntime.TensorDLTensor(Returned))
    else begin
      Output := TensorItem(Returned, 0);
      Shape := TTVMRuntime.GetTensorShape(TTVMRuntime.TensorDLTensor(Output));
    end;
    RequireCpu((Length(Shape) = 4) and (Shape[0] = 1) and (Shape[1] = 1) and
      (Shape[2] = BiRefNetSide) and (Shape[3] = BiRefNetSide), 'BiRefNet output shape differs');
    if Output = nil then
      Result := ReadFloatTensor(Returned)
    else
      Result := ReadFloatTensor(Output);
    for I := 0 to High(Result) do
      RequireCpu(not IsNan(Result[I]) and not IsInfinite(Result[I]), 'Non-finite BiRefNet output');
    BodySucceeded := True;
  finally
    TTVMRuntime.ObjectRelease(Output);
    TTVMRuntime.ObjectRelease(Returned);
    TTVMRuntime.ObjectRelease(Argument);
    FinishContext(BodySucceeded);
  end;
end;

procedure TBiRefNetModel.RemoveBackground(const Image: IMat; out Mask, Cutout: IMat;
  const ShouldStop: TCpuGraphStop);
var
  Logits: TArray<Single>;
begin
  Mask := nil;
  Cutout := nil;
  Logits := Run(BiRefNetInput(Image), ShouldStop);
  if Assigned(ShouldStop) and ShouldStop() then raise ECpuGraphStopped.Create('BiRefNet stopped');
  Mask := BiRefNetMask(Logits, Image.Width, Image.Height);
  Cutout := BiRefNetCutout(Image, Mask);
end;

initialization
  InitializeCriticalSection(CleanupLock);

finalization
  // Quarantined native groups stay untouched until process termination.
  DeleteCriticalSection(CleanupLock);

end.
