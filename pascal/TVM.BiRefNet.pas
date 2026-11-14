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
  private
    FGraph: TCpuGraph;
    FCuda: TCudaExecution;
    FWorkspace: TCudaWorkspace;
    FLease: TFileStream;
    FDevice: TDLDeviceType;
    FDeviceName: string;
    FModelSHA256: string;
    FDirectory: TWindowsModelDirectory;
  public
    constructor Create(const Bundle, RuntimeDirectory: string);
    destructor Destroy; override;
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

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, CrossPascal.Json, CrossPascal.SHA256,
  QwenCuda, OpenCVConsts, OpenCVTypes, TVM.ImageInput;

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
  Source, Target: PByte;
  Alpha: PSingle;
  X, Y, A: Integer;
begin
  CheckImage(Image);
  RequireCpu((Mask <> nil) and not Mask.IsEmpty and (Mask.Width = Image.Width) and
    (Mask.Height = Image.Height) and (Mask.Depth = CV_32F) and (Mask.Channels = 1) and
    (Mask.DataPtr <> nil) and (Mask.Step >= NativeInt(Mask.Width) * SizeOf(Single)),
    'BiRefNet requires a matching FP32 probability mask');
  Result := NewMat(Image.Width, Image.Height, CV_MAKETYPE(CV_8U, 4));
  RequireCpu((Result <> nil) and not Result.IsEmpty and (Result.DataPtr <> nil),
    'BiRefNet cutout allocation failed');
  for Y := 0 to Image.Height - 1 do
  begin
    Source := PByte(Image.DataPtr) + NativeInt(Y) * Image.Step;
    Target := PByte(Result.DataPtr) + NativeInt(Y) * Result.Step;
    Alpha := PSingle(PByte(Mask.DataPtr) + NativeInt(Y) * Mask.Step);
    for X := 0 to Image.Width - 1 do
    begin
      RequireCpu(not IsNan(Alpha[X]) and not IsInfinite(Alpha[X]) and
        (Alpha[X] >= 0) and (Alpha[X] <= 1), 'Invalid BiRefNet alpha probability');
      A := Round(Alpha[X] * 255);
      Target[X * 4] := Source[X * 3];
      Target[X * 4 + 1] := Source[X * 3 + 1];
      Target[X * 4 + 2] := Source[X * 3 + 2];
      Target[X * 4 + 3] := A;
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
begin
  inherited Create;
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
      FCuda := TCudaExecution.Create;
      Architecture := ModelText(Artifact.GetValue('cuda_arch'));
      FCuda.RequireArchitecture(Architecture);
      FDeviceName := FCuda.DeviceName + ' (' + Architecture + ')';
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
    if FCuda <> nil then FCuda.Enter;
    try
      if FCuda <> nil then
        RequireCpu(FCuda.FreeBytes >= MinimumFreeCUDA,
          'BiRefNet requires at least 4 GiB of free CUDA memory');
      FGraph := TCpuGraph.Create(LibraryName, FDevice);
      if FCuda <> nil then
        FWorkspace := TCudaWorkspace.Create(FCuda, LibraryName, WorkspaceReserve);
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    Document.Free;
  end;
end;

destructor TBiRefNetModel.Destroy;
begin
  if FCuda <> nil then FCuda.Enter;
  try
    FWorkspace.Free;
    FGraph.Free;
  finally
    if FCuda <> nil then FCuda.Leave;
  end;
  FCuda.Free;
  FLease.Free;
  FDirectory.Free;
  inherited;
end;

function TBiRefNetModel.Run(const Input: TArray<Single>;
  const ShouldStop: TCpuGraphStop): TArray<Single>;
var
  Argument, Returned, Output: TTVMFFIObjectHandle;
  Value: TTVMFFIAny;
  Shape: TArray<Int64>;
  I: Integer;
begin
  RequireCpu(Length(Input) = 3 * BiRefNetSide * BiRefNetSide, 'BiRefNet input element count differs');
  if Assigned(ShouldStop) and ShouldStop() then raise ECpuGraphStopped.Create('BiRefNet stopped');
  Argument := nil;
  Returned := nil;
  Output := nil;
  if FCuda <> nil then FCuda.Enter;
  try
    Argument := DeviceTensor([1, 3, BiRefNetSide, BiRefNetSide], kDLFloat, 32, Pointer(Input), FDevice);
    if FWorkspace <> nil then
      Value := FWorkspace.Run(FGraph, [Argument], ShouldStop)
    else
      Value := FGraph.Run([Argument], ShouldStop);
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
  finally
    TTVMRuntime.ObjectRelease(Output);
    TTVMRuntime.ObjectRelease(Returned);
    TTVMRuntime.ObjectRelease(Argument);
    if FCuda <> nil then FCuda.Leave;
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

end.
