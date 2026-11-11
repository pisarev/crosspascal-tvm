{ ************************************************************************** }
{                                                                            }
{ Gemma3nVision                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit Gemma3nVision;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM.CpuGraph, TVM.CudaExecution,
  DLPack, QwenCuda;

type
  TCudaVisionMemory = procedure of object;

  TGemma3nVision = class
  private
    FGraph: TCpuGraph;
    FWeightFile: TMappedModelFile;
    FArgs: TTensorHandles;
    FArgShapes: TArray<TArray<Int64>>;
    FArgOffsets: TArray<Int64>;
    FWeightBytes: UInt64;
    FDevice: TDLDeviceType;
    FCuda: TCudaExecution;
    FLibraryName: string;
    FBackend: string;
    FHard: TArray<Single>;
    FNorm: TArray<Single>;
    FProjection: TArray<Single>;
    FMean: array[0..2] of Single;
    FStd: array[0..2] of Single;
    FSize: Integer;
    FSoftTokens: Integer;
    FHiddenSize: Integer;
    FVocabOffset: Integer;
    FEpsilon: Single;
    FCVLibrary: string;
    FLastHash: string;
    FLastFeatures: TArray<Single>;
  public
    constructor Create(const Folder, CVLibrary: string; const HiddenSize: Integer;
      const Cuda: TCudaExecution = nil; const ManifestName: string = 'vision.json');
    destructor Destroy; override;
    function EndEmbedding(const Token: Integer): TArray<Single>;
    function Encode(const FileName: string; const ShouldStop: TQwenStopRequested; out Hash: string;
      out Reused: Boolean; const ReleaseDeviceMemory: TCudaVisionMemory = nil): TArray<Single>;
    property SoftTokens: Integer read FSoftTokens;
    property Backend: string read FBackend;
  end;

implementation

uses
  {$IFDEF FPC}Classes, Math{$ELSE}System.Classes, System.Math{$ENDIF}, TVM, OpenCV,
  OpenCVTypes, OpenCVConsts, CrossVision.Mat, CrossPascal.Json, CrossPascal.SHA256;

function ReadNpyFloats(const Path: string; const Count: Integer): TArray<Single>;
var
  Stream: TFileStream;
  Header: AnsiString;
  Magic: array[0..7] of Byte;
  HeaderSize: Word;
begin
  Stream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    Stream.ReadBuffer(Magic, SizeOf(Magic));
    RequireCpu((Magic[0] = $93) and (Magic[1] = Ord('N')) and (Magic[2] = Ord('U')) and
      (Magic[3] = Ord('M')) and (Magic[4] = Ord('P')) and (Magic[5] = Ord('Y')) and
      (Magic[6] = 1) and (Magic[7] = 0), 'Expected a NumPy v1 model array');
    Stream.ReadBuffer(HeaderSize, SizeOf(HeaderSize));
    SetLength(Header, HeaderSize);
    Stream.ReadBuffer(Pointer(Header)^, HeaderSize);
    RequireCpu((Pos('''descr'': ''<f4''', Header) > 0) and
      (Pos('''fortran_order'': False', Header) > 0) and
      (Stream.Size - Stream.Position = Int64(Count) * SizeOf(Single)),
      'Unexpected float32 array layout: ' + Path);
    SetLength(Result, Count);
    Stream.ReadBuffer(Result[0], Count * SizeOf(Single));
  finally
    Stream.Free;
  end;
end;

constructor TGemma3nVision.Create(const Folder, CVLibrary: string; const HiddenSize: Integer;
  const Cuda: TCudaExecution; const ManifestName: string);
var
  Data, LayoutData: TJSONValue;
  Manifest, Row: TJSONObject;
  Layout, Mean, Std: TJSONArray;
  Shape: TArray<Int64>;
  Offset: Int64;
  Address: Pointer;
  I, HardSize: Integer;
begin
  inherited Create;
  FCVLibrary := CVLibrary;
  FHiddenSize := HiddenSize;
  Data := ReadModelJSON(BundleMember(Folder, ManifestName));
  try
    Manifest := ModelObject(Data, 'vision manifest');
    FBackend := ModelText(Manifest.GetValue('backend'));
    RequireCpu((JSONValueText(Manifest.GetValue('schema')) = 'crosspascal.gemma3n.vision.v1') and
      ((FBackend = 'TVM CPU FP32') or ((FBackend = 'TVM CUDA FP32') and (Cuda <> nil))) and
      (ModelInt(Manifest.GetValue('text_hidden_size'), 'text hidden size') = HiddenSize),
      'Vision encoder does not match the text model or selected device');
    FDevice := kDLCPU;
    if FBackend = 'TVM CUDA FP32' then
    begin
      FDevice := kDLCUDA;
      FCuda := Cuda;
    end;
    VerifyArtifacts(Folder, ModelObject(Manifest.GetValue('artifact_sha256'), 'vision hashes'));
    FSize := ModelInt(Manifest.GetValue('image_size'), 'image size');
    FSoftTokens := ModelInt(Manifest.GetValue('soft_tokens'), 'image tokens');
    FVocabOffset := ModelInt(Manifest.GetValue('vocab_offset'), 'image vocabulary offset');
    FEpsilon := ModelFloat(Manifest.GetValue('rms_norm_eps'), 'RMS epsilon');
    RequireCpu((FSize > 0) and (FSize <= 4096) and (FSoftTokens > 0) and
      (FSoftTokens <= 4096) and (FEpsilon > 0), 'Invalid vision profile');
    Mean := ModelArray(Manifest.GetValue('mean'), 'mean');
    Std := ModelArray(Manifest.GetValue('std'), 'std');
    RequireCpu((Mean.Count = 3) and (Std.Count = 3), 'Expected RGB normalization');
    for I := 0 to 2 do
    begin
      FMean[I] := ModelFloat(Mean.Items[I], 'image mean');
      FStd[I] := ModelFloat(Std.Items[I], 'image standard deviation');
      RequireCpu(FStd[I] > 0, 'Image standard deviation must be positive');
    end;
    HardSize := FHiddenSize;
    // The shipped Gemma 3n vision projector has 2048 input channels and 128 hard tokens.
    FNorm := ReadNpyFloats(BundleMember(Folder, JSONValueText(Manifest.GetValue('hard_norm'))), HardSize);
    FHard := ReadNpyFloats(BundleMember(Folder, JSONValueText(Manifest.GetValue('hard_embeddings'))),
      128 * HardSize);
    FProjection := ReadNpyFloats(BundleMember(Folder, JSONValueText(Manifest.GetValue('projection'))),
      FHiddenSize * HardSize);
    FWeightFile := TMappedModelFile.Create(BundleMember(Folder,
      JSONValueText(Manifest.GetValue('parameters'))));
    LayoutData := ReadModelJSON(BundleMember(Folder, JSONValueText(Manifest.GetValue('layout'))));
    try
      Layout := ModelArray(LayoutData, 'vision parameter layout');
      if FDevice = kDLCUDA then
      begin
        SetLength(FArgShapes, Layout.Count);
        SetLength(FArgOffsets, Layout.Count);
      end
      else
        SetLength(FArgs, Layout.Count + 1);
      for I := 0 to Layout.Count - 1 do
      begin
        Row := ModelObject(Layout.Items[I], 'vision parameter');
        Shape := ModelShape(Row.GetValue('shape'));
        Offset := ModelInt(Row.GetValue('offset'), 'vision parameter offset');
        RequireCpu((Offset mod SizeOf(Single) = 0) and (JSONValueText(Row.GetValue('dtype')) = 'float32'),
          'Invalid vision parameter type or alignment');
        Address := FWeightFile.AtOffset(Offset, TensorElements(Shape) * SizeOf(Single));
        if FDevice = kDLCUDA then
        begin
          FArgShapes[I] := Shape;
          FArgOffsets[I] := Offset;
          Inc(FWeightBytes, TensorElements(Shape) * SizeOf(Single));
        end
        else
          FArgs[I + 1] := MappedFloatTensor(Address, Shape);
      end;
    finally
      LayoutData.Free;
    end;
    FLibraryName := BundleMember(Folder, ModelText(Manifest.GetValue('library')));
    if FCuda <> nil then FCuda.Enter;
    try
      FGraph := TCpuGraph.Create(FLibraryName, FDevice);
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    Data.Free;
  end;
end;

destructor TGemma3nVision.Destroy;
begin
  if FGraph <> nil then
  begin
    if FCuda <> nil then FCuda.Enter;
    try
      FGraph.Free;
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  end;
  ReleaseTensors(FArgs);
  FWeightFile.Free;
  inherited;
end;

procedure RMSNormalize(var Values: TArray<Single>; const Epsilon: Single);
var
  I: Integer;
  Sum, Scale, Item: Double;
begin
  Sum := 0;
  for I := 0 to High(Values) do
  begin
    Item := Values[I];
    Sum := Sum + Sqr(Item);
  end;
  Scale := 1 / Sqrt(Sum / Length(Values) + Epsilon);
  for I := 0 to High(Values) do Values[I] := Values[I] * Scale;
end;

function TGemma3nVision.EndEmbedding(const Token: Integer): TArray<Single>;
var
  Row: TArray<Single>;
  I, J, Offset: Integer;
  Sum, Weight: Double;
begin
  Offset := (Token - FVocabOffset) * Length(FNorm);
  RequireCpu((Offset >= 0) and (Offset + Length(FNorm) <= Length(FHard)),
    'Image end token is outside the projection table');
  Row := Copy(FHard, Offset, Length(FNorm));
  RMSNormalize(Row, FEpsilon);
  for I := 0 to High(Row) do Row[I] := Row[I] * FNorm[I];
  SetLength(Result, FHiddenSize);
  for I := 0 to High(Result) do
  begin
    Sum := 0;
    for J := 0 to High(Row) do
    begin
      Weight := FProjection[I * Length(Row) + J];
      Sum := Sum + Weight * Row[J];
    end;
    Result[I] := Sum;
  end;
  RMSNormalize(Result, FEpsilon);
end;

type
  TFilter = record
    First: Integer;
    Weights: TArray<Integer>;
  end;

function BilinearFilter(const InputSize, OutputSize: Integer): TArray<TFilter>;
const
  FixedOne = 1 shl 22;
var
  Scale, Radius, Center, Sum: Double;
  I, J, First, Finish: Integer;
  Weights: TArray<Double>;
begin
  SetLength(Result, OutputSize);
  Scale := InputSize / OutputSize;
  Radius := Max(1.0, Scale);
  for I := 0 to OutputSize - 1 do
  begin
    Center := (I + 0.5) * Scale;
    First := Max(0, Trunc(Center - Radius + 0.5));
    Finish := Min(InputSize, Trunc(Center + Radius + 0.5));
    SetLength(Weights, Finish - First);
    Result[I].First := First;
    SetLength(Result[I].Weights, Length(Weights));
    Sum := 0;
    for J := 0 to High(Weights) do
    begin
      Weights[J] := Max(0.0, 1.0 - Abs((J + First + 0.5 - Center) / Radius));
      Sum := Sum + Weights[J];
    end;
    for J := 0 to High(Weights) do Result[I].Weights[J] := Trunc(Weights[J] * FixedOne / Sum + 0.5);
  end;
end;

function PreparePixels(const Image: IMat; const Size: Integer;
  const Mean, Std: array of Single): TArray<Single>;
const
  Half = 1 shl 21;
var
  Horizontal, Vertical: TArray<TFilter>;
  Rows: TBytes;
  Source: PByte;
  X, Y, C, J, Sum: Integer;
  Pixel: Single;
begin
  RequireCpu((Int64(Image.Width) * Image.Height <= 64000000) and (Image.Channels = 3),
    'Expected a color image of at most 64 megapixels');
  Horizontal := BilinearFilter(Image.Width, Size);
  Vertical := BilinearFilter(Image.Height, Size);
  SetLength(Rows, Int64(Image.Height) * Size * 3);
  for Y := 0 to Image.Height - 1 do
  begin
    Source := PByte(Image.DataPtr) + NativeInt(Y) * Image.Step;
    for X := 0 to Size - 1 do
      for C := 0 to 2 do
      begin
        Sum := Half;
        for J := 0 to High(Horizontal[X].Weights) do
          Inc(Sum, Horizontal[X].Weights[J] * Source[(Horizontal[X].First + J) * 3 + 2 - C]);
        Rows[(Y * Size + X) * 3 + C] := EnsureRange(Sum shr 22, 0, 255);
      end;
  end;
  SetLength(Result, 3 * Size * Size);
  for C := 0 to 2 do
    for Y := 0 to Size - 1 do
      for X := 0 to Size - 1 do
      begin
        Sum := Half;
        for J := 0 to High(Vertical[Y].Weights) do
          Inc(Sum, Vertical[Y].Weights[J] * Rows[((Vertical[Y].First + J) * Size + X) * 3 + C]);
        Pixel := EnsureRange(Sum shr 22, 0, 255) / 255;
        Result[(C * Size + Y) * Size + X] := (Pixel - Mean[C]) / Std[C];
      end;
end;

function TGemma3nVision.Encode(const FileName: string; const ShouldStop: TQwenStopRequested; out Hash: string;
  out Reused: Boolean; const ReleaseDeviceMemory: TCudaVisionMemory): TArray<Single>;
const
  WorkspaceReserve = UInt64(2) * 1024 * 1024 * 1024;
var
  Stream: TFileStream;
  Bytes: TBytes;
  Raw: TMat;
  Image: IMat;
  Pixels: TArray<Single>;
  Value: TTVMFFIAny;
  Tensor: PDLTensor;
  DeviceArgs: TTensorHandles;
  Workspace: TCudaWorkspace;
  FreeMemory, Required: UInt64;
  Address: Pointer;
  I: Integer;

  function Features(const Reply: TTVMFFIAny): TArray<Single>;
  begin
    RequireCpu(Reply.TypeIndex = kTVMFFITensor, 'TVM vision graph returned an invalid result');
    try
      Tensor := TTVMRuntime.TensorDLTensor(Reply.VObj);
      RequireCpu((Tensor <> nil) and (Tensor^.ndim = 3) and (Tensor^.shape[0] = 1) and
        (Tensor^.shape[1] = FSoftTokens) and (Tensor^.shape[2] = FHiddenSize),
        'Unexpected TVM vision output shape');
      Result := ReadFloatTensor(Reply.VObj);
    finally
      Value := Reply;
      TTVMRuntime.ObjectRelease(Value.VObj);
    end;
  end;

begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    RequireCpu((Stream.Size > 0) and (Stream.Size <= 256 * 1024 * 1024), 'Invalid image file size');
    SetLength(Bytes, Stream.Size);
    Stream.ReadBuffer(Bytes[0], Length(Bytes));
  finally
    Stream.Free;
  end;
  Hash := SHA256Bytes(Bytes);
  Reused := Hash = FLastHash;
  if Reused then Exit(FLastFeatures);
  if Assigned(ShouldStop) and ShouldStop() then Exit(nil);
  if not CV.Module.Loaded then RequireCpu(CVInitialize(FCVLibrary), CV.Module.Error);
  Raw := nil;
  RequireCpu(CV.ImgCodecs.ImDecode(Pointer(Bytes), Length(Bytes), IMREAD_COLOR, @Raw),
    'Could not decode image: ' + FileName);
  Image := WrapMat(Raw);
  RequireCpu(not Image.IsEmpty, 'Empty image: ' + FileName);
  Pixels := PreparePixels(Image, FSize, FMean, FStd);
  if Assigned(ShouldStop) and ShouldStop() then Exit(nil);
  if FDevice = kDLCUDA then
  begin
    FCuda.Enter;
    Workspace := nil;
    try
      Required := FWeightBytes + UInt64(Length(Pixels)) * SizeOf(Single) + WorkspaceReserve;
      FreeMemory := FCuda.FreeBytes;
      if (FreeMemory < Required) and Assigned(ReleaseDeviceMemory) then
      begin
        ReleaseDeviceMemory;
        FreeMemory := FCuda.FreeBytes;
      end;
      RequireCpu(FreeMemory >= Required,
        Format('Gemma 3n vision requires %.2f GiB of free CUDA memory. Available: %.2f GiB.',
          [Required / 1073741824.0, FreeMemory / 1073741824.0]));
      SetLength(DeviceArgs, Length(FArgShapes) + 1);
      DeviceArgs[0] := DeviceTensor([1, 3, FSize, FSize], kDLFloat, 32, Pointer(Pixels), kDLCUDA);
      for I := 0 to High(FArgShapes) do
      begin
        Address := FWeightFile.AtOffset(FArgOffsets[I], TensorElements(FArgShapes[I]) * SizeOf(Single));
        DeviceArgs[I + 1] := DeviceTensor(FArgShapes[I], kDLFloat, 32, Address, kDLCUDA);
      end;
      if Assigned(ShouldStop) and ShouldStop() then Exit(nil);
      Workspace := TCudaWorkspace.Create(FCuda, FLibraryName, WorkspaceReserve);
      Result := Features(Workspace.Run(FGraph, DeviceArgs));
    finally
      try
        try
          if Length(DeviceArgs) > 0 then FCuda.Synchronize;
        finally
          Workspace.Free;
          ReleaseTensors(DeviceArgs);
        end;
      finally
        FCuda.Leave;
      end;
    end;
  end
  else begin
    FArgs[0] := CpuTensor([1, 3, FSize, FSize], kDLFloat, 32, Pointer(Pixels));
    try
      Result := Features(FGraph.Run(FArgs, TCpuGraphStop(ShouldStop)));
    finally
      TTVMRuntime.ObjectRelease(FArgs[0]);
    end;
  end;
  for I := 0 to High(Result) do
    RequireCpu(not IsNan(Result[I]) and not IsInfinite(Result[I]), 'TVM vision output is not finite');
  if Assigned(ShouldStop) and ShouldStop() then Exit(nil);
  FLastHash := Hash;
  FLastFeatures := Result;
end;

end.
