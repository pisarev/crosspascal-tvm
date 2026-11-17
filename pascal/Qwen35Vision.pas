{ ************************************************************************** }
{                                                                            }
{ Qwen35Vision                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit Qwen35Vision;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils, Classes{$ELSE}System.SysUtils, System.Classes{$ENDIF},
  DLPack, TVM, TVM.CpuGraph, TVM.CudaExecution, QwenCuda, Qwen35Weights;

type
  { Static single-image Qwen3.5 vision encoder of a crosspascal-qwen35-vision-v1
    bundle. BF16 weights are expanded to FP32 and uploaded on every Encode and
    released afterwards; only the graph stays resident. Cuda is not owned. }
  TQwen35Vision = class
  private
    FFolder: string;
    FCVLibrary: string;
    FLibraryName: string;
    FBackend: string;
    FCuda: TCudaExecution;
    FDevice: TDLDeviceType;
    FGraph: TCpuGraph;
    FFiles: TQwen35WeightFiles;
    FLocks: TArray<TFileStream>;
    FWeights: TArray<TQwen35Weight>;
    FWeightFloatBytes: UInt64;
    FImageHeight: Integer;
    FImageWidth: Integer;
    FPatch: Integer;
    FTemporal: Integer;
    FMerge: Integer;
    FGridHeight: Integer;
    FGridWidth: Integer;
    FPatchWidth: Integer;
    FSoftTokens: Integer;
    FHiddenSize: Integer;
    FLastHash: string;
    FLastFeatures: TArray<Single>;
    procedure EnsureCudaFree(const PatchElements: Int64);
    procedure LoadWeights(var Args: TTensorHandles; const ShouldStop: TQwenStopRequested);
    function Features(Reply: TTVMFFIAny): TArray<Single>;
  public
    constructor Create(const Folder, CVLibrary: string; const HiddenSize: Integer;
      const Cuda: TCudaExecution = nil; const ManifestName: string = 'vision.json');
    destructor Destroy; override;
    function Encode(const FileName: string; const ShouldStop: TQwenStopRequested; out Hash: string;
      out Reused: Boolean): TArray<Single>;
    property GridHeight: Integer read FGridHeight;
    property GridWidth: Integer read FGridWidth;
    property MergeSize: Integer read FMerge;
    property SoftTokens: Integer read FSoftTokens;
    property HiddenSize: Integer read FHiddenSize;
    property Backend: string read FBackend;
  end;

implementation

uses
  {$IFDEF FPC}Math, Generics.Collections{$ELSE}System.Math, System.Generics.Collections{$ENDIF},
  CrossPascal.Json, CrossPascal.SHA256, CrossVision.Mat, TVM.ImageInput;

const
  ManifestSchema = 'crosspascal-qwen35-vision-v1';
  PatchesInput = 'patches';
  EmbeddingsOutput = 'image_embeddings';
  VisualPrefix = 'model.visual.';
  { ONNX TensorProto.FLOAT; WriteQwen35VisionBundle writes OnnxFloat as 'dtype'. }
  OnnxFloatType = 1;
  { TQwen35WeightFiles.Weight: encoding 'bf16' -> Format 30. }
  BF16Format = 30;
  { ResizeColorImage accepts at most 4096 per side. }
  MaxImageSide = 4096;
  WorkspaceReserve = UInt64(512) * 1024 * 1024;
  GiB = 1073741824.0;
  MiB = 1048576.0;

procedure StopIfRequested(const Stop: TQwenStopRequested);
begin
  if Assigned(Stop) and Stop() then raise ECpuGraphStopped.Create('Generation stopped');
end;

function ManifestText(const Owner: TJSONObject; const Name: string): string;
var
  Value: TJSONValue;
begin
  Value := Owner.GetValue(Name);
  RequireCpu(Value is TJSONString, 'Qwen vision manifest field is not text: ' + Name);
  Result := JSONValueText(Value);
  RequireCpu(Result <> '', 'Qwen vision manifest field is empty: ' + Name);
end;

function ManifestInt(const Owner: TJSONObject; const Name: string;
  const MinValue, MaxValue: Int64): Integer;
var
  Value: Int64;
begin
  Value := ModelInt(Owner.GetValue(Name), Name);
  RequireCpu((Value >= MinValue) and (Value <= MaxValue),
    'Qwen vision manifest value is out of range: ' + Name);
  Result := Value;
end;

function SameShape(const Left, Right: array of Int64): Boolean;
var
  I: Integer;
begin
  Result := Length(Left) = Length(Right);
  if Result then
    for I := 0 to High(Left) do
      if Left[I] <> Right[I] then Exit(False);
end;

function ValidDigest(const Digest: string): Boolean;
var
  I: Integer;
begin
  Result := Length(Digest) = 64;
  if Result then
    for I := 1 to 64 do
      if not (((Digest[I] >= '0') and (Digest[I] <= '9')) or
        ((Digest[I] >= 'a') and (Digest[I] <= 'f'))) then Exit(False);
end;

{ Hashes every files[] member and keeps it open with fmShareDenyWrite, so the
  verified bytes cannot be rewritten while this instance maps or loads them. }
procedure VerifyFiles(const Folder: string; const Files: TJSONArray;
  const LibraryFile, WeightsFile: string; var Locks: TArray<TFileStream>; out WeightsBytes: Int64);
var
  Seen: TDictionary<string, Boolean>;
  Entry: TJSONObject;
  Name, Digest: string;
  Bytes: Int64;
  Stream: TFileStream;
  I: Integer;
  HasLibrary: Boolean;
begin
  WeightsBytes := -1;
  HasLibrary := False;
  Seen := TDictionary<string, Boolean>.Create;
  try
    for I := 0 to Files.Count - 1 do
    begin
      Entry := ModelObject(Files.Items[I], 'Qwen vision file entry');
      Name := ManifestText(Entry, 'file');
      RequireCpu(not Seen.ContainsKey(LowerCase(Name)), 'Duplicate Qwen vision file entry: ' + Name);
      Seen.Add(LowerCase(Name), True);
      Bytes := ModelInt(Entry.GetValue('bytes'), 'Qwen vision file bytes');
      Digest := ManifestText(Entry, 'sha256');
      RequireCpu((Bytes > 0) and ValidDigest(Digest), 'Invalid Qwen vision file entry: ' + Name);
      SetLength(Locks, Length(Locks) + 1);
      Locks[High(Locks)] := TFileStream.Create(BundleMember(Folder, Name), fmOpenRead or fmShareDenyWrite);
      Stream := Locks[High(Locks)];
      RequireCpu(Stream.Size = Bytes, 'Qwen vision artifact size changed: ' + Name);
      RequireCpu(SHA256Stream(Stream) = Digest, 'Qwen vision artifact changed: ' + Name);
      if Name = LibraryFile then HasLibrary := True;
      if Name = WeightsFile then WeightsBytes := Bytes;
    end;
    RequireCpu(HasLibrary and (WeightsBytes > 0),
      'Qwen vision manifest does not hash its library and weights');
  finally
    Seen.Free;
  end;
end;

{ CrossPascal profile, same policy as QwenVLCpu.ImageFeatures: keep the aspect
  ratio, fit and center into the manifest canvas, pad with RGB 0, then normalize
  (raw / 255 - 0.5) / 0.5. Rows: blockRow, blockCol, intraRow, intraCol; columns:
  channel, temporal (same frame repeated), patchY, patchX. }
function PreparePatches(const Image: IMat;
  const CanvasHeight, CanvasWidth, Patch, Temporal, Merge: Integer): TArray<Single>;
var
  Pixels: TBytes;
  Table: array[0..255] of Single;
  W, H, PadX, PadY, GridH, GridW, BY, BX, MY, MX, C, T, X, Y, RX, RY, I: Integer;
  At: NativeInt;
begin
  for I := 0 to 255 do Table[I] := (I / 255 - 0.5) / 0.5;
  if Int64(Image.Width) * CanvasHeight >= Int64(Image.Height) * CanvasWidth then
  begin
    W := CanvasWidth;
    H := EnsureRange(Round(Int64(Image.Height) * CanvasWidth / Image.Width), 1, CanvasHeight);
  end
  else begin
    H := CanvasHeight;
    W := EnsureRange(Round(Int64(Image.Width) * CanvasHeight / Image.Height), 1, CanvasWidth);
  end;
  PadX := Round((CanvasWidth - W) / 2);
  PadY := Round((CanvasHeight - H) / 2);
  { Packed RGB bytes, H rows of W pixels. }
  Pixels := ResizeColorImage(Image, W, H, True);
  RequireCpu(Length(Pixels) = W * H * 3, 'Unexpected resized image size');
  GridH := CanvasHeight div Patch;
  GridW := CanvasWidth div Patch;
  SetLength(Result, Int64(GridH) * GridW * 3 * Temporal * Patch * Patch);
  At := 0;
  for BY := 0 to GridH div Merge - 1 do
    for BX := 0 to GridW div Merge - 1 do
      for MY := 0 to Merge - 1 do
        for MX := 0 to Merge - 1 do
          for C := 0 to 2 do
            for T := 0 to Temporal - 1 do
              for Y := 0 to Patch - 1 do
              begin
                RY := (BY * Merge + MY) * Patch + Y - PadY;
                for X := 0 to Patch - 1 do
                begin
                  RX := (BX * Merge + MX) * Patch + X - PadX;
                  if (RY >= 0) and (RY < H) and (RX >= 0) and (RX < W) then
                    Result[At] := Table[Pixels[(RY * W + RX) * 3 + C]]
                  else
                    Result[At] := Table[0];
                  Inc(At);
                end;
              end;
  RequireCpu(At = Length(Result), 'Qwen vision patch layout mismatch');
end;

constructor TQwen35Vision.Create(const Folder, CVLibrary: string; const HiddenSize: Integer;
  const Cuda: TCudaExecution; const ManifestName: string);
var
  Data: TJSONValue;
  Manifest, Entry, WeightMap, Descriptor: TJSONObject;
  Inputs: TJSONArray;
  Seen: TDictionary<string, Boolean>;
  Device, Name, LibraryFile, WeightsFile: string;
  Shape: TArray<Int64>;
  WeightsBytes, Total: Int64;
  I: Integer;
begin
  inherited Create;
  RequireCpu(HiddenSize > 0, 'Invalid Qwen text hidden size');
  { Absolute: TCudaWorkspace loads the library with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR. }
  FFolder := ExcludeTrailingPathDelimiter(ExpandFileName(Folder));
  FCVLibrary := CVLibrary;
  FHiddenSize := HiddenSize;
  Seen := nil;
  Data := ReadModelJSON(BundleMember(FFolder, ManifestName));
  try
    Manifest := ModelObject(Data, 'Qwen vision manifest');
    RequireCpu(ManifestText(Manifest, 'schema') = ManifestSchema, 'Unsupported Qwen vision manifest schema');
    Device := ManifestText(Manifest, 'device');
    if Cuda <> nil then
    begin
      RequireCpu(Device = 'cuda', 'The Qwen vision bundle targets ' + Device + ', but CUDA was selected');
      Cuda.RequireArchitecture(ManifestText(Manifest, 'cuda_arch'));
      FCuda := Cuda;
      FDevice := kDLCUDA;
      FBackend := 'TVM CUDA FP32';
    end
    else begin
      RequireCpu((Device = 'cpu') and (Manifest.GetValue('cuda_arch') = nil),
        'The Qwen vision bundle targets ' + Device + ', but CPU was selected');
      FDevice := kDLCPU;
      FBackend := 'TVM CPU FP32';
    end;
    FImageHeight := ManifestInt(Manifest, 'image_height', 1, MaxImageSide);
    FImageWidth := ManifestInt(Manifest, 'image_width', 1, MaxImageSide);
    FPatch := ManifestInt(Manifest, 'patch_size', 1, MaxImageSide);
    FTemporal := ManifestInt(Manifest, 'temporal_patch_size', 1, 16);
    FMerge := ManifestInt(Manifest, 'spatial_merge_size', 1, 16);
    FGridHeight := ManifestInt(Manifest, 'grid_height', 1, MaxImageSide);
    FGridWidth := ManifestInt(Manifest, 'grid_width', 1, MaxImageSide);
    RequireCpu((FGridHeight * FPatch = FImageHeight) and (FGridWidth * FPatch = FImageWidth) and
      (FGridHeight mod FMerge = 0) and (FGridWidth mod FMerge = 0),
      'Qwen vision grid does not tile the image profile');
    RequireCpu(ManifestInt(Manifest, 'hidden_size', 1, MaxInt) = HiddenSize,
      'Qwen vision hidden size does not match the text model');
    FSoftTokens := (FGridHeight div FMerge) * (FGridWidth div FMerge);
    RequireCpu((Int64(3) * FTemporal * FPatch * FPatch * FGridHeight * FGridWidth <= MaxInt) and
      (Int64(FSoftTokens) * FHiddenSize <= MaxInt), 'Qwen vision profile is too large');
    FPatchWidth := 3 * FTemporal * FPatch * FPatch;
    RequireCpu((ManifestText(Manifest, 'patches_input') = PatchesInput) and
      (ManifestText(Manifest, 'embeddings_output') = EmbeddingsOutput),
      'Unexpected Qwen vision graph interface');
    LibraryFile := ManifestText(Manifest, 'library');
    WeightsFile := ManifestText(Manifest, 'weights_file');
    FLibraryName := BundleMember(FFolder, LibraryFile);
    BundleMember(FFolder, WeightsFile);
    VerifyFiles(FFolder, ModelArray(Manifest.GetValue('files'), 'Qwen vision files'),
      LibraryFile, WeightsFile, FLocks, WeightsBytes);
    Inputs := ModelArray(Manifest.GetValue('inputs'), 'Qwen vision inputs');
    WeightMap := ModelObject(Manifest.GetValue('weights'), 'Qwen vision weights');
    RequireCpu((Inputs.Count >= 2) and (WeightMap.Count = Inputs.Count - 1),
      'Qwen vision inputs and weight descriptors differ');
    FFiles := TQwen35WeightFiles.Create(FFolder, '');
    SetLength(FWeights, Inputs.Count - 1);
    Seen := TDictionary<string, Boolean>.Create;
    Total := 0;
    for I := 0 to Inputs.Count - 1 do
    begin
      Entry := ModelObject(Inputs.Items[I], 'Qwen vision input');
      Name := ManifestText(Entry, 'name');
      RequireCpu(not Seen.ContainsKey(Name), 'Duplicate Qwen vision input: ' + Name);
      Seen.Add(Name, True);
      RequireCpu(ModelInt(Entry.GetValue('dtype'), 'input dtype') = OnnxFloatType,
        'Qwen vision input is not FP32: ' + Name);
      Shape := ModelShape(Entry.GetValue('shape'));
      if I = 0 then
      begin
        RequireCpu((Name = PatchesInput) and
          SameShape(Shape, [Int64(FGridHeight) * FGridWidth, Int64(FPatchWidth)]),
          'Qwen vision patches input differs from the image profile');
        Continue;
      end;
      RequireCpu((Length(Name) > Length(VisualPrefix)) and (Copy(Name, 1, Length(VisualPrefix)) = VisualPrefix),
        'Unexpected Qwen vision input: ' + Name);
      Descriptor := ModelObject(WeightMap.GetValue(Name), 'Qwen vision weight ' + Name);
      RequireCpu((ManifestText(Descriptor, 'file') = WeightsFile) and (ManifestText(Descriptor, 'encoding') = 'bf16'),
        'Unexpected Qwen vision weight storage: ' + Name);
      FWeights[I - 1] := FFiles.Weight(Descriptor);
      RequireCpu(
        (FWeights[I - 1].Format = BF16Format) and SameShape(FWeights[I - 1].Shape, Shape) and (ModelInt(Descriptor.GetValue('elements'), 'weight elements') = FWeights[I - 1].Elements) and (FWeights[I - 1].Offset = Total),
        'Qwen vision weight layout differs: ' + Name
      );
      Inc(Total, FWeights[I - 1].Bytes);
      Inc(FWeightFloatBytes, UInt64(FWeights[I - 1].Elements) * SizeOf(Single));
    end;
    RequireCpu(Total = WeightsBytes, 'Qwen vision weights do not cover ' + WeightsFile);
    if FCuda <> nil then FCuda.Enter;
    try
      FGraph := TCpuGraph.Create(FLibraryName, FDevice);
    finally
      if FCuda <> nil then FCuda.Leave;
    end;
  finally
    Seen.Free;
    Data.Free;
  end;
end;

destructor TQwen35Vision.Destroy;
var
  I: Integer;
begin
  try
    if FGraph <> nil then
    begin
      if FCuda <> nil then FCuda.Enter;
      try
        FreeAndNil(FGraph);
      finally
        if FCuda <> nil then FCuda.Leave;
      end;
    end;
  finally
    FFiles.Free;
    for I := 0 to High(FLocks) do FLocks[I].Free;
    inherited;
  end;
end;

procedure TQwen35Vision.EnsureCudaFree(const PatchElements: Int64);
var
  Required, Available: UInt64;
begin
  Required := FWeightFloatBytes + UInt64(PatchElements) * SizeOf(Single) + WorkspaceReserve;
  Available := FCuda.FreeBytes;
  if Available < Required then
    raise ECpuGraph.CreateFmt('Qwen3.5 vision needs %.2f GiB of free CUDA memory on %s ' +
      '(%.2f GiB FP32 weights, %.1f MiB patches, %.0f MiB workspace); %.2f GiB is free.',
      [Required / GiB, FCuda.DeviceName, FWeightFloatBytes / GiB,
       PatchElements * SizeOf(Single) / MiB, WorkspaceReserve / MiB, Available / GiB]);
end;

{ Exact BF16 -> FP32 expansion; DeviceTensor copies, so each host array is
  dropped immediately and the mapped BF16 pages leave the working set. }
procedure TQwen35Vision.LoadWeights(var Args: TTensorHandles; const ShouldStop: TQwenStopRequested);
var
  Floats: TArray<Single>;
  I: Integer;
begin
  for I := 0 to High(FWeights) do
  begin
    StopIfRequested(ShouldStop);
    Floats := FFiles.FloatRange(FWeights[I], 0, FWeights[I].Elements);
    Args[I + 1] := DeviceTensor(FWeights[I].Shape, kDLFloat, 32, Pointer(Floats), FDevice);
    Floats := nil;
    FFiles.ReleaseResidentRange(FWeights[I], 0, FWeights[I].Elements);
  end;
end;

function TQwen35Vision.Features(Reply: TTVMFFIAny): TArray<Single>;
var
  Tensor: PDLTensor;
  I: Integer;
begin
  if Reply.TypeIndex <> kTVMFFITensor then
  begin
    if Reply.TypeIndex >= kTVMFFIStaticObjectBegin then TTVMRuntime.ObjectRelease(Reply.VObj);
    raise ECpuGraph.Create('Qwen vision graph did not return a tensor');
  end;
  try
    Tensor := TTVMRuntime.TensorDLTensor(Reply.VObj);
    RequireCpu((Tensor <> nil) and (Tensor^.ndim = 2) and (Tensor^.shape[0] = FSoftTokens) and
      (Tensor^.shape[1] = FHiddenSize), 'Unexpected Qwen vision output shape');
    { ReadFloatTensor requires float32. }
    Result := ReadFloatTensor(Reply.VObj);
  finally
    TTVMRuntime.ObjectRelease(Reply.VObj);
  end;
  RequireCpu(Length(Result) = FSoftTokens * FHiddenSize, 'Incomplete Qwen vision output');
  for I := 0 to High(Result) do
    RequireCpu(not IsNan(Result[I]) and not IsInfinite(Result[I]), 'Qwen vision output is not finite');
end;

function TQwen35Vision.Encode(const FileName: string; const ShouldStop: TQwenStopRequested;
  out Hash: string; out Reused: Boolean): TArray<Single>;
var
  Bytes: TBytes;
  Image: IMat;
  Patches: TArray<Single>;
  Args: TTensorHandles;
  Workspace: TCudaWorkspace;
  PatchShape: TArray<Int64>;
begin
  Result := nil;
  Bytes := ReadImageBytes(FileName);
  Hash := SHA256Bytes(Bytes);
  { The profile is fixed per instance, so the image digest is the whole key. }
  StopIfRequested(ShouldStop);
  Reused := (FLastHash <> '') and (Hash = FLastHash);
  if Reused then Exit(Copy(FLastFeatures));
  StopIfRequested(ShouldStop);
  Image := DecodeColorImage(Bytes, FCVLibrary);
  Bytes := nil;
  Patches := PreparePatches(Image, FImageHeight, FImageWidth, FPatch, FTemporal, FMerge);
  Image := nil;
  StopIfRequested(ShouldStop);
  PatchShape := TArray<Int64>.Create(Int64(FGridHeight) * FGridWidth, FPatchWidth);
  SetLength(Args, Length(FWeights) + 1);
  if FCuda = nil then
  begin
    try
      Args[0] := CpuTensor(PatchShape, kDLFloat, 32, Pointer(Patches));
      Patches := nil;
      LoadWeights(Args, ShouldStop);
      StopIfRequested(ShouldStop);
      Result := Features(FGraph.Run(Args, TCpuGraphStop(ShouldStop)));
    finally
      ReleaseTensors(Args);
    end;
  end
  else begin
    Workspace := nil;
    FCuda.Enter;
    try
      try
        EnsureCudaFree(Length(Patches));
        Args[0] := DeviceTensor(PatchShape, kDLFloat, 32, Pointer(Patches), kDLCUDA);
        Patches := nil;
        LoadWeights(Args, ShouldStop);
        StopIfRequested(ShouldStop);
        Workspace := TCudaWorkspace.Create(FCuda, FLibraryName, WorkspaceReserve);
        Result := Features(Workspace.Run(FGraph, Args, TCpuGraphStop(ShouldStop)));
      finally
        try
          FCuda.Synchronize;
        finally
          Workspace.Free;
          ReleaseTensors(Args);
        end;
      end;
    finally
      FCuda.Leave;
    end;
  end;
  StopIfRequested(ShouldStop);
  FLastFeatures := Copy(Result);
  FLastHash := Hash;
end;

end.
