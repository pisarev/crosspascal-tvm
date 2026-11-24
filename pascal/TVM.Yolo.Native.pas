{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Native                                                            }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Native;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  {$IFDEF FPC}SysUtils, SyncObjs{$ELSE}System.SysUtils, System.SyncObjs{$ENDIF},
  DLPack, TVM, TVM.CpuGraph, TVM.Yolo.Profile, TVM.Yolo.Classification,
  TVM.Yolo.Detections, TVM.Yolo.Oriented, TVM.Yolo.Masks;

type
  TYoloPreparedImage = record
    Width: Integer;
    Height: Integer;
    Pixels: TArray<Single>;
  end;

  { All coordinates and maps are restored to the original image size.
    Instance mask planes contain 0/1, semantic maps contain class IDs, and
    depth planes contain the calibrated model estimates in meters. }
  TYoloNativeResult = record
    Task: TYoloTask;
    Width: Integer;
    Height: Integer;
    Classifications: TYoloClassifications;
    Detections: TYoloRawDetections;
    Oriented: TYoloOrientedDetections;
    InstanceMasks: TYoloInstanceMasks;
    ClassMap: TBytes;
    Depth: TArray<Single>;
  end;

  { The application selects and validates its TVM runtime before construction.
    One model instance reuses its graph/input tensor and serializes inference.
    Artifact identity and device compatibility belong to the bundle loader. }
  TYoloNativeModel = class
  private
    FProfile: TYoloProfile;
    FDevice: TDLDeviceType;
    FGraph: TCpuGraph;
    FInput: TTVMFFIObjectHandle;
    FLock: TCriticalSection;
    FResultByteLimit: Int64;
    FShouldStop: TCpuGraphStop;
    procedure ValidateOutput(const Handle: TTVMFFIObjectHandle; const Index: Integer);
    procedure RequireResultSpace(const Width, Height, Planes, BytesPerPixel: Integer);
  public
    constructor Create(const Profile: TYoloProfile; const ModelLibrary: string;
      const Device: TDLDeviceType; const ResultByteLimit: Int64 = 512 * 1024 * 1024;
      const ShouldStop: TCpuGraphStop = nil);
    destructor Destroy; override;
    function Infer(const Image: TYoloPreparedImage): TYoloNativeResult;
    function Profile: TYoloProfile;
  end;

function CopyYoloProfile(const Source: TYoloProfile): TYoloProfile;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.Yolo.OutputGeometry, TVM.Yolo.RunOptions,
  TVM.Yolo.Semantic;

function CopyYoloProfile(const Source: TYoloProfile): TYoloProfile;
var
  I: Integer;
begin
  Result := Source;
  Result.ClassNames := Copy(Source.ClassNames);
  Result.Mean := Copy(Source.Mean);
  Result.Std := Copy(Source.Std);
  Result.Outputs := Copy(Source.Outputs);
  for I := 0 to High(Result.Outputs) do Result.Outputs[I].Shape := Copy(Source.Outputs[I].Shape);
end;

constructor TYoloNativeModel.Create(const Profile: TYoloProfile; const ModelLibrary: string;
  const Device: TDLDeviceType; const ResultByteLimit: Int64; const ShouldStop: TCpuGraphStop);
var
  I: Integer;
  RawBytes: Int64;
begin
  inherited Create;
  Profile.Validate;
  RequireCpu((Device in [kDLCPU, kDLCUDA]) and (ResultByteLimit > 0) and
    (ResultByteLimit <= High(Integer)), 'Invalid YOLO device or result memory limit');
  RawBytes := 0;
  for I := 0 to High(Profile.Outputs) do Inc(RawBytes, Profile.Outputs[I].ByteCount);
  RequireCpu((RawBytes <= ResultByteLimit) and
    (Int64(Profile.InputWidth) * Profile.InputHeight * 3 * SizeOf(Single) <= ResultByteLimit),
    'YOLO input or raw outputs exceed the configured result memory limit');
  FProfile := CopyYoloProfile(Profile);
  FDevice := Device;
  FResultByteLimit := ResultByteLimit;
  FShouldStop := ShouldStop;
  FLock := TCriticalSection.Create;
  FGraph := TCpuGraph.Create(ModelLibrary, FDevice);
  FInput := DeviceTensor([1, 3, FProfile.InputHeight, FProfile.InputWidth], kDLFloat, 32, nil, FDevice);
end;

destructor TYoloNativeModel.Destroy;
begin
  try
    FGraph.Free;
  finally
    TTVMRuntime.ObjectRelease(FInput);
    FLock.Free;
    inherited;
  end;
end;

function TYoloNativeModel.Profile: TYoloProfile;
begin
  Result := CopyYoloProfile(FProfile);
end;

procedure TYoloNativeModel.RequireResultSpace(const Width, Height, Planes, BytesPerPixel: Integer);
var
  Pixels: Int64;
begin
  RequireCpu((Width > 0) and (Height > 0) and (Planes >= 0) and (BytesPerPixel > 0),
    'Invalid YOLO result dimensions');
  Pixels := Int64(Width) * Height;
  if Planes = 0 then Exit;
  RequireCpu((Pixels <= FResultByteLimit div BytesPerPixel) and
    (Planes <= FResultByteLimit div (Pixels * BytesPerPixel)),
    'YOLO result exceeds the configured result memory limit');
end;

procedure TYoloNativeModel.ValidateOutput(const Handle: TTVMFFIObjectHandle; const Index: Integer);
var
  Tensor: PDLTensor;
  Shape: TArray<Int64>;
  I, Code, Bits: Integer;
begin
  Tensor := TTVMRuntime.TensorDLTensor(Handle);
  RequireCpu((Tensor <> nil) and (Tensor.dtype.lanes = 1) and
    (Tensor.device.device_type = FDevice) and (Tensor.device.device_id = 0),
    'YOLO output has an unexpected storage device');
  Shape := TTVMRuntime.GetTensorShape(Tensor);
  RequireCpu(Length(Shape) = Length(FProfile.Outputs[Index].Shape),
    'YOLO output rank differs from its profile');
  for I := 0 to High(Shape) do
    RequireCpu(Shape[I] = FProfile.Outputs[Index].Shape[I], 'YOLO output shape differs from its profile');
  Code := Ord(kDLFloat);
  Bits := 32;
  if FProfile.Outputs[Index].DataType = yttUInt8 then
  begin
    Code := Ord(kDLUInt);
    Bits := 8;
  end;
  RequireCpu((Tensor.dtype.code = Code) and (Tensor.dtype.bits = Bits),
    'YOLO output data type differs from its profile');
end;

function TYoloNativeModel.Infer(const Image: TYoloPreparedImage): TYoloNativeResult;
var
  Value, Count: TTVMFFIAny;
  Handles: TTensorHandles;
  Floats: TArray<TArray<Single>>;
  Bytes: TBytes;
  Error: TTVMCallError;
  I, J: Integer;
begin
  FLock.Enter;
  try
    CheckYoloRunStop(FShouldStop);
    RequireCpu((Image.Width > 0) and (Image.Height > 0) and
      (Int64(Image.Width) * Image.Height <= High(Integer)) and
      (Length(Image.Pixels) = Int64(FProfile.InputWidth) * FProfile.InputHeight * 3),
      'YOLO prepared image dimensions differ from its profile');
    for I := 0 to High(Image.Pixels) do
      RequireCpu(not IsNan(Image.Pixels[I]) and not IsInfinite(Image.Pixels[I]),
        'YOLO prepared image contains nonfinite input');
    Result := Default(TYoloNativeResult);
    Result.Task := FProfile.Task;
    Result.Width := Image.Width;
    Result.Height := Image.Height;
    RequireTVM(TTVMRuntime.CopyBytesToTensor(FInput, Pointer(Image.Pixels),
      Int64(Length(Image.Pixels)) * SizeOf(Single), @Error), @Error);
    Value := FGraph.Run([FInput], FShouldStop);
    try
      RequireCpu(Value.TypeIndex >= kTVMFFIStaticObjectBegin,
        'YOLO graph did not return a tensor or tensor tuple');
      SetLength(Handles, Length(FProfile.Outputs));
      try
        if Value.TypeIndex = kTVMFFITensor then
        begin
          RequireCpu(Length(Handles) = 1, 'YOLO graph returned too few outputs');
          Handles[0] := Value.VObj; TTVMRuntime.ObjectRetain(Handles[0]);
        end
        else begin
          RequireTVM(TTVMRuntime.CallGlobal('ffi.ArraySize', [TTVMRuntime.AnyObject(Value.VObj)], Count, @Error), @Error);
          RequireCpu((Count.TypeIndex = kTVMFFIInt) and (Count.VInt64 = Length(Handles)),
            'YOLO graph output count differs from its profile');
          for I := 0 to High(Handles) do Handles[I] := TensorItem(Value.VObj, I);
        end;
        SetLength(Floats, Length(Handles));
        for I := 0 to High(Handles) do
        begin
          ValidateOutput(Handles[I], I);
          if FProfile.Outputs[I].DataType = yttFloat32 then
          begin
            Floats[I] := ReadFloatTensor(Handles[I]);
            for J := 0 to High(Floats[I]) do
              RequireCpu(not IsNan(Floats[I][J]) and not IsInfinite(Floats[I][J]),
                'YOLO graph returned nonfinite output');
          end
          else begin
            SetLength(Bytes, FProfile.Outputs[I].ByteCount);
            RequireTVM(TTVMRuntime.CopyBytesFromTensor(Handles[I], Pointer(Bytes), Length(Bytes), @Error), @Error);
            for J := 0 to High(Bytes) do
              RequireCpu(Bytes[J] < FProfile.ClassCount, 'YOLO class map contains an invalid class ID');
          end;
        end;
      finally
        ReleaseTensors(Handles);
      end;
    finally
      if Value.TypeIndex >= kTVMFFIStaticObjectBegin then TTVMRuntime.ObjectRelease(Value.VObj);
    end;
    CheckYoloRunStop(FShouldStop);
    case FProfile.Task of
      ytClassification:
        Result.Classifications := DecodeYoloClassification(Floats[0], FProfile.ClassScoreKind, FProfile.TopK);
      ytDetection, ytInstances, ytPose:
        begin
          Result.Detections := DecodeYoloDetections(Floats[0], FProfile.BoxCount,
            FProfile.FeatureCount, FProfile.Detections);
          if FProfile.Task = ytPose then
            Result.Detections := ProjectYoloDetections(Result.Detections,
              FProfile.ImageTransform(Image.Width, Image.Height), FProfile.KeypointCount,
              FProfile.KeypointDimensions)
          else
            Result.Detections := ProjectYoloDetections(Result.Detections,
              FProfile.ImageTransform(Image.Width, Image.Height));
          if FProfile.Task = ytInstances then
          begin
            RequireResultSpace(Image.Width, Image.Height, Length(Result.Detections), 1);
            Result.InstanceMasks := RestoreYoloInstanceMasks(
              Floats[1],
              FProfile.Outputs[1].Shape[3],
              FProfile.Outputs[1].Shape[2],
              FProfile.Detections.ExtraCount,
              Result.Detections,
              Image.Width,
              Image.Height,
              FProfile.InstanceMaskKind,
              FProfile.PrototypeStride
            );
          end;
        end;
      ytOriented:
        begin
          Result.Oriented := DecodeYoloOrientedDetections(Floats[0], FProfile.BoxCount,
            FProfile.FeatureCount, FProfile.Detections, FProfile.ClassOffset);
          Result.Oriented := ProjectYoloOrientedDetections(Result.Oriented,
            FProfile.ImageTransform(Image.Width, Image.Height));
        end;
      ytSemantic:
        begin
          if FProfile.Outputs[0].DataType = yttFloat32 then
          begin
            RequireResultSpace(Image.Width, Image.Height, 1, 9);
            Result.ClassMap := RestoreYoloSemanticLogits(Floats[0], FProfile.Outputs[0].Shape[3],
              FProfile.Outputs[0].Shape[2], FProfile.ClassCount, FProfile.InputWidth,
              FProfile.InputHeight, Image.Width, Image.Height, FShouldStop);
          end
          else begin
            RequireResultSpace(Image.Width, Image.Height, 1, 1);
            Result.ClassMap := RestoreYoloClassMap(Bytes, FProfile.Outputs[0].Shape[2],
              FProfile.Outputs[0].Shape[1], FProfile.InputWidth, FProfile.InputHeight, Image.Width,
              Image.Height);
          end;
        end;
      ytDepth:
        begin
          RequireResultSpace(Image.Width, Image.Height, 1, SizeOf(Single));
          for I := 0 to High(Floats[0]) do
            RequireCpu(Floats[0][I] > 0, 'YOLO depth must be positive');
          Result.Depth := RestoreYoloDepth(Floats[0], FProfile.Outputs[0].Shape[3],
            FProfile.Outputs[0].Shape[2], FProfile.InputWidth, FProfile.InputHeight, Image.Width,
            Image.Height);
        end;
    end;
    CheckYoloRunStop(FShouldStop);
  finally
    FLock.Leave;
  end;
end;

end.
