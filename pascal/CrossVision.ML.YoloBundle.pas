{ ************************************************************************** }
{                                                                            }
{ CrossVision.ML.YoloBundle                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossVision.ML.YoloBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  CrossVision.ML, CrossVision.ML.Depth, TVM.Yolo.Profile, TVM.Yolo.RunOptions;

type
  IYoloBundle = interface
    ['{9FBBF26C-9807-4E78-B7E4-25A5CD9109D5}']
    function Profile: TYoloProfile;
    function OpenClassifier: IClassifier;
    function OpenDetector: IDetector;
    function OpenInstanceSegmenter: IInstanceSegmenter;
    function OpenPoseEstimator: IPoseEstimator;
    function OpenOrientedDetector: IOrientedDetector;
    function OpenSemanticSegmenter: ISemanticSegmenter;
    function OpenDepthEstimator: IDepthEstimator;
  end;

{ ExpectedProfile and the manifest pin come from the application's trusted
  catalog. Self-consistency of an embedded manifest alone is insufficient.
  Manifest and model files remain held while a bundle or its models are alive. }
function LoadYoloBundle(const BundleDirectory, ManifestSHA256: string; const ManifestBytes: Int64;
  const RuntimeDirectory: string; const ExpectedProfile: TYoloProfile;
  const ResultByteLimit: Int64 = 512 * 1024 * 1024): IYoloBundle;

{ The trusted profile and bundle identity remain unchanged. Run settings alter
  only postprocessing. A callback owner must outlive the returned bundle and
  every opened model; callbacks execute on the owning inference thread. }
function LoadYoloBundleForRun(const BundleDirectory, ManifestSHA256: string;
  const ManifestBytes: Int64; const RuntimeDirectory: string; const ExpectedProfile: TYoloProfile;
  const Options: TYoloRunOptions; const ResultByteLimit: Int64 = 512 * 1024 * 1024): IYoloBundle;

implementation

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, DLPack, CrossPascal.Json,
  CrossVision.ML.Yolo, TVM.Yolo.Native, TVM.Yolo.Identity, TVM.NativeFiles, TVM.NativeDevice,
  TVM.NativeRuntime;

type
  TYoloBundle = class(TInterfacedObject, IYoloBundle)
  private
    FManifest: INativeFileLease;
    FArtifact: INativeFileLease;
    FProfile: TYoloProfile;
    FRunProfile: TYoloProfile;
    FOptions: TYoloRunOptions;
    FDevice: TDLDeviceType;
    FResultByteLimit: Int64;
  public
    constructor Create(const BundleDirectory, ManifestSHA256: string; const ManifestBytes: Int64;
      const RuntimeDirectory: string; const ExpectedProfile: TYoloProfile;
      const Options: TYoloRunOptions; const ResultByteLimit: Int64);
    function Profile: TYoloProfile;
    function OpenClassifier: IClassifier;
    function OpenDetector: IDetector;
    function OpenInstanceSegmenter: IInstanceSegmenter;
    function OpenPoseEstimator: IPoseEstimator;
    function OpenOrientedDetector: IOrientedDetector;
    function OpenSemanticSegmenter: ISemanticSegmenter;
    function OpenDepthEstimator: IDepthEstimator;
  end;

constructor TYoloBundle.Create(const BundleDirectory, ManifestSHA256: string; const ManifestBytes: Int64;
  const RuntimeDirectory: string; const ExpectedProfile: TYoloProfile; const Options: TYoloRunOptions;
  const ResultByteLimit: Int64);
var
  Root, Artifact, Provenance, Runtime: TJSONObject;
  Identity, Device, Architecture: string;
begin
  inherited Create;
  ExpectedProfile.Validate;
  Options.Validate;
  CheckYoloRunStop(Options.ShouldStop);
  NativeRequire((ResultByteLimit > 0) and (ResultByteLimit <= High(Integer)),
    'Invalid YOLO bundle result memory limit');
  FManifest := HoldNativeFile(BundleDirectory, 'manifest.json', ManifestSHA256, ManifestBytes);
  Root := ReadNativeObject(FManifest);
  try
    NativeRequire(NativeText(Root, 'schema') = 'crosspascal.yolo-bundle.v1',
      'Unsupported YOLO bundle schema');
    FProfile := ReadYoloProfile(NativeField(Root, 'profile'));
    Identity := YoloProfileIdentity(FProfile);
    NativeRequire((NativeText(Root, 'profile_identity') = Identity) and (Identity = YoloProfileIdentity(ExpectedProfile)),
      'YOLO bundle profile differs from the trusted catalog');
    Provenance := NativeObject(NativeField(Root, 'provenance'));
    NativeRequire((NativeText(Provenance, 'source_sha256') = FProfile.SourceSHA256) and
      (NativeText(Provenance, 'graph_recipe_sha256') = FProfile.RecipeSHA256),
      'YOLO bundle source or graph recipe binding differs from its profile');
    Artifact := NativeObject(NativeField(Root, 'artifact'));
    NativeRequire(
      (NativeText(Artifact, 'os') = NativeHostOS) and (NativeText(Artifact, 'arch') = 'x86_64') and (NativeText(Artifact, 'entry') = 'main') and (NativeText(Artifact, 'tvm') = '0.26.0') and (NativeText(Artifact, 'ffi_abi') = '0.1.14'),
      'YOLO artifact platform, entry or ABI does not match this application'
    );
    NativeRequire((NativeText(Artifact, 'cpu_isa') = 'avx2_fma') and NativeCpuSupportsAVX2FMA,
      'The native YOLO host requires CPU AVX2/FMA and enabled OS AVX state');
    Device := NativeText(Artifact, 'device');
    if Device = 'cpu' then
      FDevice := kDLCPU
    else if Device = 'cuda' then
    begin
      FDevice := kDLCUDA;
      Architecture := NativeText(Artifact, 'cuda_arch');
      NativeRequire((Architecture = 'sm_75') or (Architecture = 'sm_86') or (Architecture = 'sm_89') or (Architecture = 'sm_120'),
        'Unsupported YOLO CUDA artifact target');
      NativeRequire(Architecture = NativeCudaArchitecture(0),
        'YOLO CUDA artifact does not match the selected GPU compute capability');
    end
    else
      raise ENativePackage.Create('Unsupported YOLO artifact device');
    FArtifact := HoldNativeFile(BundleDirectory, NativeText(Artifact, 'file'),
      NativeText(Artifact, 'sha256'), NativeInteger(Artifact, 'bytes'));
    RequireNativeX64Library(FArtifact);
    Runtime := NativeObject(NativeField(Root, 'runtime'));
    InitializeNativeRuntime(RuntimeDirectory, NativeText(Runtime, 'manifest_sha256'),
      NativeInteger(Runtime, 'manifest_bytes'), FDevice = kDLCUDA);
    FResultByteLimit := ResultByteLimit;
    FOptions := Options;
    FRunProfile := CopyYoloProfile(FProfile);
    FRunProfile.Detections.Confidence := Options.Confidence;
    FRunProfile.Detections.NmsIoU := Options.NmsIoU;
    FRunProfile.Validate;
  finally
    Root.Free;
  end;
end;

function TYoloBundle.Profile: TYoloProfile;
begin
  Result := CopyYoloProfile(FProfile);
end;

function TYoloBundle.OpenClassifier: IClassifier;
begin
  Result := CreateYoloClassifier(FRunProfile, FArtifact.FileName, FDevice, Self, FResultByteLimit,
    FOptions.ShouldStop);
end;

function TYoloBundle.OpenDetector: IDetector;
begin
  Result := CreateYoloDetector(FRunProfile, FArtifact.FileName, FDevice, Self, FResultByteLimit,
    FOptions.ShouldStop);
end;

function TYoloBundle.OpenInstanceSegmenter: IInstanceSegmenter;
begin
  Result := CreateYoloInstanceSegmenter(FRunProfile, FArtifact.FileName, FDevice, Self,
    FResultByteLimit, FOptions.ShouldStop);
end;

function TYoloBundle.OpenPoseEstimator: IPoseEstimator;
begin
  Result := CreateYoloPoseEstimator(FRunProfile, FArtifact.FileName, FDevice, Self,
    FResultByteLimit, FOptions.ShouldStop);
end;

function TYoloBundle.OpenOrientedDetector: IOrientedDetector;
begin
  Result := CreateYoloOrientedDetector(FRunProfile, FArtifact.FileName, FDevice, Self,
    FResultByteLimit, FOptions.ShouldStop);
end;

function TYoloBundle.OpenSemanticSegmenter: ISemanticSegmenter;
begin
  Result := CreateYoloSemanticSegmenter(FRunProfile, FArtifact.FileName, FDevice, Self,
    FResultByteLimit, FOptions.ShouldStop);
end;

function TYoloBundle.OpenDepthEstimator: IDepthEstimator;
begin
  Result := CreateYoloDepthEstimator(FRunProfile, FArtifact.FileName, FDevice, Self,
    FResultByteLimit, FOptions.ShouldStop);
end;

function LoadYoloBundle(const BundleDirectory, ManifestSHA256: string; const ManifestBytes: Int64;
  const RuntimeDirectory: string; const ExpectedProfile: TYoloProfile;
  const ResultByteLimit: Int64): IYoloBundle;
begin
  Result := TYoloBundle.Create(BundleDirectory, ManifestSHA256, ManifestBytes, RuntimeDirectory,
    ExpectedProfile, TYoloRunOptions.FromProfile(ExpectedProfile), ResultByteLimit);
end;

function LoadYoloBundleForRun(const BundleDirectory, ManifestSHA256: string;
  const ManifestBytes: Int64; const RuntimeDirectory: string; const ExpectedProfile: TYoloProfile;
  const Options: TYoloRunOptions; const ResultByteLimit: Int64): IYoloBundle;
begin
  Result := TYoloBundle.Create(BundleDirectory, ManifestSHA256, ManifestBytes,
    RuntimeDirectory, ExpectedProfile, Options, ResultByteLimit);
end;

end.
