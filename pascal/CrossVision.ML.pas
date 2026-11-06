{ ************************************************************************** }
{                                                                            }
{ CrossVision.ML                                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossVision.ML;

interface

uses
  CrossVision.Geometry.Types,
  CrossVision.Mat;

type
  TDetection = record
    Box: TRectI;
    Score: Double;
    ClassId: Integer;
    ClassName: string;
  end;
  TDetectionArray = TArray<TDetection>;

  TInstance = record
    Box: TRectI;
    Score: Double;
    ClassId: Integer;
    ClassName: string;
    Mask: IMat;
  end;
  TInstanceArray = TArray<TInstance>;

  TClassification = record
    ClassId: Integer;
    ClassName: string;
    Score: Double;
  end;
  TClassificationArray = TArray<TClassification>;

  TOcrLine = record
    Text: string;
    Box: TRectI;
    Score: Double;
  end;
  TOcrResult = TArray<TOcrLine>;

  IDetector = interface
    ['{D1E2C3A4-0001-4B5C-9D6E-7F8091A2B3C4}']
    function Detect(const Image: IMat): TDetectionArray;
  end;

  ISegmenter = interface
    ['{D1E2C3A4-0002-4B5C-9D6E-7F8091A2B3C4}']
    function Segment(const Image: IMat): IMat;
  end;

  IInstanceSegmenter = interface
    ['{D1E2C3A4-0005-4B5C-9D6E-7F8091A2B3C4}']
    function SegmentInstances(const Image: IMat): TInstanceArray;
  end;

  TSemanticMask = record
    ClassId: Integer;
    ClassName: string;
    Mask: IMat;
  end;
  TSemanticMaskArray = TArray<TSemanticMask>;

  ISemanticSegmenter = interface
    ['{D1E2C3A4-0006-4B5C-9D6E-7F8091A2B3C4}']
    function SegmentClasses(const Image: IMat): TSemanticMaskArray;
  end;

  IClassifier = interface
    ['{D1E2C3A4-0003-4B5C-9D6E-7F8091A2B3C4}']
    function Classify(const Image: IMat): TClassificationArray;
  end;

  TOrientedBox = record
    CenterX: Single;
    CenterY: Single;
    Width: Single;
    Height: Single;
    Angle: Single;
  end;

  TOrientedDetection = record
    Box: TOrientedBox;
    Score: Double;
    ClassId: Integer;
    ClassName: string;
  end;
  TOrientedDetectionArray = TArray<TOrientedDetection>;

  IOrientedDetector = interface
    ['{D1E2C3A4-0007-4B5C-9D6E-7F8091A2B3C4}']
    function DetectOriented(const Image: IMat): TOrientedDetectionArray;
  end;

  TKeypoint = record
    X: Single;
    Y: Single;
    Visibility: Single;
  end;
  TKeypointArray = TArray<TKeypoint>;

  TPose = record
    Box: TRectI;
    Score: Double;
    ClassId: Integer;
    ClassName: string;
    Keypoints: TKeypointArray;
  end;
  TPoseArray = TArray<TPose>;

  IPoseEstimator = interface
    ['{D1E2C3A4-0008-4B5C-9D6E-7F8091A2B3C4}']
    function EstimatePoses(const Image: IMat): TPoseArray;
  end;

  IOcr = interface
    ['{D1E2C3A4-0004-4B5C-9D6E-7F8091A2B3C4}']
    function Recognize(const Image: IMat): TOcrResult;
  end;

  TStopReason = (srEos, srStopSequence, srLength, srCancelled, srError);

  TGeneratedToken = record
    Id: Integer;
    Text: string;
  end;

  TGenerationOptions = record
    MaxNewTokens: Integer;
    Temperature: Single;
    TopK: Integer;
    TopP: Single;
    Seed: Int64;
    StopSequences: TArray<string>;
    class function Default: TGenerationOptions; static;
  end;

  TGenerationResult = record
    Text: string;
    StopReason: TStopReason;
    PromptTokens: Integer;
    GeneratedTokens: Integer;
  end;

  ITokenSink = interface
    ['{CEC4EC0F-F1EE-448A-97A5-81082771F7EE}']
    function OnToken(const Token: TGeneratedToken): Boolean;
  end;

  ITextGenerationSession = interface
    ['{3F815515-65A6-4D81-A686-27F290106E0A}']
    function Generate(const Prompt: string): string; overload;
    function Generate(const Prompt: string; const Sink: ITokenSink): TGenerationResult; overload;
    procedure Reset;
  end;

  ITextGenerator = interface
    ['{B58627A8-D61E-48F6-B8F0-73354F972530}']
    function CreateSession(const Options: TGenerationOptions): ITextGenerationSession;
  end;

function Detection(const Box: TRectI; const Score: Double; const ClassId: Integer;
  const ClassName: string): TDetection;
function Instance(const Box: TRectI; const Score: Double; const ClassId: Integer;
  const ClassName: string; const Mask: IMat): TInstance;
function Classification(const ClassId: Integer; const ClassName: string;
  const Score: Double): TClassification;
function OcrLine(const Text: string; const Box: TRectI; const Score: Double): TOcrLine;
function SemanticMask(const ClassId: Integer; const ClassName: string;
  const Mask: IMat): TSemanticMask;

implementation

function Detection(const Box: TRectI; const Score: Double; const ClassId: Integer;
  const ClassName: string): TDetection;
begin
  Result.Box := Box;
  Result.Score := Score;
  Result.ClassId := ClassId;
  Result.ClassName := ClassName;
end;

function SemanticMask(const ClassId: Integer; const ClassName: string;
  const Mask: IMat): TSemanticMask;
begin
  Result.ClassId := ClassId;
  Result.ClassName := ClassName;
  Result.Mask := Mask;
end;

function Instance(const Box: TRectI; const Score: Double; const ClassId: Integer;
  const ClassName: string; const Mask: IMat): TInstance;
begin
  Result.Box := Box;
  Result.Score := Score;
  Result.ClassId := ClassId;
  Result.ClassName := ClassName;
  Result.Mask := Mask;
end;

function Classification(const ClassId: Integer; const ClassName: string;
  const Score: Double): TClassification;
begin
  Result.ClassId := ClassId;
  Result.ClassName := ClassName;
  Result.Score := Score;
end;

function OcrLine(const Text: string; const Box: TRectI; const Score: Double): TOcrLine;
begin
  Result.Text := Text;
  Result.Box := Box;
  Result.Score := Score;
end;

class function TGenerationOptions.Default: TGenerationOptions;
begin
  Result.MaxNewTokens := 256;
  Result.Temperature := 0;
  Result.TopK := 0;
  Result.TopP := 0;
  Result.Seed := 0;
  Result.StopSequences := nil;
end;

end.
