{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.Identity                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.Identity;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  TVM.Yolo.Profile;

{ Versioned binary identity of the validated, normalized profile: length-prefixed
  UTF8, little-endian integers and IEEE Float32. JSON ordering, formatting and
  equivalent decimal spellings do not change the identity. }
function YoloProfileIdentity(const Profile: TYoloProfile): string;

implementation

uses
  {$IFDEF FPC}Classes, SysUtils{$ELSE}System.Classes, System.SysUtils{$ENDIF}, CrossPascal.SHA256;

function YoloProfileIdentity(const Profile: TYoloProfile): string;
var
  Stream: TMemoryStream;
  I, J: Integer;

  procedure U32(const Value: UInt32);
  var
    Bytes: array[0..3] of Byte;
    N: Integer;
  begin
    for N := 0 to 3 do Bytes[N] := (Value shr (N * 8)) and $FF;
    Stream.WriteBuffer(Bytes, SizeOf(Bytes));
  end;

  procedure U64(const Value: UInt64);
  var
    Bytes: array[0..7] of Byte;
    N: Integer;
  begin
    for N := 0 to 7 do Bytes[N] := (Value shr (N * 8)) and $FF;
    Stream.WriteBuffer(Bytes, SizeOf(Bytes));
  end;

  procedure Text(const Value: string);
  var
    Encoded: UTF8String;
  begin
    Encoded := UTF8Encode(Value); U32(Length(Encoded));
    if Length(Encoded) > 0 then Stream.WriteBuffer(Encoded[1], Length(Encoded));
  end;

  procedure Float(const Value: Single);
  var
    Bits: UInt32;
  begin
    Bits := 0;
    if Value <> 0 then Move(Value, Bits, SizeOf(Bits));
    U32(Bits);
  end;

begin
  Profile.Validate;
  Stream := TMemoryStream.Create;
  try
    Text('crosspascal.yolo-profile.identity.v1');
    Text(Profile.ID); Text(Profile.Family); Text(Profile.InputName);
    Text(Profile.SourceSHA256); Text(Profile.RecipeSHA256); Text(Profile.RecipeFile);
    U32(Ord(Profile.Task)); U32(Ord(Profile.Preprocessing));
    U32(Profile.InputWidth); U32(Profile.InputHeight);
    U32(Profile.ClassCount);
    for I := 0 to High(Profile.ClassNames) do Text(Profile.ClassNames[I]);
    for I := 0 to 2 do Float(Profile.Mean[I]);
    for I := 0 to 2 do Float(Profile.Std[I]);
    U32(Length(Profile.Outputs));
    for I := 0 to High(Profile.Outputs) do
    begin
      Text(Profile.Outputs[I].Name); U32(Ord(Profile.Outputs[I].DataType));
      U32(Length(Profile.Outputs[I].Shape));
      for J := 0 to High(Profile.Outputs[I].Shape) do U64(Profile.Outputs[I].Shape[J]);
    end;
    U32(Ord(Profile.ScaleUp)); U32(Profile.Padding);
    U32(Ord(Profile.Detections.Layout)); U32(Profile.Detections.ClassCount); U32(Profile.Detections.ExtraCount);
    Float(Profile.Detections.Confidence); Float(Profile.Detections.NmsIoU);
    U32(Profile.Detections.MaxDetections); U32(Profile.Detections.MaxNmsCandidates);
    U32(Ord(Profile.Detections.ApplyObjectnessToExtra)); U32(Ord(Profile.NmsKind));
    Text(Profile.BoxEncoding); U32(Ord(Profile.CoordinateMode)); U32(Ord(Profile.ClassScoreKind));
    U32(Profile.TopK); U32(Ord(Profile.InstanceMaskKind)); U32(Profile.PrototypeStride);
    U32(Profile.KeypointCount); U32(Profile.KeypointDimensions); Float(Profile.ClassOffset);
    Text(Profile.DepthUnits); U32(Ord(Profile.DepthAlignCorners));
    Stream.Position := 0;
    Result := SHA256Stream(Stream);
  finally
    Stream.Free;
  end;
end;

end.
