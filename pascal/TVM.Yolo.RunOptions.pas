{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.RunOptions                                                        }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.RunOptions;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  TVM.CpuGraph, TVM.Yolo.Profile;

type
  { These are inference settings, not part of the authenticated graph identity.
    NMS-free heads keep their graph selection and do not acquire a second NMS.
    ShouldStop runs on the inference thread and must be thread safe; its owner
    must outlive the bundle and every model opened from it. }
  TYoloRunOptions = record
    Confidence: Single;
    NmsIoU: Single;
    ShouldStop: TCpuGraphStop;
    class function FromProfile(const Profile: TYoloProfile): TYoloRunOptions; static;
    procedure Validate;
  end;

procedure CheckYoloRunStop(const ShouldStop: TCpuGraphStop);

implementation

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF};

class function TYoloRunOptions.FromProfile(const Profile: TYoloProfile): TYoloRunOptions;
begin
  Result := Default(TYoloRunOptions);
  Result.Confidence := Profile.Detections.Confidence;
  Result.NmsIoU := Profile.Detections.NmsIoU;
end;

procedure TYoloRunOptions.Validate;
begin
  if IsNan(Confidence) or IsInfinite(Confidence) or
    (Confidence < 0) or (Confidence > 1) or IsNan(NmsIoU) or
    IsInfinite(NmsIoU) or (NmsIoU < 0) or (NmsIoU > 1) then
      raise EArgumentException.Create('YOLO confidence and NMS thresholds must be finite values from zero to one');
end;

procedure CheckYoloRunStop(const ShouldStop: TCpuGraphStop);
begin
  if Assigned(ShouldStop) and ShouldStop() then
    raise ECpuGraphStopped.Create('YOLO inference stopped');
end;

end.
