{ ************************************************************************** }
{                                                                            }
{ TVM.Yolo.OutputGeometry                                                    }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.Yolo.OutputGeometry;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  TVM.Yolo.Detections, TVM.Yolo.Oriented, TVM.Yolo.Transform;

{ Returns independent arrays in original-image coordinates. For poses the
  profile supplies K and D (2 or 3); visibility is preserved when D=3. }
function ProjectYoloDetections(const Detections: TYoloRawDetections;
  const Transform: TYoloImageTransform; const KeypointCount: Integer = 0;
  const KeypointDimensions: Integer = 0): TYoloRawDetections;
function ProjectYoloOrientedDetections(const Detections: TYoloOrientedDetections;
  const Transform: TYoloImageTransform): TYoloOrientedDetections;

implementation

uses
  {$IFDEF FPC}SysUtils, Math, Types{$ELSE}System.SysUtils, System.Math, System.Types{$ENDIF};

procedure ValidateTransform(const Transform: TYoloImageTransform);
begin
  if (Transform.GainX <= 0) or (Transform.GainY <= 0) then
    raise EArgumentException.Create('Uninitialized YOLO output transform');
end;

function ProjectYoloDetections(const Detections: TYoloRawDetections; const Transform: TYoloImageTransform;
  const KeypointCount, KeypointDimensions: Integer): TYoloRawDetections;
var
  I, K, At: Integer;
  Point, Projected: TPointF;
begin
  ValidateTransform(Transform);
  if (KeypointCount < 0) or ((KeypointCount = 0) and (KeypointDimensions <> 0)) or
    ((KeypointCount > 0) and not (KeypointDimensions in [2, 3])) then
      raise EArgumentException.Create('Invalid YOLO keypoint shape');
  Result := Copy(Detections);
  for I := 0 to High(Result) do
  begin
    Result[I].Box := Transform.ImageBox(Detections[I].Box);
    Result[I].Extra := Copy(Detections[I].Extra);
    if KeypointCount = 0 then Continue;
    if Int64(KeypointCount) * KeypointDimensions <> Length(Result[I].Extra) then
      raise EArgumentException.Create('YOLO keypoint count differs from its profile');
    for K := 0 to KeypointCount - 1 do
    begin
      At := K * KeypointDimensions;
      Point.X := Detections[I].Extra[At];
      Point.Y := Detections[I].Extra[At + 1];
      Projected := Transform.ImagePoint(Point);
      Result[I].Extra[At] := Projected.X;
      Result[I].Extra[At + 1] := Projected.Y;
    end;
  end;
end;

function ProjectExtent(const Value: Single; const Gain: Double): Single;
var
  Scaled: Double;
begin
  Scaled := Value / Gain;
  if IsNan(Scaled) or IsInfinite(Scaled) or (Abs(Scaled) > MaxSingle) then
    raise EArgumentException.Create('YOLO oriented extent exceeds Single');
  Result := Scaled;
end;

function ProjectYoloOrientedDetections(const Detections: TYoloOrientedDetections;
  const Transform: TYoloImageTransform): TYoloOrientedDetections;
var
  I: Integer;
  Point, Projected: TPointF;
begin
  ValidateTransform(Transform);
  Result := Copy(Detections);
  for I := 0 to High(Result) do
  begin
    Point.X := Detections[I].Box.CenterX;
    Point.Y := Detections[I].Box.CenterY;
    Projected := Transform.ImagePoint(Point, False);
    Result[I].Box.CenterX := Projected.X;
    Result[I].Box.CenterY := Projected.Y;
    Result[I].Box.Width := ProjectExtent(Detections[I].Box.Width, Transform.GainX);
    Result[I].Box.Height := ProjectExtent(Detections[I].Box.Height, Transform.GainY);
  end;
end;

end.
