{ ************************************************************************** }
{                                                                            }
{ CrossVision.ML.Depth                                                       }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossVision.ML.Depth;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossVision.Mat;

type
  TDepthUnits = (duMeters);
  TDepthMap = record
    { Original-size CV_32FC1, estimated depth along the camera viewing axis. }
    Values: IMat;
    Units: TDepthUnits;
  end;

  IDepthEstimator = interface
    ['{D8C2D924-34F0-40C0-9931-382D67EF3860}']
    function EstimateDepth(const Image: IMat): TDepthMap;
  end;

implementation

end.
