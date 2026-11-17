{ ************************************************************************** }
{                                                                            }
{ Qwen35CudaBuffers                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit Qwen35CudaBuffers;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, TVM, DLPack, TVM.CpuGraph;

type
  TQwen35DecodedPoolStats = record
    PlannedBytes: UInt64;
    AllocatedBytes: UInt64;
    Allocations: UInt64;
    Rents: UInt64;
    Groups: UInt64;
  end;

  { Each slot holds one simultaneously live layer parameter. The caller must
    synchronize CUDA and finish using every view before EndGroup. Views retain
    their storage independently of this pool, including across destruction. }
  TQwen35CudaBufferPool = class
  private
    FCapacity: TArray<UInt64>;
    FStorage: TTensorHandles;
    FBusy: TArray<Boolean>;
    FActive: Boolean;
    FStats: TQwen35DecodedPoolStats;
  public
    constructor Create(const Capacities: array of UInt64);
    destructor Destroy; override;
    procedure BeginGroup;
    function Rent(const Slot: Integer; const Shape: TArray<Int64>): TTVMFFIObjectHandle;
    procedure EndGroup;
    { Caller synchronizes CUDA; no group or borrowed view may be active. }
    procedure ReleaseStorage;
    property Stats: TQwen35DecodedPoolStats read FStats;
  end;

implementation

type
  PBufferView = ^TBufferView;
  TBufferView = record
    Managed: TDLManagedTensorVersioned;
    Parent: TTVMFFIObjectHandle;
  end;

procedure ReleaseBufferView(Managed: PDLManagedTensorVersioned); cdecl;
var
  View: PBufferView;
begin
  if Managed = nil then Exit;
  View := PBufferView(Managed);
  TTVMRuntime.ObjectRelease(View.Parent);
  FreeMem(View);
end;

constructor TQwen35CudaBufferPool.Create(const Capacities: array of UInt64);
var
  I: Integer;
begin
  inherited Create;
  RequireCpu((Length(Capacities) > 0) and (Length(Capacities) <= 64),
    'Invalid Qwen decoded buffer slot count');
  SetLength(FCapacity, Length(Capacities));
  SetLength(FStorage, Length(Capacities));
  SetLength(FBusy, Length(Capacities));
  for I := 0 to High(Capacities) do
  begin
    RequireCpu((Capacities[I] > 0) and (Capacities[I] mod SizeOf(Single) = 0) and
      (Capacities[I] <= UInt64(64) * 1024 * 1024 * 1024 - FStats.PlannedBytes),
      'Invalid Qwen decoded buffer capacity');
    FCapacity[I] := Capacities[I];
    Inc(FStats.PlannedBytes, Capacities[I]);
  end;
end;

destructor TQwen35CudaBufferPool.Destroy;
begin
  ReleaseTensors(FStorage);
  inherited;
end;

procedure TQwen35CudaBufferPool.BeginGroup;
begin
  RequireCpu(not FActive, 'Qwen decoded buffer group is already active');
  FActive := True;
  Inc(FStats.Groups);
end;

function TQwen35CudaBufferPool.Rent(const Slot: Integer;
  const Shape: TArray<Int64>): TTVMFFIObjectHandle;
var
  Elements: Int64;
  View: PBufferView;
  ShapeData: PInt64;
  Root: PDLTensor;
  I, Status: Integer;
  Error: TTVMCallError;
begin
  Result := nil;
  RequireCpu(FActive, 'Qwen decoded buffer requires an active group');
  RequireCpu((Slot >= 0) and (Slot < Length(FStorage)), 'Invalid Qwen decoded buffer slot');
  RequireCpu(not FBusy[Slot], 'Qwen decoded buffer slot is already in use');
  Elements := TensorElements(Shape);
  RequireCpu((Elements > 0) and (UInt64(Elements) <= FCapacity[Slot] div SizeOf(Single)),
    'Qwen decoded tensor exceeds its buffer capacity');
  RequireCpu(Assigned(TTVMRuntime.TModule.TVMFFITensorFromDLPackVersioned),
    'TVM DLPack tensor import is unavailable');
  if FStorage[Slot] = nil then
  begin
    RequireTVM(TTVMRuntime.AllocDeviceTensor([Int64(FCapacity[Slot])], Ord(kDLUInt), 8, 1,
      Ord(kDLCUDA), 0, FStorage[Slot], @Error), @Error);
    Inc(FStats.AllocatedBytes, FCapacity[Slot]);
    Inc(FStats.Allocations);
  end;
  Root := TTVMRuntime.TensorDLTensor(FStorage[Slot]);
  GetMem(View, SizeOf(TBufferView) + Length(Shape) * SizeOf(Int64));
  FillChar(View^, SizeOf(TBufferView), 0);
  ShapeData := PInt64(PByte(View) + SizeOf(TBufferView));
  for I := 0 to High(Shape) do ShapeData[I] := Shape[I];
  View.Parent := FStorage[Slot];
  TTVMRuntime.ObjectRetain(View.Parent);
  View.Managed.version.major := DLPACK_MAJOR_VERSION;
  View.Managed.version.minor := DLPACK_MINOR_VERSION;
  View.Managed.deleter := ReleaseBufferView;
  View.Managed.dl_tensor := Root^;
  View.Managed.dl_tensor.ndim := Length(Shape);
  View.Managed.dl_tensor.shape := ShapeData;
  View.Managed.dl_tensor.strides := nil;
  View.Managed.dl_tensor.dtype.code := Ord(kDLFloat);
  View.Managed.dl_tensor.dtype.bits := 32;
  View.Managed.dl_tensor.dtype.lanes := 1;
  Status := TTVMRuntime.TModule.TVMFFITensorFromDLPackVersioned(@View.Managed, 0, 1, Result);
  if Status <> 0 then
  begin
    ReleaseBufferView(@View.Managed);
    Result := nil;
    raise ECpuGraph.CreateFmt('Qwen decoded buffer view import failed with status %d', [Status]);
  end;
  FBusy[Slot] := True;
  Inc(FStats.Rents);
end;

procedure TQwen35CudaBufferPool.ReleaseStorage;
begin
  RequireCpu(not FActive, 'Qwen decoded buffers are still in use');
  ReleaseTensors(FStorage);
  SetLength(FStorage, Length(FCapacity));
  FStats.AllocatedBytes := 0;
end;

procedure TQwen35CudaBufferPool.EndGroup;
var
  I: Integer;
begin
  RequireCpu(FActive, 'Qwen decoded buffer group is not active');
  for I := 0 to High(FBusy) do FBusy[I] := False;
  FActive := False;
end;

end.
