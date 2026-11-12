{ ************************************************************************** }
{                                                                            }
{ TVM.CudaExecution                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.CudaExecution;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  {$IFDEF FPC}SysUtils, Windows{$ELSE}System.SysUtils, Winapi.Windows{$ENDIF},
  CrossPascal.Json, TVM, TVM.CpuGraph;

type
  TCudaExecution = class
  private type
    TInit = function(Flags: Cardinal): Integer; stdcall;
    TDeviceGet = function(out Device: Integer; Ordinal: Integer): Integer; stdcall;
    TDeviceAttribute = function(out Value: Integer; Attribute, Device: Integer): Integer; stdcall;
    TDeviceName = function(Name: PAnsiChar; Count, Device: Integer): Integer; stdcall;
    TRetain = function(out Context: Pointer; Device: Integer): Integer; stdcall;
    TRelease = function(Device: Integer): Integer; stdcall;
    TPush = function(Context: Pointer): Integer; stdcall;
    TPop = function(out Context: Pointer): Integer; stdcall;
    TSync = function: Integer; stdcall;
    TMemoryInfo = function(out FreeBytes, TotalBytes: NativeUInt): Integer; stdcall;
    TAlloc = function(out Address: UInt64; Bytes: NativeUInt): Integer; stdcall;
    TFree = function(Address: UInt64): Integer; stdcall;
  private
    FDriver: HMODULE;
    FDevice: Integer;
    FContext: Pointer;
    FRelease: TRelease;
    FPush: TPush;
    FPop: TPop;
    FSync: TSync;
    FMemoryInfo: TMemoryInfo;
    FAlloc: TAlloc;
    FFree: TFree;
    function Entry(const Name: AnsiString): Pointer;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Enter;
    procedure Leave;
    procedure Synchronize;
    function FreeBytes: UInt64;
    procedure MemoryBytes(out Available, Total: UInt64);
    function ComputeVersion: string;
    function Architecture: string;
    procedure RequireArchitecture(const Target: string);
    function DeviceName: string;
  end;

  TCudaWorkspace = class
  private type
    TAlloc = function(Kind, Device: Integer; Bytes: UInt64; Code, Bits: Integer): Pointer; cdecl;
    TRelease = function(Kind, Device: Integer; Address: Pointer): Integer; cdecl;
    TBuffer = record
      Address: UInt64;
      Bytes: UInt64;
      InUse: Boolean;
    end;
  private
    FCuda: TCudaExecution;
    FLibrary: HMODULE;
    FAllocSlot: PPointer;
    FFreeSlot: PPointer;
    FSavedAlloc: Pointer;
    FSavedFree: Pointer;
    FOriginalAlloc: TAlloc;
    FOriginalFree: TRelease;
    FBuffers: TArray<TBuffer>;
    FReserve: UInt64;
    FAllocated: UInt64;
    FError: string;
    function Allocate(const Kind, Device: Integer; const Bytes: UInt64; const Code, Bits: Integer): Pointer;
    function Release(const Kind, Device: Integer; const Address: Pointer): Integer;
  public
    constructor Create(const Cuda: TCudaExecution; const LibraryName: string; const Reserve: UInt64);
    destructor Destroy; override;
    procedure BeginRun;
    procedure EndRun;
    procedure CheckError;
    function Run(const Graph: TCpuGraph; const Args: array of TTVMFFIObjectHandle;
      const Stop: TCpuGraphStop = nil): TTVMFFIAny;
  end;

function ReadLanguageManifest(const Folder: string): TJSONValue;

implementation

var
  WorkspaceLock: TRTLCriticalSection;

threadvar
  ActiveWorkspace: TCudaWorkspace;

function ReadLanguageManifest(const Folder: string): TJSONValue;
var
  Index, Selected: TJSONValue;
  Root, Variants, Model: TJSONObject;
  Cuda: TCudaExecution;
  Target, Member: string;
begin
  Index := ReadModelJSON(BundleMember(Folder, 'model.json'));
  try
    Root := ModelObject(Index, 'language model manifest');
    if Root.GetValue('cuda_variants') = nil then
    begin
      Result := Index;
      Index := nil;
      Exit;
    end;
    RequireCpu(ModelText(Root.GetValue('backend')) = 'TVM CUDA W8A32', 'Invalid CUDA variant index');
    Variants := ModelObject(Root.GetValue('cuda_variants'), 'CUDA model variants');
    Cuda := TCudaExecution.Create;
    try
      Target := Cuda.Architecture;
    finally
      Cuda.Free;
    end;
    Member := ModelText(Variants.GetValue(Target));
    RequireCpu(Member <> '', 'This bundle has no ' + Target + ' variant. Choose a CPU bundle.');
    Selected := ReadModelJSON(BundleMember(Folder, Member));
    try
      Model := ModelObject(Selected, 'CUDA language model variant');
      RequireCpu((ModelText(Model.GetValue('cuda_arch')) = Target) and
        (ModelText(Model.GetValue('schema')) = ModelText(Root.GetValue('schema'))),
        'The selected CUDA model does not match the variant index');
      Result := Selected;
      Selected := nil;
    finally
      Selected.Free;
    end;
  finally
    Index.Free;
  end;
end;

procedure CheckCuda(const Status: Integer; const Operation: string);
begin
  if Status <> 0 then
    raise Exception.CreateFmt('%s failed with CUDA status %d', [Operation, Status]);
end;

function TCudaExecution.Entry(const Name: AnsiString): Pointer;
begin
  Result := GetProcAddress(FDriver, PAnsiChar(Name));
  if Result = nil then raise Exception.Create('Missing CUDA driver entry: ' + string(Name));
end;

constructor TCudaExecution.Create;
var
  Init: TInit;
  GetDevice: TDeviceGet;
  Retain: TRetain;
begin
  inherited Create;
  FDriver := LoadLibraryExW('nvcuda.dll', 0, $800);
  if FDriver = 0 then RaiseLastOSError;
  Init := TInit(Entry('cuInit'));
  GetDevice := TDeviceGet(Entry('cuDeviceGet'));
  Retain := TRetain(Entry('cuDevicePrimaryCtxRetain'));
  FRelease := TRelease(Entry('cuDevicePrimaryCtxRelease_v2'));
  FPush := TPush(Entry('cuCtxPushCurrent_v2'));
  FPop := TPop(Entry('cuCtxPopCurrent_v2'));
  FSync := TSync(Entry('cuCtxSynchronize'));
  FMemoryInfo := TMemoryInfo(Entry('cuMemGetInfo_v2'));
  FAlloc := TAlloc(Entry('cuMemAlloc_v2'));
  FFree := TFree(Entry('cuMemFree_v2'));
  CheckCuda(Init(0), 'Initialize CUDA');
  CheckCuda(GetDevice(FDevice, 0), 'Select CUDA device');
  CheckCuda(Retain(FContext, FDevice), 'Retain the CUDA primary context');
end;

destructor TCudaExecution.Destroy;
begin
  if FContext <> nil then FRelease(FDevice);
  if FDriver <> 0 then FreeLibrary(FDriver);
  inherited;
end;

procedure TCudaExecution.Enter;
begin
  CheckCuda(FPush(FContext), 'Enter the CUDA primary context');
end;

procedure TCudaExecution.Leave;
var
  Context: Pointer;
begin
  CheckCuda(FPop(Context), 'Restore the previous CUDA context');
  if Context <> FContext then raise Exception.Create('CUDA context changed during execution');
end;

procedure TCudaExecution.Synchronize;
begin
  CheckCuda(FSync(), 'Synchronize CUDA');
end;

function TCudaExecution.FreeBytes: UInt64;
var
  Total: UInt64;
begin
  MemoryBytes(Result, Total);
end;

procedure TCudaExecution.MemoryBytes(out Available, Total: UInt64);
var
  FreeMemory, TotalMemory: NativeUInt;
begin
  CheckCuda(FMemoryInfo(FreeMemory, TotalMemory), 'Read CUDA memory');
  Available := FreeMemory;
  Total := TotalMemory;
end;

function TCudaExecution.DeviceName: string;
var
  Query: TDeviceName;
  Buffer: array[0..255] of AnsiChar;
begin
  Query := TDeviceName(Entry('cuDeviceGetName'));
  FillChar(Buffer, SizeOf(Buffer), 0);
  CheckCuda(Query(@Buffer[0], SizeOf(Buffer), FDevice), 'Read CUDA device name');
  Result := UTF8Decode(UTF8String(PAnsiChar(@Buffer[0])));
end;

function TCudaExecution.ComputeVersion: string;
const
  ComputeMajor = 75;
  ComputeMinor = 76;
var
  Query: TDeviceAttribute;
  Major, Minor: Integer;
begin
  Query := TDeviceAttribute(Entry('cuDeviceGetAttribute'));
  CheckCuda(Query(Major, ComputeMajor, FDevice), 'Read CUDA compute capability');
  CheckCuda(Query(Minor, ComputeMinor, FDevice), 'Read CUDA compute capability');
  Result := IntToStr(Major) + '.' + IntToStr(Minor);
end;

function TCudaExecution.Architecture: string;
begin
  Result := 'sm_' + StringReplace(ComputeVersion, '.', '', []);
end;

procedure TCudaExecution.RequireArchitecture(const Target: string);
var
  Actual: string;
begin
  Actual := Architecture;
  RequireCpu(Target = Actual, 'The model bundle targets ' + Target + ', but ' +
    DeviceName + ' requires ' + Actual + '.');
end;

function WorkspaceAllocate(Kind, Device: Integer; Bytes: UInt64; Code, Bits: Integer): Pointer; cdecl;
begin
  Result := nil;
  if ActiveWorkspace = nil then Exit;
  try
    Result := ActiveWorkspace.Allocate(Kind, Device, Bytes, Code, Bits);
  except
    on E: Exception do ActiveWorkspace.FError := E.Message;
  end;
end;

function WorkspaceRelease(Kind, Device: Integer; Address: Pointer): Integer; cdecl;
begin
  Result := -1;
  if ActiveWorkspace = nil then Exit;
  try
    Result := ActiveWorkspace.Release(Kind, Device, Address);
  except
    on E: Exception do ActiveWorkspace.FError := E.Message;
  end;
end;

constructor TCudaWorkspace.Create(const Cuda: TCudaExecution; const LibraryName: string;
  const Reserve: UInt64);
begin
  inherited Create;
  FCuda := Cuda;
  FReserve := Reserve;
  FLibrary := LoadLibraryExW(PWideChar(LibraryName), 0, $1100);
  if FLibrary = 0 then RaiseLastOSError;
  FAllocSlot := PPointer(GetProcAddress(FLibrary, '__TVMBackendAllocWorkspace'));
  FFreeSlot := PPointer(GetProcAddress(FLibrary, '__TVMBackendFreeWorkspace'));
  if (FAllocSlot = nil) <> (FFreeSlot = nil) then
    raise Exception.Create('Incomplete TVM workspace exports');
end;

destructor TCudaWorkspace.Destroy;
var
  I: Integer;
begin
  for I := 0 to High(FBuffers) do FCuda.FFree(FBuffers[I].Address);
  if FLibrary <> 0 then FreeLibrary(FLibrary);
  inherited;
end;

function TCudaWorkspace.Allocate(const Kind, Device: Integer; const Bytes: UInt64;
  const Code, Bits: Integer): Pointer;
var
  I, Best: Integer;
  Count, Address: UInt64;
begin
  if Kind <> 2 then Exit(FOriginalAlloc(Kind, Device, Bytes, Code, Bits));
  if Device <> 0 then raise Exception.Create('Unexpected CUDA workspace device');
  Best := -1;
  for I := 0 to High(FBuffers) do
    if not FBuffers[I].InUse and (FBuffers[I].Bytes >= Bytes) and
      ((Best < 0) or (FBuffers[I].Bytes < FBuffers[Best].Bytes)) then Best := I;
  if Best < 0 then
  begin
    Count := Bytes;
    if Count < 256 then Count := 256;
    if Count > FReserve - FAllocated then
      raise Exception.Create('CUDA workspace exceeds its reserve');
    Best := Length(FBuffers);
    SetLength(FBuffers, Best + 1);
    try
      CheckCuda(FCuda.FAlloc(Address, Count), 'Allocate CUDA workspace');
    except
      SetLength(FBuffers, Best);
      raise;
    end;
    FBuffers[Best].Address := Address;
    FBuffers[Best].Bytes := Count;
    Inc(FAllocated, Count);
  end;
  FBuffers[Best].InUse := True;
  Result := Pointer(NativeUInt(FBuffers[Best].Address));
end;

function TCudaWorkspace.Release(const Kind, Device: Integer; const Address: Pointer): Integer;
var
  I: Integer;
begin
  if Kind <> 2 then Exit(FOriginalFree(Kind, Device, Address));
  if Device = 0 then
    for I := 0 to High(FBuffers) do
      if FBuffers[I].InUse and (FBuffers[I].Address = NativeUInt(Address)) then
      begin
        FBuffers[I].InUse := False;
        Exit(0);
      end;
  raise Exception.Create('Invalid CUDA workspace release');
end;

procedure TCudaWorkspace.BeginRun;
begin
  FError := '';
  if FAllocSlot = nil then Exit;
  EnterCriticalSection(WorkspaceLock);
  try
    if ActiveWorkspace <> nil then raise Exception.Create('Nested CUDA workspace use');
    FSavedAlloc := FAllocSlot^;
    FSavedFree := FFreeSlot^;
    FOriginalAlloc := TAlloc(FSavedAlloc);
    FOriginalFree := TRelease(FSavedFree);
    if not Assigned(FOriginalAlloc) or not Assigned(FOriginalFree) then
      raise Exception.Create('TVM has not initialized the workspace exports');
    ActiveWorkspace := Self;
    FAllocSlot^ := @WorkspaceAllocate;
    FFreeSlot^ := @WorkspaceRelease;
  except
    LeaveCriticalSection(WorkspaceLock);
    raise;
  end;
end;

procedure TCudaWorkspace.EndRun;
begin
  if FAllocSlot = nil then Exit;
  FAllocSlot^ := FSavedAlloc;
  FFreeSlot^ := FSavedFree;
  ActiveWorkspace := nil;
  LeaveCriticalSection(WorkspaceLock);
end;

procedure TCudaWorkspace.CheckError;
begin
  if FError <> '' then raise Exception.Create(FError);
end;

function TCudaWorkspace.Run(const Graph: TCpuGraph; const Args: array of TTVMFFIObjectHandle;
  const Stop: TCpuGraphStop): TTVMFFIAny;
begin
  BeginRun;
  try
    try
      Result := Graph.Run(Args, Stop);
    except
      CheckError;
      raise;
    end;
    if FError <> '' then
    begin
      if Result.TypeIndex >= kTVMFFIStaticObjectBegin then TTVMRuntime.ObjectRelease(Result.VObj);
      CheckError;
    end;
  finally
    EndRun;
  end;
end;

initialization
  InitializeCriticalSection(WorkspaceLock);

finalization
  DeleteCriticalSection(WorkspaceLock);

end.
