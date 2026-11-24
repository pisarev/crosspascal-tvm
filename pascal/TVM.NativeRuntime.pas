{ ************************************************************************** }
{                                                                            }
{ TVM.NativeRuntime                                                          }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit TVM.NativeRuntime;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

{ The caller pins runtime.json through a trusted bundle/catalog. One runtime
  is selected per process and remains resident until application finalization. This
  initializes only the runtime; the compiler library is neither required nor
  loaded. Applications using this API compile TVM with TVM_LAZY_INIT. }
procedure InitializeNativeRuntime(const Directory, ManifestSHA256: string;
  const ManifestBytes: Int64; const NeedCuda: Boolean);
function NativeRuntimeDirectory: string;

implementation

uses
  {$IFDEF FPC}SysUtils, {$IFDEF MSWINDOWS}Windows,{$ENDIF}Dynlibs, SyncObjs
  {$ELSE}System.SysUtils, Winapi.Windows, System.SyncObjs{$ENDIF},
  CrossPascal.Json, TVM, TVM.NativeFiles;

type
  TNativeLibraryHandle = {$IFDEF FPC}TLibHandle{$ELSE}HMODULE{$ENDIF};
  TRuntimeFile = record
    Role: string;
    Lease: INativeFileLease;
  end;

var
  RuntimeLock: TCriticalSection;
  SelectedDirectory, SelectedManifestSHA256: string;
  SelectedManifestBytes: Int64;
  RuntimeFailed: Boolean = False;
  HeldFiles: TArray<INativeFileLease>;
  DependencyHandles: TArray<TNativeLibraryHandle>;

function LoadNativeLibrary(const FileName: string): TNativeLibraryHandle;
begin
  {$IFDEF MSWINDOWS}
  { Search dependencies only in this library's directory and System32. }
  Result := {$IFDEF FPC}Windows.{$ENDIF}LoadLibraryExW(PWideChar(FileName), 0, $900);
  {$ELSE}
  Result := LoadLibrary(UTF8Encode(FileName));
  {$ENDIF}
  NativeRequire(Result <> 0, 'Cannot load native runtime library: ' + ExtractFileName(FileName));
end;

procedure CloseLibrary(const Handle: TNativeLibraryHandle);
begin
  if Handle <> 0 then {$IFDEF FPC}UnloadLibrary(Handle){$ELSE}FreeLibrary(Handle){$ENDIF};
end;

function NativeRuntimeDirectory: string;
begin
  RuntimeLock.Enter;
  try
    Result := SelectedDirectory;
  finally
    RuntimeLock.Leave;
  end;
end;

procedure InitializeNativeRuntime(const Directory, ManifestSHA256: string;
  const ManifestBytes: Int64; const NeedCuda: Boolean);
var
  Root, Row: TJSONObject;
  Rows: TJSONArray;
  Manifest: INativeFileLease;
  Files: TArray<TRuntimeFile>;
  Leases: TArray<INativeFileLease>;
  Handles: TArray<TNativeLibraryHandle>;
  Version: TTVMFFIVersion;
  GetVersion: TTVMRuntime.TModule.TTVMFFIGetVersion;
  FullDirectory, Role, ExpectedName: string;
  I, J, FFIIndex, RuntimeIndex, ExtraIndex, CudaIndex: Integer;
  GlobalsAssigned: Boolean;

  function RoleIndex(const Value: string): Integer;
  var
    N: Integer;
  begin
    Result := -1;
    for N := 0 to High(Files) do
      if Files[N].Role = Value then
      begin
        NativeRequire(Result < 0, 'Duplicate native runtime library role: ' + Value);
        Result := N;
      end;
  end;

  procedure LoadIndex(const Index: Integer);
  begin
    Handles[Index] := LoadNativeLibrary(Files[Index].Lease.FileName);
  end;

  procedure LoadCuda;
  var
    N: Integer;
  begin
    NativeRequire(CudaIndex >= 0, 'The selected runtime kit has no CUDA library');
    for N := 0 to High(Files) do if Files[N].Role = 'dependency_cuda' then LoadIndex(N);
    LoadIndex(CudaIndex);
  end;

  procedure RetainDependencies;
  var
    N, At: Integer;
  begin
    At := Length(DependencyHandles);
    for N := 0 to High(Files) do
      if (Files[N].Role = 'dependency_cuda') and (Handles[N] <> 0) then
      begin
        SetLength(DependencyHandles, At + 1);
        DependencyHandles[At] := Handles[N];
        Handles[N] := 0;
        Inc(At);
      end;
  end;

begin
  RuntimeLock.Enter;
  try
    NativeRequire(not RuntimeFailed, 'Native runtime initialization failed; restart the application');
    FullDirectory := ExcludeTrailingPathDelimiter(ExpandFileName(Directory));
    Manifest := HoldNativeFile(FullDirectory, 'runtime.json', ManifestSHA256, ManifestBytes);
    Root := ReadNativeObject(Manifest);
    try
      NativeRequire(NativeText(Root, 'schema') = 'crosspascal.native-runtime.v1',
        'Unsupported native runtime schema');
      NativeRequire((NativeText(Root, 'os') = NativeHostOS) and (NativeText(Root, 'arch') = 'x86_64'),
        'Native runtime OS or architecture does not match this application');
      NativeRequire((NativeText(Root, 'tvm') = '0.26.0') and (NativeText(Root, 'ffi_abi') = '0.1.14'),
        'Unsupported native runtime or FFI ABI');
      Rows := NativeArray(NativeField(Root, 'files'));
      NativeRequire((Rows.Count >= 3) and (Rows.Count <= 32), 'Invalid native runtime file count');
      SetLength(Files, Rows.Count); SetLength(Handles, Rows.Count); SetLength(Leases, Rows.Count + 1);
      Leases[0] := Manifest;
      for I := 0 to Rows.Count - 1 do
      begin
        Row := NativeObject(Rows.Items[I]);
        Role := NativeText(Row, 'role');
        NativeRequire((Role = 'ffi') or (Role = 'runtime') or (Role = 'extra') or
          (Role = 'cuda') or (Role = 'dependency_cuda'), 'Unknown native runtime file role');
        ExpectedName := NativeText(Row, 'file');
        if Role = 'ffi' then
          NativeRequire(ExpectedName = TTVMRuntime.TModule.FFIName, 'Unexpected native FFI filename');
        if Role = 'runtime' then
          NativeRequire(ExpectedName = TTVMRuntime.TModule.RuntimeName, 'Unexpected native runtime filename');
        if Role = 'extra' then
          NativeRequire(ExpectedName = TTVMRuntime.TModule.ExtraRuntimeName,
            'Unexpected extra runtime filename');
        if Role = 'cuda' then
          NativeRequire(ExpectedName = TTVMRuntime.TModule.CudaRuntimeName,
            'Unexpected CUDA runtime filename');
        Files[I].Role := Role;
        Files[I].Lease := HoldNativeFile(FullDirectory, ExpectedName, NativeText(Row, 'sha256'),
          NativeInteger(Row, 'bytes'));
        RequireNativeX64Library(Files[I].Lease);
        Leases[I + 1] := Files[I].Lease;
        for J := 0 to I - 1 do
          NativeRequire(not SameFileName(Files[I].Lease.FileName, Files[J].Lease.FileName),
            'Duplicate native runtime filename');
      end;
      FFIIndex := RoleIndex('ffi');
      RuntimeIndex := RoleIndex('runtime');
      ExtraIndex := RoleIndex('extra');
      CudaIndex := RoleIndex('cuda');
      NativeRequire((FFIIndex >= 0) and (RuntimeIndex >= 0) and (ExtraIndex >= 0),
        'The native runtime kit is incomplete');
      if SelectedDirectory <> '' then
      begin
        NativeRequire((TTVMRuntime.TModule.Handle <> 0) and (TTVMRuntime.TModule.RuntimeHandle <> 0),
          'The selected native runtime was finalized outside its process lifetime');
        NativeRequire(SameFileName(FullDirectory, SelectedDirectory) and
          (ManifestSHA256 = SelectedManifestSHA256) and (ManifestBytes = SelectedManifestBytes),
          'Restart before selecting another native runtime kit');
        if NeedCuda and (TTVMRuntime.TModule.CudaRuntimeHandle = 0) then
          try
            LoadCuda;
            TTVMRuntime.TModule.CudaRuntimeHandle := Handles[CudaIndex];
            Handles[CudaIndex] := 0; RetainDependencies;
          finally
            for I := High(Handles) downto 0 do CloseLibrary(Handles[I]);
          end;
        Exit;
      end;
      NativeRequire((TTVMRuntime.TModule.Handle = 0) and (TTVMRuntime.TModule.RuntimeHandle = 0),
        'Select the native runtime before initializing TVM through another API');
      GlobalsAssigned := False;
      try
        LoadIndex(FFIIndex);
        GetVersion := TTVMRuntime.TModule.TTVMFFIGetVersion(GetProcAddress(Handles[FFIIndex],
          'TVMFFIGetVersion'));
        NativeRequire(Assigned(GetVersion), 'Native FFI version query is absent');
        GetVersion(Version);
        NativeRequire((Version.Major = 0) and (Version.Minor = 1) and (Version.Patch = 14),
          'The native library FFI ABI differs from the selected kit');
        LoadIndex(RuntimeIndex); LoadIndex(ExtraIndex);
        if NeedCuda then LoadCuda;
        TTVMRuntime.TModule.Handle := Handles[FFIIndex];
        TTVMRuntime.TModule.RuntimeHandle := Handles[RuntimeIndex];
        TTVMRuntime.TModule.ExtraRuntimeHandle := Handles[ExtraIndex];
        if NeedCuda then TTVMRuntime.TModule.CudaRuntimeHandle := Handles[CudaIndex];
        TTVMRuntime.TModule.LibraryPath := IncludeTrailingPathDelimiter(FullDirectory);
        TTVMRuntime.TModule.LibraryName := Files[RuntimeIndex].Lease.FileName;
        TTVMBackend.TModule.Handle := Handles[RuntimeIndex];
        GlobalsAssigned := True;
        HeldFiles := Leases;
        TTVMRuntime.TModule.Initialize(Handles[FFIIndex], True);
        TTVMBackend.TModule.Initialize(Handles[RuntimeIndex], True);
        TTVMRuntime.TModule.ResolveGlobals;
        NativeRequire(
          Assigned(TTVMRuntime.TModule.GModuleLoadFromFile) and Assigned(TTVMRuntime.TModule.GModuleGetFunction) and Assigned(TTVMRuntime.TModule.GTensorCopyFromBytes) and Assigned(TTVMRuntime.TModule.GTensorCopyToBytes),
          'The native runtime VM or tensor API is incomplete'
        );
        SelectedDirectory := FullDirectory;
        SelectedManifestSHA256 := ManifestSHA256;
        SelectedManifestBytes := ManifestBytes;
        RetainDependencies;
      except
        if GlobalsAssigned then
        begin
          { Do not unload code referenced by partially bound global function
            pointers. A failed admission is terminal for this process. }
          RuntimeFailed := True;
          HeldFiles := Leases; RetainDependencies;
        end
        else
          for I := High(Handles) downto 0 do CloseLibrary(Handles[I]);
        raise;
      end;
    finally
      Root.Free;
    end;
  finally
    RuntimeLock.Leave;
  end;
end;

initialization
  RuntimeLock := TCriticalSection.Create;
finalization
  {$IFDEF MSWINDOWS}
  { Complete the selected runtime's DLL cleanup before Windows process detach.
    Applications release all models and join their inference threads first. }
  if (SelectedDirectory <> '') and not RuntimeFailed then TVM.Finalize;
  {$ENDIF}
  HeldFiles := nil;
  RuntimeLock.Free;
end.
