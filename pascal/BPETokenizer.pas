{ ************************************************************************** }
{                                                                            }
{ BPETokenizer                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit BPETokenizer;

{$POINTERMATH ON}

interface

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF};

type
  TCustomBPETokenizer = class(TObject)
  private
    FKeys: array of Int64;
    FRank: array of Integer;
    FJoined: array of Integer;
    FMask: Integer;
  protected
    function Slot(const Key: Int64): Integer;
    function MergeKey(const Left, Right: Integer): Int64;
    procedure ClearMerges;
    procedure ReserveMerges(const PairCount: Integer);
    procedure Put(const Left, Right, Joined, Rank: Integer);
    function Find(const Left, Right: Integer; out Rank, Joined: Integer): Boolean;
    function MergeBPE(var Ids: TArray<Integer>; const Count: Integer): Integer;
    function ReadWholeFile(const Path: string; out Data: TBytes): Boolean;
    procedure RequireLoaded;
  public
    destructor Destroy; override;
    function Load(const Directory: string; const Error: PString): Boolean; virtual; abstract;
    function Loaded: Boolean; virtual; abstract;
    function TokenCount: Integer; virtual; abstract;
    function MergeCount: Integer;
    function Encode(const Text: UnicodeString): TArray<Integer>; overload; virtual; abstract;
    function Decode(const Ids: array of Integer): UnicodeString; virtual; abstract;
    function SpecialId(const Text: UnicodeString): Integer; virtual; abstract;
  end;

implementation

uses
  {$IFDEF FPC}Classes{$ELSE}System.Classes{$ENDIF};

destructor TCustomBPETokenizer.Destroy;
begin
  inherited Destroy;
end;

function TCustomBPETokenizer.Slot(const Key: Int64): Integer;
begin
  Result := Integer((UInt64(Key) * UInt64(2654435761)) shr 11) and FMask;
end;

function TCustomBPETokenizer.MergeKey(const Left, Right: Integer): Int64;
begin
  Result := (Int64(Left) shl 20) or Int64(Right);
end;

procedure TCustomBPETokenizer.ClearMerges;
begin
  FMask := 0;
  SetLength(FKeys, 0);
  SetLength(FRank, 0);
  SetLength(FJoined, 0);
end;

procedure TCustomBPETokenizer.ReserveMerges(const PairCount: Integer);
var
  Capacity: Integer;
begin
  Capacity := 1;
  while Capacity < PairCount * 2 do
    Capacity := Capacity * 2;
  FMask := Capacity - 1;
  SetLength(FKeys, Capacity);
  SetLength(FRank, Capacity);
  SetLength(FJoined, Capacity);
end;

procedure TCustomBPETokenizer.Put(const Left, Right, Joined, Rank: Integer);
var
  Key: Int64;
  At: Integer;
begin
  Key := MergeKey(Left, Right);
  At := Slot(Key);
  while FKeys[At] <> 0 do
  begin
    if FKeys[At] = Key + 1 then
      Exit;
    At := (At + 1) and FMask;
  end;
  FKeys[At] := Key + 1;
  FRank[At] := Rank;
  FJoined[At] := Joined;
end;

function TCustomBPETokenizer.Find(const Left, Right: Integer; out Rank, Joined: Integer): Boolean;
var
  Key: Int64;
  At: Integer;
begin
  Rank := 0;
  Joined := 0;
  if (Left < 0) or (Right < 0) then
    Exit(False);
  Key := MergeKey(Left, Right);
  At := Slot(Key);
  while FKeys[At] <> 0 do
  begin
    if FKeys[At] = Key + 1 then
    begin
      Rank := FRank[At];
      Joined := FJoined[At];
      Exit(True);
    end;
    At := (At + 1) and FMask;
  end;
  Result := False;
end;

function TCustomBPETokenizer.MergeBPE(var Ids: TArray<Integer>; const Count: Integer): Integer;
var
  Live, I, BestAt, BestRank, Rank, Joined, NewJoined: Integer;
begin
  Live := Count;
  while Live > 1 do
  begin
    BestAt := -1;
    BestRank := MaxInt;
    NewJoined := 0;
    for I := 0 to Live - 2 do
      if Find(Ids[I], Ids[I + 1], Rank, Joined) and (Rank < BestRank) then
      begin
        BestRank := Rank;
        BestAt := I;
        NewJoined := Joined;
      end;
    if BestAt < 0 then
      Break;
    Ids[BestAt] := NewJoined;
    for I := BestAt + 1 to Live - 2 do
      Ids[I] := Ids[I + 1];
    Dec(Live);
  end;
  Result := Live;
end;

function TCustomBPETokenizer.ReadWholeFile(const Path: string; out Data: TBytes): Boolean;
var
  Stream: TFileStream;
  Size: Integer;
begin
  Result := False;
  SetLength(Data, 0);
  try
    Stream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
    try
      if Stream.Size > High(Integer) then
        Exit;
      Size := Integer(Stream.Size);
      SetLength(Data, Size);
      if Size = 0 then
        Exit(True);
      Result := Stream.Read(Data[0], Size) = Size;
    finally
      Stream.Free;
    end;
  except
    on E: EStreamError do Result := False;
    on E: EOSError do Result := False;
  end;
end;

procedure TCustomBPETokenizer.RequireLoaded;
begin
  if not Loaded then
    raise Exception.CreateFmt('%s: tokenizer is not loaded; call Load first',
      [ClassName]);
end;

function TCustomBPETokenizer.MergeCount: Integer;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(FKeys) do
    if FKeys[I] <> 0 then
      Inc(Result);
end;

end.
