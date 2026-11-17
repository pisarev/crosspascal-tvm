{ ************************************************************************** }
{                                                                            }
{ Qwen35Prompt                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit Qwen35Prompt;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

const
  { Exclusive bound for every rotary axis. Positions below 2^24 convert to
    FP32 exactly before the graph multiplies them by inverse frequencies. }
  Qwen35MaxRopePosition = 16777216;
  Qwen35MaxPromptRows = 1048576;

type
  { Temporal, height and width rotary positions of one decoder row. }
  TQwen35RopePosition = array[0..2] of Int64;

  TQwen35PromptRow = record
    Token: Integer;   { Token ID; the caller's pad ID for visual rows. }
    Visual: Integer;  { -1 for a token embedding, else a row of Embeddings. }
    Rope: TQwen35RopePosition;
  end;

  { Prepared decoder input. The row index is the KV cache position. Visual
    rows take flat row-major FP32 Embeddings in row order instead of token
    embeddings. NextRope is one past the largest rotary position, so decode
    continues at cache position + NextRope - Length(Rows) on all three axes. }
  TQwen35PreparedPrompt = record
    Rows: TArray<TQwen35PromptRow>;
    Embeddings: TArray<Single>;
    VisualCount: Integer;
    NextRope: Int64;
    { Text rows continue at NextRope with equal positions on all axes. }
    procedure AppendText(const IDs: array of Integer);
    { Caller-computed positions; each must be at least NextRope. }
    procedure AppendVisualRows(const PadId: Integer; const Positions: array of TQwen35RopePosition;
      const Features: TArray<Single>; const Width: Integer);
  end;

procedure ValidateQwen35Prompt(const Prompt: TQwen35PreparedPrompt;
  const Width, Vocabulary, Context: Integer);
function Qwen35RopeDelta(const Prompt: TQwen35PreparedPrompt): Int64;

implementation

uses
  {$IFDEF FPC}Math{$ELSE}System.Math{$ENDIF}, TVM.CpuGraph;

const
  MaxEmbeddingValues = High(Integer) div SizeOf(Single);

procedure TQwen35PreparedPrompt.AppendText(const IDs: array of Integer);
var
  Start, I, J: Integer;
  Total: Int64;
begin
  Total := Int64(Length(Rows)) + Length(IDs);
  RequireCpu(Total <= Qwen35MaxPromptRows, 'Qwen prompt has too many rows');
  RequireCpu((NextRope >= 0) and (NextRope <= Qwen35MaxRopePosition) and
    (Length(IDs) <= Qwen35MaxRopePosition - NextRope),
    'Qwen rotary positions exceed the exact FP32 range');
  for I := 0 to High(IDs) do RequireCpu(IDs[I] >= 0, 'Qwen token ID is negative');
  Start := Length(Rows);
  SetLength(Rows, Integer(Total));
  for I := 0 to High(IDs) do
  begin
    Rows[Start + I].Token := IDs[I];
    Rows[Start + I].Visual := -1;
    for J := 0 to 2 do Rows[Start + I].Rope[J] := NextRope;
    Inc(NextRope);
  end;
end;

procedure TQwen35PreparedPrompt.AppendVisualRows(const PadId: Integer;
  const Positions: array of TQwen35RopePosition; const Features: TArray<Single>;
  const Width: Integer);
var
  Start, Offset, I, J: Integer;
  Total, Values, Highest: Int64;
begin
  RequireCpu((PadId >= 0) and (Width > 0) and (Length(Positions) > 0),
    'Qwen visual rows need a pad ID, a width and at least one row');
  Total := Int64(Length(Rows)) + Length(Positions);
  RequireCpu(Total <= Qwen35MaxPromptRows, 'Qwen prompt has too many rows');
  RequireCpu((VisualCount >= 0) and (Int64(VisualCount) * Width = Length(Embeddings)),
    'Qwen visual rows differ in width');
  Values := Int64(Length(Positions)) * Width;
  RequireCpu(Values = Length(Features), 'Qwen visual feature count differs from its rows');
  RequireCpu(Values <= MaxEmbeddingValues - Int64(Length(Embeddings)),
    'Qwen visual features are too large');
  for I := 0 to High(Features) do
    RequireCpu(not IsNan(Features[I]) and not IsInfinite(Features[I]),
      'Non-finite Qwen visual feature');
  RequireCpu(NextRope >= 0, 'Invalid Qwen next rotary position');
  Highest := NextRope - 1;
  for I := 0 to High(Positions) do
    for J := 0 to 2 do
    begin
      RequireCpu((Positions[I][J] >= NextRope) and (Positions[I][J] < Qwen35MaxRopePosition),
        'Qwen visual rotary position precedes earlier rows or exceeds the exact FP32 range');
      if Positions[I][J] > Highest then Highest := Positions[I][J];
    end;
  Start := Length(Rows);
  SetLength(Rows, Integer(Total));
  for I := 0 to High(Positions) do
  begin
    Rows[Start + I].Token := PadId;
    Rows[Start + I].Visual := VisualCount + I;
    Rows[Start + I].Rope := Positions[I];
  end;
  Offset := Length(Embeddings);
  SetLength(Embeddings, Offset + Integer(Values));
  Move(Features[0], Embeddings[Offset], NativeInt(Values) * SizeOf(Single));
  Inc(VisualCount, Length(Positions));
  NextRope := Highest + 1;
end;

procedure ValidateQwen35Prompt(const Prompt: TQwen35PreparedPrompt;
  const Width, Vocabulary, Context: Integer);
var
  I, J, Visual: Integer;
  Highest: Int64;
begin
  RequireCpu((Width > 0) and (Vocabulary > 0) and (Context > 0), 'Invalid Qwen prompt limits');
  RequireCpu((Length(Prompt.Rows) > 0) and (Length(Prompt.Rows) <= Context) and
    (Length(Prompt.Rows) <= Qwen35MaxPromptRows), 'Prompt exceeds compiled Qwen context');
  RequireCpu((Prompt.VisualCount >= 0) and (Length(Prompt.Embeddings) <= MaxEmbeddingValues) and
    (Int64(Prompt.VisualCount) * Width = Length(Prompt.Embeddings)),
    'Qwen visual embeddings differ from the model width');
  Visual := 0;
  Highest := -1;
  for I := 0 to High(Prompt.Rows) do
  begin
    RequireCpu((Prompt.Rows[I].Token >= 0) and (Prompt.Rows[I].Token < Vocabulary),
      'Prompt token is outside tokenizer vocabulary');
    if Prompt.Rows[I].Visual <> -1 then
    begin
      RequireCpu(Prompt.Rows[I].Visual = Visual, 'Qwen visual rows are not in feature order');
      Inc(Visual);
    end;
    for J := 0 to 2 do
    begin
      RequireCpu((Prompt.Rows[I].Rope[J] >= 0) and (Prompt.Rows[I].Rope[J] < Qwen35MaxRopePosition),
        'Qwen rotary position is out of range');
      if Prompt.Rows[I].Rope[J] > Highest then Highest := Prompt.Rows[I].Rope[J];
    end;
  end;
  RequireCpu(Visual = Prompt.VisualCount, 'Qwen visual row count differs from its embeddings');
  RequireCpu(Prompt.NextRope = Highest + 1, 'Qwen next rotary position differs from the prompt');
  for I := 0 to High(Prompt.Embeddings) do
    RequireCpu(not IsNan(Prompt.Embeddings[I]) and not IsInfinite(Prompt.Embeddings[I]),
      'Non-finite Qwen visual feature');
end;

function Qwen35RopeDelta(const Prompt: TQwen35PreparedPrompt): Int64;
begin
  Result := Prompt.NextRope - Length(Prompt.Rows);
end;

end.
