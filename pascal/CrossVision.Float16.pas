{ ************************************************************************** }
{                                                                            }
{ CrossVision.Float16                                                        }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossVision.Float16;

{$IFDEF FPC}
{$MODE DELPHI}
{$ENDIF}

interface

type
  TFloat16 = record
  private
    const
      Float16SignMask = $8000;
      Float16ExponentMask = $7C00;
      Float16FractionMask = $03FF;
      Float16Bias = 15;
      SingleBias = 127;
      Float16MantissaBits = 10;
      SingleMantissaBits = 23;
      MantissaShift = SingleMantissaBits - Float16MantissaBits;
      RoundingGuardBit = Cardinal(1) shl (MantissaShift - 1);
      RoundingStickyMask = RoundingGuardBit - 1;
  public
    var
      Value: Word;
    class function FromSingle(const Float32: Single): TFloat16; static;
    function ToSingle: Single;
  end;

implementation

class function TFloat16.FromSingle(const Float32: Single): TFloat16;
var
  Bits, Sign, Mantissa, HalfExponent, HalfMantissa, GuardMask, StickyMask: LongWord;
  Exponent, TotalShift, Diff: Integer;
  GuardBitIsOne, StickyBitsAreSet, LSBIsOne, RoundUp: Boolean;
begin
  Bits := PCardinal(@Float32)^;
  Sign := (Bits shr 31) and $1;
  Exponent := (Bits shr 23) and $FF;
  Mantissa := Bits and $7FFFFF;
  if Exponent = $FF then
  begin
    if Mantissa <> 0 then
    begin
      HalfMantissa := Mantissa shr MantissaShift;
      if HalfMantissa = 0 then HalfMantissa := 1;
      Result.Value := Word((Sign shl 15) or Float16ExponentMask or HalfMantissa);
    end
    else
      Result.Value := Word((Sign shl 15) or Float16ExponentMask);
    Exit;
  end;
  Result.Value := Word(Sign shl 15);
  if Exponent > 142 then
  begin
    Result.Value := Result.Value or Float16ExponentMask;
    Exit;
  end;
  if Exponent < 113 then
  begin
    Mantissa := Mantissa or (1 shl SingleMantissaBits);
    Diff := 113 - Exponent;
    TotalShift := Diff + MantissaShift;
    if TotalShift > 24 then
      Exit;
    GuardMask := Cardinal(1) shl (TotalShift - 1);
    StickyMask := GuardMask - 1;
    GuardBitIsOne := (Mantissa and GuardMask) <> 0;
    StickyBitsAreSet := (Mantissa and StickyMask) <> 0;
    HalfMantissa := Mantissa shr TotalShift;
    LSBIsOne := (HalfMantissa and 1) <> 0;
    RoundUp := GuardBitIsOne and (StickyBitsAreSet or LSBIsOne);
    if RoundUp then HalfMantissa := HalfMantissa + 1;
    if HalfMantissa = (1 shl Float16MantissaBits) then
      Result.Value := Result.Value or (1 shl Float16MantissaBits)
    else
      Result.Value := Result.Value or Word(HalfMantissa);
    Exit;
  end;
  HalfExponent := (Exponent - SingleBias + Float16Bias);
  GuardBitIsOne := (Mantissa and RoundingGuardBit) <> 0;
  StickyBitsAreSet := (Mantissa and RoundingStickyMask) <> 0;
  HalfMantissa := Mantissa shr MantissaShift;
  LSBIsOne := (HalfMantissa and 1) <> 0;
  RoundUp := GuardBitIsOne and (StickyBitsAreSet or LSBIsOne);
  if RoundUp then
  begin
    HalfMantissa := HalfMantissa + 1;
    if HalfMantissa = (1 shl Float16MantissaBits) then
    begin
      HalfMantissa := 0;
      HalfExponent := HalfExponent + 1;
      if HalfExponent = (Float16ExponentMask shr Float16MantissaBits) then
      begin
        Result.Value := Result.Value or Float16ExponentMask;
        Exit;
      end;
    end;
  end;
  Result.Value := Result.Value or Word(HalfExponent shl Float16MantissaBits) or Word(HalfMantissa);
end;

function TFloat16.ToSingle: Single;
type
  TFloatUnion = record
    case Integer of
      0: (Float32: Single);
      1: (Int32: Cardinal);
  end;
var
  Sign, Mantissa: LongWord;
  Exponent: Integer;
  FloatUnion: TFloatUnion;
begin
  Sign := (Value shr 15) and $1;
  Exponent  := (Value shr 10) and $1F;
  Mantissa := Value and $3FF;
  if Exponent = 0 then
  begin
    if Mantissa = 0 then
      FloatUnion.Int32 := Sign shl 31
    else begin
      Exponent := 1;
      while (Mantissa and $400) = 0 do
      begin
        Mantissa := Mantissa shl 1;
        Dec(Exponent);
      end;
      Mantissa := Mantissa and $3FF;
      Exponent := Exponent + (127 - 15);
      FloatUnion.Int32 := (Sign shl 31) or (LongWord(Exponent) shl 23) or (Mantissa shl 13);
    end;
  end
  else if Exponent = 31 then
      FloatUnion.Int32 := (Sign shl 31) or ($FF shl 23) or (Mantissa shl 13)
    else begin
      Exponent := Exponent + (127 - 15);
      FloatUnion.Int32 := (Sign shl 31) or (LongWord(Exponent) shl 23) or (Mantissa shl 13);
    end;
  Result := FloatUnion.Float32;
end;

end.
