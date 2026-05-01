program MambaVisualizerCLI;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  uMambaFFI in 'uMambaFFI.pas';

var
  DLL: TMambaDLL;
  Signal: TArray<Single>;
  Channels: TArray<Single>;
  Reconstructed: TArray<Single>;
  i: Integer;
  seqLen, levels, channelsLen: NativeUInt;
begin
  try
    WriteLn('=== Mamba-rs FFI Test ===');
    WriteLn;

    DLL := TMambaDLL.Create;
    try
      DLL.Load(ExtractFilePath(ParamStr(0)) + 'mamba_rs.dll');
      WriteLn('DLL loaded successfully');
      WriteLn;

      seqLen := 64;
      SetLength(Signal, seqLen);
      DLL.GenerateSignal(mstMultiScale, @Signal[0], seqLen, 42);

      WriteLn('Generated multi-scale signal (first 10 values):');
      for i := 0 to 9 do
        WriteLn(Format('  [%d] = %.6f', [i, Signal[i]]));
      WriteLn;

      levels := 3;
      channelsLen := DLL.HaarChannelsLen(seqLen, levels);
      SetLength(Channels, channelsLen);
      DLL.HaarDecompose(@Signal[0], seqLen, levels, @Channels[0]);

      WriteLn(Format('Wavelet decomposition (levels=%d, channels_len=%d)', [levels, channelsLen]));
      WriteLn('First 10 channel values:');
      for i := 0 to 9 do
        WriteLn(Format('  [%d] = %.6f', [i, Channels[i]]));
      WriteLn;

      SetLength(Reconstructed, seqLen);
      DLL.HaarReconstruct(@Channels[0], channelsLen, seqLen, levels, @Reconstructed[0]);

      WriteLn('Reconstruction error (should be ~0):');
      for i := 0 to 9 do
        WriteLn(Format('  [%d] signal=%.6f reconstructed=%.6f error=%.9f',
          [i, Signal[i], Reconstructed[i], Abs(Signal[i] - Reconstructed[i])]));

      WriteLn;
      WriteLn('=== FFI test passed ===');
    finally
      DLL.Free;
    end;
  except
    on E: Exception do
      WriteLn('Error: ', E.Message);
  end;
end.
