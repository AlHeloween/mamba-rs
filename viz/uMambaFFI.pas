unit uMambaFFI;

interface

uses
  System.SysUtils, System.Classes, Winapi.Windows;

type
  TMambaHandle = type Pointer;

  TFfiMambaConfig = record
    d_model: NativeUInt;
    d_state: NativeUInt;
    d_conv: NativeUInt;
    expand: NativeUInt;
    n_layers: NativeUInt;
    use_wavelet: Integer;
    wavelet_levels: NativeUInt;
  end;
  PFfiMambaConfig = ^TFfiMambaConfig;

  TMambaSignalType = (
    mstMultiScale,
    mstRegimeChange,
    mstMarketLike,
    mstWhiteNoise,
    mstBrownianNoise,
    mstMultiFreq
  );

  TSignalGenerateProc = procedure(out: PSingle; len: NativeUInt); cdecl;
  TSignalGenerateProcSeed = procedure(out: PSingle; len: NativeUInt; seed: UInt64); cdecl;
  THaarLevelsFunc = function(seq_len: NativeUInt): NativeUInt; cdecl;
  THaarChannelsLenFunc = function(seq_len: NativeUInt; levels: NativeUInt): NativeUInt; cdecl;
  THaarDecomposeProc = procedure(signal: PSingle; seq_len: NativeUInt; levels: NativeUInt; out_buf: PSingle); cdecl;
  THaarReconstructProc = procedure(channels: PSingle; channels_len: NativeUInt; original_len: NativeUInt; levels: NativeUInt; out_buf: PSingle); cdecl;
  TMambaBackboneNewFunc = function(cfg: PFfiMambaConfig; input_dim: NativeUInt; seq_len: NativeUInt; seed: UInt64): Pointer; cdecl;
  TMambaBackboneFreeProc = procedure(handle: Pointer); cdecl;
  TMambaForwardStepFunc = function(handle: Pointer; input: PSingle; input_len: NativeUInt; output: PSingle): Integer; cdecl;
  TMambaForwardSeqFunc = function(handle: Pointer; input: PSingle; seq_len: NativeUInt; input_dim: NativeUInt; output: PSingle): Integer; cdecl;
  TMambaGetDimFunc = function(handle: Pointer): NativeUInt; cdecl;

  TMambaDLL = class
  private
    FLib: THandle;
    FSignalMultiScale: TSignalGenerateProc;
    FSignalRegimeChange: TSignalGenerateProc;
    FSignalMarketLike: TSignalGenerateProcSeed;
    FSignalWhiteNoise: TSignalGenerateProcSeed;
    FSignalBrownianNoise: TSignalGenerateProcSeed;
    FSignalMultiFreq: TSignalGenerateProc;
    FHaarAutoLevels: THaarLevelsFunc;
    FHaarChannelsLen: THaarChannelsLenFunc;
    FHaarDecompose: THaarDecomposeProc;
    FHaarReconstruct: THaarReconstructProc;
    FMambaNew: TMambaBackboneNewFunc;
    FMambaFree: TMambaBackboneFreeProc;
    FMambaForwardStep: TMambaForwardStepFunc;
    FMambaForwardSeq: TMambaForwardSeqFunc;
    FMambaDModel: TMambaGetDimFunc;
    FMambaInputDim: TMambaGetDimFunc;
    FLoaded: Boolean;
    procedure LoadExports;
  public
    destructor Destroy; override;
    procedure Load(const dllPath: string);
    procedure Unload;
    function Loaded: Boolean;

    procedure GenerateSignal(signalType: TMambaSignalType; outBuf: PSingle; len: NativeUInt; seed: UInt64 = 42);
    function HaarLevels(seq_len: NativeUInt): NativeUInt;
    function HaarChannelsLen(seq_len: NativeUInt; levels: NativeUInt): NativeUInt;
    procedure HaarDecompose(signal: PSingle; seq_len: NativeUInt; levels: NativeUInt; outBuf: PSingle);
    procedure HaarReconstruct(channels: PSingle; channels_len: NativeUInt; original_len: NativeUInt; levels: NativeUInt; outBuf: PSingle);
    function MambaBackboneNew(const cfg: TFfiMambaConfig; input_dim: NativeUInt; seed: UInt64 = 42): TMambaHandle;
    procedure MambaBackboneFree(handle: TMambaHandle);
    function MambaForwardStep(handle: TMambaHandle; input: PSingle; input_len: NativeUInt; output: PSingle): Boolean;
    function MambaForwardSequence(handle: TMambaHandle; input: PSingle; seq_len: NativeUInt; input_dim: NativeUInt; output: PSingle): Boolean;
    function MambaDModel(handle: TMambaHandle): NativeUInt;
    function MambaInputDim(handle: TMambaHandle): NativeUInt;
  end;

implementation

{ TMambaDLL }

destructor TMambaDLL.Destroy;
begin
  Unload;
  inherited;
end;

procedure TMambaDLL.Load(const dllPath: string);
begin
  Unload;
  FLib := LoadLibrary(PChar(dllPath));
  if FLib = 0 then
    raise Exception.CreateFmt('Failed to load DLL: %s', [dllPath]);
  FLoaded := True;
  LoadExports;
end;

procedure TMambaDLL.LoadExports;
begin
  @FSignalMultiScale := GetProcAddress(FLib, 'signal_generate_multiscale');
  @FSignalRegimeChange := GetProcAddress(FLib, 'signal_generate_regime_change');
  @FSignalMarketLike := GetProcAddress(FLib, 'signal_generate_market_like');
  @FSignalWhiteNoise := GetProcAddress(FLib, 'signal_generate_white_noise');
  @FSignalBrownianNoise := GetProcAddress(FLib, 'signal_generate_brownian_noise');
  @FSignalMultiFreq := GetProcAddress(FLib, 'signal_generate_multi_freq');
  @FHaarAutoLevels := GetProcAddress(FLib, 'haar_auto_levels');
  @FHaarChannelsLen := GetProcAddress(FLib, 'haar_channels_len');
  @FHaarDecompose := GetProcAddress(FLib, 'haar_decompose_to_channels');
  @FHaarReconstruct := GetProcAddress(FLib, 'haar_reconstruct_from_channels');
  @FMambaNew := GetProcAddress(FLib, 'mamba_backbone_new');
  @FMambaFree := GetProcAddress(FLib, 'mamba_backbone_free');
  @FMambaForwardStep := GetProcAddress(FLib, 'mamba_backbone_forward_step');
  @FMambaForwardSeq := GetProcAddress(FLib, 'mamba_backbone_forward_sequence');
  @FMambaDModel := GetProcAddress(FLib, 'mamba_backbone_d_model');
  @FMambaInputDim := GetProcAddress(FLib, 'mamba_backbone_input_dim');
end;

procedure TMambaDLL.Unload;
begin
  if FLib <> 0 then
  begin
    FreeLibrary(FLib);
    FLib := 0;
  end;
  FLoaded := False;
  @FSignalMultiScale := nil;
  @FSignalRegimeChange := nil;
  @FSignalMarketLike := nil;
  @FSignalWhiteNoise := nil;
  @FSignalBrownianNoise := nil;
  @FSignalMultiFreq := nil;
  @FHaarAutoLevels := nil;
  @FHaarChannelsLen := nil;
  @FHaarDecompose := nil;
  @FHaarReconstruct := nil;
  @FMambaNew := nil;
  @FMambaFree := nil;
  @FMambaForwardStep := nil;
  @FMambaForwardSeq := nil;
  @FMambaDModel := nil;
  @FMambaInputDim := nil;
end;

function TMambaDLL.Loaded: Boolean;
begin
  Result := FLoaded;
end;

procedure TMambaDLL.GenerateSignal(signalType: TMambaSignalType; outBuf: PSingle; len: NativeUInt; seed: UInt64);
begin
  case signalType of
    mstMultiScale: FSignalMultiScale(outBuf, len);
    mstRegimeChange: FSignalRegimeChange(outBuf, len);
    mstMarketLike: FSignalMarketLike(outBuf, len, seed);
    mstWhiteNoise: FSignalWhiteNoise(outBuf, len, seed);
    mstBrownianNoise: FSignalBrownianNoise(outBuf, len, seed);
    mstMultiFreq: FSignalMultiFreq(outBuf, len);
  end;
end;

function TMambaDLL.HaarLevels(seq_len: NativeUInt): NativeUInt;
begin
  Result := FHaarAutoLevels(seq_len);
end;

function TMambaDLL.HaarChannelsLen(seq_len: NativeUInt; levels: NativeUInt): NativeUInt;
begin
  Result := FHaarChannelsLen(seq_len, levels);
end;

procedure TMambaDLL.HaarDecompose(signal: PSingle; seq_len: NativeUInt; levels: NativeUInt; outBuf: PSingle);
begin
  FHaarDecompose(signal, seq_len, levels, outBuf);
end;

procedure TMambaDLL.HaarReconstruct(channels: PSingle; channels_len: NativeUInt; original_len: NativeUInt; levels: NativeUInt; outBuf: PSingle);
begin
  FHaarReconstruct(channels, channels_len, original_len, levels, outBuf);
end;

function TMambaDLL.MambaBackboneNew(const cfg: TFfiMambaConfig; input_dim: NativeUInt; seed: UInt64): TMambaHandle;
begin
  Result := TMambaHandle(FMambaNew(@cfg, input_dim, 0, seed));
end;

procedure TMambaDLL.MambaBackboneFree(handle: TMambaHandle);
begin
  if handle <> nil then
    FMambaFree(handle);
end;

function TMambaDLL.MambaForwardStep(handle: TMambaHandle; input: PSingle; input_len: NativeUInt; output: PSingle): Boolean;
begin
  Result := FMambaForwardStep(handle, input, input_len, output) <> 0;
end;

function TMambaDLL.MambaForwardSequence(handle: TMambaHandle; input: PSingle; seq_len: NativeUInt; input_dim: NativeUInt; output: PSingle): Boolean;
begin
  Result := FMambaForwardSeq(handle, input, seq_len, input_dim, output) <> 0;
end;

function TMambaDLL.MambaDModel(handle: TMambaHandle): NativeUInt;
begin
  Result := FMambaDModel(handle);
end;

function TMambaDLL.MambaInputDim(handle: TMambaHandle): NativeUInt;
begin
  Result := FMambaInputDim(handle);
end;

end.
