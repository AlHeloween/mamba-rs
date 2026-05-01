program MambaVisualizer;

uses
  Vcl.Forms,
  uMain in 'uMain.pas' {FormMain},
  uMambaFFI in 'uMambaFFI.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TFormMain, FormMain);
  Application.Run;
end.
