@echo off
setlocal
set "PATH=D:\USESoft\RAD_Pascal\Bin;D:\USESoft\RAD_Pascal\Bin64;%PATH%"
cd /d "%~dp0"
dcc64 -LN. -E. -NSWinapi;System;Vcl MambaTestCLI.dpr
endlocal
