@echo off
rem Double-click to start Year Review. The first start sets everything up by itself.
setlocal
chcp 65001 >nul
cd /d "%~dp0app"
set PYTHONUTF8=1
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo First start: installing uv, which takes care of Python for this app ^(one time^) ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  if errorlevel 1 goto failed
)
uv run --python 3.12 review.py %*
if errorlevel 1 goto failed
goto :eof
:failed
echo.
echo Something went wrong, see above.
pause
