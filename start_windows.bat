@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo Air Math Vision - Windows
echo ==================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo [ERROR] Missing YuNet face detection model.
    echo Run setup_windows.bat again.
    pause
    exit /b 1
)

if not exist "models\face_recognition_sface_2021dec.onnx" (
    echo [ERROR] Missing SFace recognition model.
    echo Run setup_windows.bat again.
    pause
    exit /b 1
)

echo Checking Python imports...
".venv\Scripts\python.exe" -c "import fastapi,uvicorn,cv2,numpy,mediapipe as mp; print('MediaPipe:',mp.__version__); assert mp.__version__=='0.10.21', 'Run setup_windows.bat: MediaPipe must be 0.10.21'; assert hasattr(mp,'solutions'), 'Run setup_windows.bat: mp.solutions missing'; print('Python dependencies: OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency check failed.
    echo Run setup_windows.bat once, then start again.
    pause
    exit /b 1
)

if exist "server.log" del /q "server.log" >nul 2>nul

echo Starting Air Math Vision server...
start "Air Math Vision Server" /MIN cmd /c ""%CD%\.venv\Scripts\python.exe" "%CD%\run.py" > "%CD%\server.log" 2>&1"

echo Waiting for http://127.0.0.1:8000 ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline=(Get-Date).AddSeconds(30);" ^
  "while((Get-Date) -lt $deadline) {" ^
  "  try {" ^
  "    $r=Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/health' -TimeoutSec 2;" ^
  "    if($r.StatusCode -eq 200){ exit 0 }" ^
  "  } catch {};" ^
  "  Start-Sleep -Milliseconds 500" ^
  "}; exit 1"

if errorlevel 1 (
    echo.
    echo ==================================================
    echo [ERROR] Server did not start.
    echo ==================================================
    echo.
    echo Last server output:
    echo --------------------------------------------------
    if exist "server.log" (
        type "server.log"
    ) else (
        echo server.log was not created.
    )
    echo --------------------------------------------------
    pause
    exit /b 1
)

echo Server is ONLINE.
echo Opening dashboard...
start "" "http://127.0.0.1:8000"

echo.
echo Dashboard:
echo http://127.0.0.1:8000
echo.
pause
