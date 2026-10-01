@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo Air Math Vision - Windows setup
echo ==================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" was not found.
    echo Install Python 3.12 x64 from python.org and enable the Python launcher.
    pause
    exit /b 1
)

set "PY_VER="

py -3.12 --version >nul 2>nul
if not errorlevel 1 (
    set "PY_VER=-3.12"
    goto :python_found
)

py -3.11 --version >nul 2>nul
if not errorlevel 1 (
    set "PY_VER=-3.11"
    goto :python_found
)

echo [ERROR] Python 3.12 or Python 3.11 was not found.
echo Recommended: Python 3.12 x64.
echo.
py -0p
pause
exit /b 1

:python_found
echo Using:
py %PY_VER% --version
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/6] Creating virtual environment...
    py %PY_VER% -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [1/6] Existing virtual environment detected.
    ".venv\Scripts\python.exe" --version
)

echo [2/6] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/6] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [4/6] Verifying MediaPipe legacy Solutions API...
".venv\Scripts\python.exe" -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); assert mp.__version__ == '0.10.21', 'Wrong MediaPipe version'; assert hasattr(mp, 'solutions'), 'mp.solutions missing'; print('MediaPipe Solutions API: OK')"
if errorlevel 1 (
    echo.
    echo Repairing MediaPipe installation...
    ".venv\Scripts\python.exe" -m pip uninstall -y mediapipe
    if errorlevel 1 goto :fail

    ".venv\Scripts\python.exe" -m pip install --no-cache-dir --force-reinstall "mediapipe==0.10.21"
    if errorlevel 1 goto :fail

    ".venv\Scripts\python.exe" -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); assert hasattr(mp, 'solutions'); print('MediaPipe Solutions API: OK')"
    if errorlevel 1 goto :fail
)

if not exist "models" mkdir "models"

echo [5/6] Checking YuNet face detector...
if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo Downloading YuNet model...
    curl.exe -L --fail --retry 3 ^
      "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" ^
      -o "models\face_detection_yunet_2023mar.onnx"
    if errorlevel 1 goto :fail
) else (
    echo YuNet model already exists.
)

echo [6/6] Checking SFace recognizer...
if not exist "models\face_recognition_sface_2021dec.onnx" (
    echo Downloading SFace model...
    curl.exe -L --fail --retry 3 ^
      "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" ^
      -o "models\face_recognition_sface_2021dec.onnx"
    if errorlevel 1 goto :fail
) else (
    echo SFace model already exists.
)

echo.
echo ==================================================
echo Setup completed successfully.
echo MediaPipe 0.10.21 + Solutions API verified.
echo Now run start_windows.bat
echo ==================================================
pause
exit /b 0

:fail
echo.
echo ==================================================
echo [ERROR] Setup failed.
echo ==================================================
echo Read the error above.
pause
exit /b 1
