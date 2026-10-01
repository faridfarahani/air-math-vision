@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo Air Math Vision - Diagnostics
echo ==================================================
echo.

echo [1] Python:
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version
) else (
    echo Virtual environment missing.
)
echo.

echo [2] MediaPipe:
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import mediapipe as mp; print('Version:',mp.__version__); print('Has solutions:',hasattr(mp,'solutions')); print('Package:',mp.__file__)"
)
echo.

echo [3] Core imports:
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import fastapi,uvicorn,cv2,numpy; print('FastAPI OK'); print('Uvicorn OK'); print('OpenCV:',cv2.__version__); print('NumPy:',numpy.__version__)"
)
echo.

echo [4] Port 8000:
netstat -ano | findstr ":8000"
echo.

echo [5] Server log:
if exist "server.log" (
    type "server.log"
) else (
    echo No server.log.
)

echo.
pause
