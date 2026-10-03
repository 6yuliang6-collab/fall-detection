@echo off
cd /d "%~dp0"

echo ==========================================
echo   Fall Detection - One-click Start
echo ==========================================
echo.

echo [1/3] Starting cloud backend + dashboard ...
start "cloud-backend" cmd /k "venv\Scripts\python.exe system\cloud_backend.py"
timeout /t 3 /nobreak >nul

echo [2/3] Starting camera detector ...
start "edge-detector" cmd /k "venv\Scripts\python.exe system\edge_detector.py --source 0"
timeout /t 2 /nobreak >nul

echo [3/3] Opening dashboard in browser ...
start http://127.0.0.1:5000

echo.
echo Done! Two terminal windows + browser opened.
echo   - Camera window : press q in the video to quit
echo   - Cloud window  : press Ctrl+C to stop
echo   - Dashboard     : http://127.0.0.1:5000
echo.
pause
