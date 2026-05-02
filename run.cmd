@echo off
setlocal EnableExtensions

REM Minimal reproducible demo pipeline.
REM Double-click this file, or run from Command Prompt:
REM   run.cmd
REM   run.cmd S1
REM   run.cmd S1 S13

cd /d "%~dp0"

set "PYTHON_EXE="
set "PYTHON_ARGS="

if not "%DEMO_PYTHON%"=="" (
    set "PYTHON_EXE=%DEMO_PYTHON%"
    set "PYTHON_ARGS="
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    set "PYTHON_ARGS="
) else (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    ) else (
        python --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
            set "PYTHON_ARGS="
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python was not found.
    echo.
    echo Install Python 3.10 or newer, then run this file again.
    goto :error
)

if not exist "Data\demo_subjects.csv" (
    echo [ERROR] Demo data is missing: Data\demo_subjects.csv
    echo.
    echo Re-download the full Unsupervised-Human-Motion-Analysis repository, including the Data directory.
    goto :error
)

echo ============================================================
echo  Unsupervised-Human-Motion-Analysis demo
echo ============================================================
echo Python command: "%PYTHON_EXE%" %PYTHON_ARGS%
echo.

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import numpy, pandas, scipy, sklearn, skimage, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required Python packages from requirements.txt...
    "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed.
        echo Try running this manually:
        echo   "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
        goto :error
    )
)

if "%~1"=="" (
    echo Subjects: all demo subjects
    "%PYTHON_EXE%" %PYTHON_ARGS% -u demo_pipeline.py
) else (
    echo Subjects: %*
    "%PYTHON_EXE%" %PYTHON_ARGS% -u demo_pipeline.py --subject %*
)
if errorlevel 1 goto :error

echo.
echo [OK] Demo finished. Results are under:
echo %CD%\Results
goto :done

:error
echo.
echo [ERROR] Demo did not finish.
if /I not "%DEMO_NO_PAUSE%"=="1" pause
exit /b 1

:done
if /I not "%DEMO_NO_PAUSE%"=="1" pause
exit /b 0
