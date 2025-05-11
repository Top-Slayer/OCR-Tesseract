@echo off

@REM  Checking Python 
setlocal

echo Checking Python...

python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not installed. Will download and install...
    python --version
    goto DOWNLOAD
)

for /f "tokens=2 delims= " %%A in ('python --version') do set CURRENT_VERSION=%%A
echo Found Python version: %CURRENT_VERSION%

if "%CURRENT_VERSION%"=="3.11.9" (
    echo Python version matches! No action needed.
    goto END
) else (
    echo Python version does not match. Need to install correct version...
    goto DOWNLOAD
)

:Download
set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=%TEMP%\python_installer.exe
curl %PYTHON_URL% -o %PYTHON_INSTALLER%

if exist "%PYTHON_INSTALLER%" (
    echo Download complete!
    echo Installing silently...

    :: Silent install Python
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

    echo Waiting for installation to finish...
    timeout /t 10

    echo Checking installation...
    python --version
) else (
    echo Failed to download Python installer.
)

:END


@REM  Checking CRAFT-pytorch folder
echo Checking CRAFT-pytorch folder ...
if exist "CRAFT-pytorch\" (
    echo CRAFT-pytorch exists.
) else (
    echo Installing CRAFT-pytorch ...
    curl -L https://github.com/clovaai/CRAFT-pytorch/archive/e332dd8b718e291f51b66ff8f9ef2c98ee4474c8.zip -o CRAFT.zip
    tar -xvf CRAFT.zip
    ren CRAFT-pytorch-e332dd8b718e291f51b66ff8f9ef2c98ee4474c8 CRAFT-pytorch
    timeout /t 2 /nobreak >nul
    del CRAFT.zip
)

if not exist "craft_mlt_25k.pth" (
    echo Installing craft_mlt_25k.pth ...
    curl -L -O https://drive.usercontent.google.com/u/0/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ&export=download
    ren uc craft_mlt_25k.pth
)


@REM  Checking Real-ESRGAN folder
echo Checking Real-ESRGAN folder ...
if exist "Real-ESRGAN\" (
    echo Real-ESRGAN exists.
) else (
    echo Installing Real-ESRGAN ...
    curl -L -O https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
    mkdir Real-ESRGAN
    tar -xvf realesrgan-ncnn-vulkan-20220424-windows.zip -C Real-ESRGAN
    timeout /t 2 /nobreak >nul
    del realesrgan-ncnn-vulkan-20220424-windows.zip
)


@REM  Checking Tesserect 
echo Checking Tesserect...
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Tesseract is installed.
) else (
    echo Installing Tesseract ...
    curl -L https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe -o tesseract.exe
    tesseract.exe
    timeout /t 2 /nobreak >nul
    del tesseract.exe
)

echo [ Install dependecies successfully ]

timeout /t 3