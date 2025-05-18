@echo off

call install-dep.cmd

setlocal

if not exist "env\" (
    echo Creating virtual environment...
    pip install virtualenv
    python -m venv env
)

call env\Scripts\activate

pip install -r requirements.txt
timeout /t 3
python main.py

deactivate
endlocal
