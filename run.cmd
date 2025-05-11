@echo off

call install-dep.cmd

if not exist "env\" (
    pip install virtualenv
    python -m venv env
    pip install -r requirements.txt
) 

call env\Scripts\activate
python main.py
deactivate