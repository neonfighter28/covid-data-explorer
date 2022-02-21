#!/bin/python

import os
import subprocess

SHOW = " -s False"

if os.name == "nt":
    PYTHON_PATH = f"{os.getcwd()}\\venv\\scripts\\python.exe"
elif os.name == "posix":
    PYTHON_PATH = f"{os.getcwd()}/venv/bin/python"

OUT = ""

subprocess.run([PYTHON_PATH, f"{os.getcwd()}/refresh_data.py"], capture_output=OUT)

print(OUT)