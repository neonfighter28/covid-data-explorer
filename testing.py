#!/bin/python

import os
import sys
from config import TEST_CASES as TESTS

if os.name == "nt":
    PYTHON_PATH = f"{os.getcwd()}\\venv\\scripts\\python.exe"
elif os.name == "posix":
    PYTHON_PATH = f"{os.getcwd()}/venv/bin/python"
try:
    if sys.argv[1] == "--full":
        for i in TESTS:
            os.system(f"echo {i} | {PYTHON_PATH} main.py")
except IndexError:
    os.system(f"echo {TESTS[0]} | {PYTHON_PATH} main.py")