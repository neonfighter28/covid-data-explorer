#!/bin/python

import os
from config import TEST_CASES as TESTS

PYTHON_PATH = f"{os.getcwd()}/venv/bin/python"

for i in TESTS:
    os.system(f"echo {i} | {PYTHON_PATH} main.py")