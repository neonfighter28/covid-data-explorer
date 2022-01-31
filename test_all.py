#!/bin/python

import os
import subprocess
import get_data
import main

SHOW = " -s False"

if os.name == "nt":
    PYTHON_PATH = f"{os.getcwd()}\\venv\\scripts\\python.exe"
elif os.name == "posix":
    PYTHON_PATH = f"{os.getcwd()}/venv/bin/python"


def run(BASE_TEST, CASES, EOL=""):
    for test in CASES:
        print(f"echo {BASE_TEST+SHOW} {test} | {PYTHON_PATH} main.py")
        s = subprocess.run([PYTHON_PATH, "main.py", f"{BASE_TEST+SHOW} {test} {EOL}"])
        assert s.returncode == 0


def test_switzerland():
    BASE_TEST = "plot"
    CASES = [
        "-d cs",
        "-d re+mb",
        "-d str+ld",
        "-d str+ld+cs+mb"
    ]
    run(BASE_TEST, CASES)


def test_single_country():
    BASE_TEST = "plot -c germany"
    CASES = [
        "-d cs",
        "-d re+mb",
        "-d str+ld",
        "-d str+ld+cs+mb"
    ]
    run(BASE_TEST, CASES)


def test_multiple_countries():
    BASE_TEST = "plot -cc germany+italy"
    CASES = [
        "-d cs",
        "-d re+mb",
        "-d str+ld",
        "-d re+cs+mb+str",
        "-d total_cases"
    ]
    run(BASE_TEST, CASES)



def test_arbitrary():
    BASE_TEST = "plot -c germany"
    CASES = [
        "-d total_cases",
        "-d new_cases"
    ]
    run(BASE_TEST, CASES)


def test_sep_names():
    BASE_TEST = "plot -c United-States"
    CASES = [
        "-d total_cases",
        "-d new_cases",
        "-d cs",
        "-d re+mb",
        "-d str+ld",
        "-d re+cs+mb+str"
    ]
    run(BASE_TEST, CASES)


# def test_bad_inputs():
#     BASE_TEST = "plot "
#     EOL = "\nplot -d cs"
#     CASES = [
#         "ld+cs",  # omit -d
#         "United+States",  # bad countries
#         "-d switerland"  # spelling mistake
#     ]
#     run(BASE_TEST, CASES, EOL)
