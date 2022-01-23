import os

PYTHON_PATH = f"{os.getcwd()}\\venv\\scripts\\python.exe"

TESTS = [
    "plot -s False -d cs",
    "plot -c germany -s False -d cs+str+re+mb"
]

for i in TESTS:
    os.system(f"echo {i} | {PYTHON_PATH} main.py")