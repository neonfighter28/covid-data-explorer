# Covid Data Explorer

[![Project Tests](https://github.com/neonfighter28/covid-data-explorer/actions/workflows/pytest.yml/badge.svg)](https://github.com/neonfighter28/covid-data-explorer/actions/workflows/pytest.yml)

## Usage

    python main.py [args]

## 1. Idea

### Product

>    TUI, handling the display of Covid-Data of graphs of multiple datasets

### Layer-Based Design

#### Layer 1

>    User Input/Parsing and displaying of graphs

#### Layer 2

>    Data structures and preparing for output

#### Layer 3

>    Data layer, interfacing with web/cache

## 2. Technologies used

## Libraries

    - requests      - parse websites for data URLs
    - matplotlib    - display graphs
    - pandas        - CSV processing
    - tqdm          - progress bar
    - jedi          - lang server
    - mypy          - typehints
    - numpy         - calculations
    - six           - compat
    - flake8        - linting
    - autopep8      - formatting

## Data Sources

>Lockdown measures by statistikZH

    https://raw.githubusercontent.com/statistikZH/covid19zeitmarker/master/covid19zeitmarker.csv

>Global measures by Oxford

    https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv

>Global Covid data by Our World In Data

    https://covid.ourworldindata.org/data/owid-covid-data.csv

>Global mobility data by apple

    https://covid19-static.cdn-apple.com/

>Reproduction values, provided by Bundesamt für Gesundheit

    https://www.covid19.admin.ch/api/data/documentation/models/sources-definitions-redailyincomingdata.md

## 4. Usage

    ```txt
    plt | plot
        --country   | -c    [COUNTRY]       | Default: switzerland
        NotImplemented --startdate | -sd   [DD.MM.YY]      | Default: None
        NotImplemented --enddate   | -ed   [DD.MM.YY]      | Default: None
        --show      | -s    [bool]          | Default: True Whether to show the plot
        --data      | -d    [ARG+ARG+...]   | Default: None
    help
        data
        country
    ```
