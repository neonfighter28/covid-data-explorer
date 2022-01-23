HELP_DATA = """
Usage of the data argument:

!Important:
All data arguments need to be concatenated by a "+" sign
Ex:     reproduction+cases
        cs+mb+ld

Supported values are currently:
    -   re | reproduction
    -   cs | cases
    -   mb | mobility
    -   ld | lockdown

Unsupported values can still be plotted, although there is no
shortened version of them, and their names will be the same as in the dataset
There is also a high chance they will not be displayed correctly

All unsupported values are listed in the table below. Their explanation is found
under the official dataset documentation:
https://github.com/owid/covid-19-data/tree/master/public/data

--------------------------------------------------------------------------------
| Meta:                                                                        |
|------------------------------------------------------------------------------|
| continent          | location         | date             | iso_code          |
|------------------------------------------------------------------------------|
| Cases & Deaths:                                                              |
|------------------------------------------------------------------------------|
| total_cases                           | total_deaths                         |
| new_deaths                            | new_deaths_smoothed                  |
| total_cases_per_million               | new_cases_per_million                |
| new_cases_smoothed_per_million        | total_deaths_per_million             |
| new_deaths_per_million                | new_deaths_smoothed_per_million      |
| reproduction_rate                     |                                      |
|------------------------------------------------------------------------------|
| Hospital Data                          |                                     |
|------------------------------------------------------------------------------|
| icu_patients                          | weekly_hosp_admissions_per_million   |
| icu_patients_per_million              | hosp_patients                        |
| hosp_patients_per_million             | weekly_icu_admissions                |
| weekly_icu_admissions_per_million     | weekly_hosp_admissions               |
| weekly_hosp_admissions_per_million    |                                      |
|------------------------------------------------------------------------------|
| Testing Data                                                                 |
|------------------------------------------------------------------------------|
| new_tests                             | tests_per_case                       |
| total_tests                           | total_tests_per_thousand             |
| new_tests_per_thousand                | new_tests_smoothed                   |
| new_tests_smoothed_per_thousand       | positive_rate                        |
| tests_units                           |                                      |
|------------------------------------------------------------------------------|
| Vaccination Data                                                             |
|------------------------------------------------------------------------------|
| total_vaccinations                    | people_vaccinated                    |
| people_fully_vaccinated               | total_boosters                       |
| new_vaccinations                      | new_vaccinations_smoothed            |
| total_vaccinations_per_hundred        | people_vaccinated_per_hundred        |
| people_fully_vaccinated_per_hundred   | total_boosters_per_hundred           |
| new_vaccinations_smoothed_per_million | new_people_vaccinated_smoothed       |
| new_people_vaccinated_smoothed_per_hundred                                   |
|------------------------------------------------------------------------------|
| General Population Data                                                      |
|------------------------------------------------------------------------------|
| population                            | population_density                   |
| median_age                            | aged_65_older                        |
| aged_70_older                         | gdp_per_capita                       |
| extreme_poverty                       | cardiovasc_death_rate                |
| diabetes_prevalence                   | female_smokers                       |
| male_smokers                          | handwashing_facilities               |
| hospital_beds_per_thousand            | life_expectancy                      |
| human_development_index               | excess_mortality_cumulative_absolute |
| excess_mortality_cumulative           | excess_mortality                     |
| excess_mortality_cumulative_per_million                                      |
--------------------------------------------------------------------------------
"""

HELP_COUNTRY = """
Usage of the country argument:
If the country name is split by a whitespace, e.g. United States,
the whitespace must be replaced by a dash, "-"

e.g. "United States" -> "United-States"
"""
