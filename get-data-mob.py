"""
Usage: py get-data.py [country]
If parameter isn't specified, switzerland will be used

Goals:
Plot containing 4 lines:
1. Traffic Data of country -> walking
2. Traffic Data of country -> driving
3. Traffic Data of country -> transit
4. COVID Data of country
5. New dataset on whether there was a lockdown or not

    Dates of lockdown:
    - 16.03.20-11.05.20,
    - 18.10.20-17.02.21,

Traffic Data needs to be normalized, to account for weekends/days off

-> Predict COVID Data for the next 2 weeks
-> Predict traffic Data for the next 2 weeks
"""

from helper import *

try:
    country = argv[1]
except IndexError:
    country = "switzerland" # default

confirmed_df, deaths_df, latest_data, us_medical_data, apple_mobility = get_data()

# convert apple mobility data from |  geo_type | region | alternative_name | sub_region | country | DATE | ... | DATE
# to dict

# Create Data Structures
default_mob_data_dict = {}

try:
    for i, key in enumerate(apple_mobility):
        default_mob_data_dict[key] = None
except KeyError:
    pass

index = []

# Get corresponding data rows for country:
for i, v in enumerate(apple_mobility.region):
    if v.upper() == country.upper():
        print("yas queen")
        index.append(i)

datasets = []
# Add Values to data structure
try:
    x = 1
    for i in index:
        mob_data_dict = default_mob_data_dict.copy()
        for k, v in mob_data_dict.items():
            mob_data_dict[k] = apple_mobility.loc[i + 2 + x][k]
        x+=1
        datasets.append(mob_data_dict)
        del mob_data_dict
        #print(datasets)
except KeyError:
    pass

# Values to x and y axis
x, y = [], []
i = -1

datasets_as_xy = []
prev_dataset = None
for dataset in datasets:
    if prev_dataset == dataset:
        print("Shit")
    prev_dataset = dataset

    i= -1
    temp = []
    for k, v in dataset.items():
        i+=1
        if i < 6:
            continue
        else:
            x.append(k); y.append(v)
            temp.append(x)
            temp.append(y)
            datasets_as_xy.append(temp)

plt.figure(figsize=(16, 10))

x = datasets_as_xy[0][0]
y = datasets_as_xy[0][1]
x1 = datasets_as_xy[1][0]
y1 = datasets_as_xy[1][1]

plt.plot(x,y)
plt.plot(x1, y1)
#plt.plot(datasets_as_xy[1][0], color="orange")
#plt.plot(datasets_as_xy[2][0], color="green")

plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel(' Increase of traffic routing requests in %, baseline at 100', size = 20)
plt.xticks(size=10, rotation=90, ticks=[1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
plt.yticks(size=10)
plt.grid()
plt.legend([f"Traffic requests for {country}"], loc=9)
plt.show()

