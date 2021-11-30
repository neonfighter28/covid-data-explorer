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

cols = confirmed_df.keys()

confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
# recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]

dates = confirmed.keys()
world_cases = []
total_deaths = []
mortality_rate = []
# recovery_rate = []
# total_recovered = []
# total_active = []

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
#     recovered_sum = recoveries[i].sum()

    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
#     total_recovered.append(recovered_sum)
#     total_active.append(confirmed_sum-death_sum-recovered_sum)

    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
#     recovery_rate.append(recovered_sum/confirmed_sum)


# window size
window = 7

# confirmed cases
world_daily_increase = daily_increase(world_cases)
world_confirmed_avg= moving_average(world_cases, window)
world_daily_increase_avg = moving_average(world_daily_increase, window)

# deaths
world_daily_death = daily_increase(total_deaths)
world_death_avg = moving_average(total_deaths, window)
world_daily_death_avg = moving_average(world_daily_death, window)


# recoveries
# world_daily_recovery = daily_increase(total_recovered)
# world_recovery_avg = moving_average(total_recovered, window)
# world_daily_recovery_avg = moving_average(world_daily_recovery, window)


# active
# world_active_avg = moving_average(total_active, window)



days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
# total_recovered = np.array(total_recovered).reshape(-1, 1)

days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



# slightly modify the data to fit the model better (regression models cannot pick the pattern)
days_to_skip = 376
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22[days_to_skip:], world_cases[days_to_skip:], test_size=0.08, shuffle=False)

# convert apple mobility data from |  geo_type | region | alternative_name | sub_region | country | DATE | ... | DATE
# to dict

# Create Data Structures
def_mob_data_dict = {}

class MobilityData:
    def __init__(self):
        self.data = None
        self.x = None
        self.y = None

class AppleMobilityDataCountry:
    def __init__(self):
        self.transit_data = None
        self.walking_data = None
        self.driving_data = None

    class MobilityData:
        def __init__(self, geo_type, region, transportation_type, alternative_name, sub_region, country, data, index):
            self.geo_type = geo_type
            self.region = region
            self.transportation_type = transportation_type
            self.alternative_name = alternative_name
            self.sub_region = sub_region
            self.country = country
            self.data = data
            self.index = index


    def get_data_rows_from_index(self):
        region = apple_mobility.region[self.index]
        geo_type = apple_mobility.geo_type[self.index]
        transportation_type = apple_mobility.transportation_type[self.index]
        alternative_name = apple_mobility.alternative_name[self.index]
        sub_region = apple_mobility.sub_region[self.index]

        match transportation_type:
            case ["walking"]:
                self.walking_data = MobilityData(region, geo_type, transportation_type, alternative_name, sub_region, country, data, index)
                MobilityData.transportation_type = "walking"
                MobilityData.region = region
                


mob_data = AppleMobilityDataCountry()

try:
    for i, key in enumerate(apple_mobility):
        def_mob_data_dict[key] = []
except KeyError:
    pass

# Get corresponding data rows for country:
for i, v in enumerate(apple_mobility.region):
    if v.upper() == country.upper():
        print("yas queen")
        mob_data.index = i

mob_data.get_data_rows_from_index()

datasets = []
# Add Values to data structure
try:
    for i in range(3):
        mob_data_dict = def_mob_data_dict
        for k, v in mob_data_dict.items():
            mob_data_dict[k] = apple_mobility.loc[index + 2][k]
        datasets.append(mob_data_dict)
        print(datasets)
except KeyError:
    pass

# Values to x and y axis
x, y = [], []
i = -1

datasets_as_xy = []
for dataset in datasets:
    temp = []
    for k, v in dataset.items():
        i+=1
        if i < 6:
            continue
        x.append(k); y.append(v)
    temp.append(x)
    temp.append(y)
    datasets_as_xy.append(temp)

adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 10))
plt.plot(datasets_as_xy[0][0], datasets_as_xy[0][1], color="blue")
plt.plot(datasets_as_xy[1][0], color="orange")
plt.plot(datasets_as_xy[2][0], color="green")

plt.xlabel('Days Since 1/22/2020', size=15)
plt.ylabel(' Increase of traffic routing requests in %, baseline at 100', size = 20)
plt.xticks(size=10, rotation=90, ticks=[1,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
plt.yticks(size=30)
plt.grid()
plt.legend(["Worldwide traffic requests"], loc=9)
plt.show()


exit(1)


plt.figure(figsize=(16, 10))
plt.plot(adjusted_dates, total_deaths)
plt.plot(adjusted_dates, world_death_avg, linestyle='dashed', color='orange')
plt.title('# of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Worldwide Coronavirus Deaths', 'Moving Average {} Days'.format(window)], prop={'size': 20})
plt.xticks(size=20, ticks=1)
plt.grid()
plt.yticks(size=20)
plt.show()

# plt.figure(figsize=(16, 10))
# plt.plot(adjusted_dates, total_recovered)
# plt.plot(adjusted_dates, world_recovery_avg, linestyle='dashed', color='orange')
# plt.title('# of Coronavirus Recoveries Over Time', size=30)
# plt.xlabel('Days Since 1/22/2020', size=30)
# plt.ylabel('# of Cases', size=30)
# plt.legend(['Worldwide Coronavirus Recoveries', 'Moving Average {} Days'.format(window)], prop={'size': 20})
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.show()

# plt.figure(figsize=(16, 10))
# plt.plot(adjusted_dates, total_active)
# plt.plot(adjusted_dates, world_active_avg, linestyle='dashed', color='orange')
# plt.title('# of Coronavirus Active Cases Over Time', size=30)
# plt.xlabel('Days Since 1/22/2020', size=30)
# plt.ylabel('# of Active Cases', size=30)
# plt.legend(['Worldwide Coronavirus Active Cases', 'Moving Average {} Days'.format(window)], prop={'size': 20})
# plt.xticks(size=20)
# plt.yticks(size=20)
# plt.show()

