"""
Usage: py get-data.py [country]
If parameter isn't specified, switzerland will be used

Goals:
Plot containing 4 lines:
1. Traffic Data of country -> walking | DONE
2. Traffic Data of country -> driving | DONE
3. Traffic Data of country -> transit | DONE
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
default_mob_data_dict = {}

try:
    for i, key in enumerate(apple_mobility):
        default_mob_data_dict[key] = []
except KeyError:
    pass

index = []
# Get corresponding data rows for country:
for i, v in enumerate(apple_mobility.region):
    if v.upper() == country.upper():
        index.append(i)

datasets = []
# Add Values to data structure
try:
    for i in index:
        mob_data_dict = default_mob_data_dict.copy()
        for k, v in mob_data_dict.items():
            mob_data_dict[k] = apple_mobility.loc[i][k]
        datasets.append(mob_data_dict)

except KeyError:
    pass

# Values to x and y axis
x, y = [], []
i = -1

datasets_as_xy = []
prev_dataset = None
for dataset in datasets:
    i= 0
    temp = []
    temp2 = []
    for k, v in dataset.items():
        i+=1
        temp = []
        if i < 7:
            continue
        else:
            temp.append(k)
            temp.append(v)
        temp2.append(temp)
    datasets_as_xy.append(temp2)

adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 10))

ax = plt.gca()
ax2 = ax.twinx()

for z, value in enumerate(datasets_as_xy):
    data_x = []
    data_y = []
    for i in value:
        data_x.append(i[0])
        data_y.append(i[1])

    match z:
        case 0:
            ax.plot(data_x, data_y, color="#FE9402", label="Driving")
        case 1:
            ax.plot(data_x, data_y, color="#FE2D55", label="Transit")
        case 2:
            ax.plot(data_x, data_y, color="#AF51DE", label="Walking")
        case _:
            ax.plot(data_x, data_y, color="black")

plt.legend(["driving", "transit", "walking"])

ax2.set_ylim(ymax=1000000)
ax2.plot(data_x[2:], world_daily_increase, color="salmon", label="Daily Incidence")
ax2.plot(data_x[2:], world_daily_increase_avg, color="red", label="Daily Incidence, normalized")
ax.legend()
ax2.legend()
ax2.set_ylabel('COVID19 Cases Worldwide')

plt.xlabel('Days Since 1/22/2020', size=15)
ax.set_ylabel(' Increase of traffic routing requests in %, baseline at 100', size = 20)
plt.xticks(size=10, rotation=90, ticks=[1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
plt.yticks(size=10, rotation=90)
plt.grid()
#plt.legend([f"Traffic requests for {country}"], loc=9)
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

