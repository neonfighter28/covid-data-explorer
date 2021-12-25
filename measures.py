import datetime
import pandas as pd

class Tags:
    def __init__(self):
        self.öv = "Öffentlicher Verkehr"
        self.flugverkehr = "Flugverkehr"
        self.maskenpflicht = "Maskenpflicht"
        self.zertifikat = "Zertifikat"
        self.quarantäne = "Quarantäne"

tags = Tags()

class Measure:
    _total_weight = 0
    # List of datetime objects from jan 1 to today
    _dates = pd.date_range(end = datetime.datetime.today(), start ="2020-01-01").to_pydatetime().tolist()
    print(_dates)


    def __init__(self, name=None, weight=0, start_date=None, end_date="today", tags=[]):
        """class for measure

        Args:
            name (str, optional): Name of the measure. Defaults to None.
            weight (int, optional): weight to be given the measure, classified on how often it affects a person a day. Defaults to 0.
            start_date (date the measure was introduced as dd.mm.yyyy, optional): [description]. Defaults to None.
            end_date (date the measure was lifted as dd.mm.yyyy, optional): [description]. Defaults to None.
        """

        self.name = name
        self.weight = weight
        self.start_end_dates = self.StartEndDatePair(start_date, end_date)
        self.tags = tags



    class StartEndDatePair():
        def __init__(self, start_dates, end_dates):
            self.all_start_dates = []
            self.all_end_dates = []
            if type(start_dates) == list:
                for index, start_date in enumerate(start_dates):
                    self.all_start_dates.append(self.strptime(start_date))
                    self.all_end_dates.append(self.strptime(end_dates[index]))
            else:
                self.all_start_dates.append(self.strptime(start_dates))
                self.all_end_dates.append(self.strptime(end_dates))

        def get_nth_pair(self, n) -> (datetime.date, datetime.date):
            return self.all_start_dates[n], self.all_end_dates[n]

        def occurrs_multiple_times(self):
            if len(all_start_dates) == 1: return True
            else: return False

        def strptime(self, dt):
            if dt == "today":
                return datetime.datetime.today()
            return datetime.datetime.strptime(dt, "%d.%m.%Y")

measures = [
    # Februar 2020
    Measure("Informationskampagne",         0.2,        "27.02.2020", end_date="today"),

    # März 2020
    Measure("Verbot 100+ Personen",         0.4,        "13.03.2020",),
    Measure("Schulschliessungen",           1,          "13.03.2020", end_date="11.05.2020"),
    Measure("Einreiseeinschränkungen",      0.1*1,      "13.03.2020",),
    Measure("10 Mia CHF Soforthilfe",       0,          "13.03.2020"),
    Measure("Schliessung Personenbezogene Betriebe (Coiffeure), etc",2,"16.03.2020", end_date="27.04.2020"),
    Measure("Schliessung Fachmärkte",       1,          "16.03.2020", end_date="27.04.2020"),
    Measure("Massnahmen Beerdigungen/OPs",  1,          "16.03.2020", end_date="27.04.2020"),
    Measure("Schliessungen anderer Läden",  2,          "16.03.2020", end_date="11.05.2020"),
    Measure("Schliessung Restaurants",      0.5,        "16.03.2020", end_date="11.05.2020"),
    Measure("Verbot 5+ Personen",           1,          "20.03.2020"),
    Measure("Maskenpflicht ÖV",             1,          "06.07.2020", tags=[tags.öv, tags.maskenpflicht]),
    Measure("Quarantäne für Einreisende",   1*4/365,    "15.08.2020"),
    Measure("Maskenpflicht Flugzeuge",      0.5/365,    "15.08.2020", tags=[tags.maskenpflicht, tags.flugverkehr]),
    Measure("Arbeitsnehmende im Homeoffice",0.8,        "19.08.2020"),
    Measure("Sitzpflicht in Gastronomie",   1/7,        "19.10.2020"),
    Measure("Verbot Ansammlungen 15+",      0.5,        "19.10.2020"),
    Measure("Einschr. Priv Veranstaltungen",1/7,        "19.10.2020"),
    Measure("Maskenpflicht Wartebereich",   0.7,        "19.10.2020", tags=[tags.maskenpflicht]),
    Measure("Maskenpflicht Innenbereich",   1,          "19.10.2020"),
    Measure("Verbot Diskothek/Tanz",        1/7,        "29.10.2020"),
    Measure("Sperrstunde 2300-0600",        2/7,        "29.10.2020"),
    Measure("Rest: Max Gruppen von 4",      1/7,        "29.10.2020"),
    Measure("Verbot 50+ Personen",          1/14,       "29.10.2020"),
    Measure("Verbot 10+ priv",              0.7,        "29.10.2020"),
    Measure("Schutzmassnahmen <15 personen",0.6,        "29.10.2020"),
    Measure("Chorverbot",                   1/50,       "29.10.2020"),
    Measure("Präsenzverbot Hochschulen",    0.5,        "02.11.2020"),
]