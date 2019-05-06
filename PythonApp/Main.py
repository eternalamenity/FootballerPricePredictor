import externals.WebScrapper.get_players_data as pld
import externals.WebScrapper.get_teams_urls as tu
import pandas as pd
import numpy as np


def cleanData(data):
    data = data[data.SpG != "Undefined"]
    data = data.replace(regex=r'^GK.*$', value='GK')
    data.to_csv(r'externals/WebScrapper/players_data/clean_data.csv')
    return data


#First things to do are those 2 lines below - so comment all the other stuff
#tu.get_teams_urls(0)
#pld.to_csv()

data = pd.read_csv('externals/WebScrapper/players_data/whoscored_data.csv')
data = cleanData(data)
print(data.head(20))
print(data.describe())