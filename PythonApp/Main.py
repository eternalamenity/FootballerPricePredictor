import externals.WebScrapper.get_players_data as pld
import externals.WebScrapper.get_teams_urls as tu
import pandas as pd
import numpy as np


def cleanData(data):
    data = data[data.SpG != "Undefined"]
    for index, row in data.iterrows():
        positions = row['Playing Positions (Position-Apps-Goals-Assists-Rating)']
        splitted = positions.split('/')
        cleanedPositions = ''
        for i in splitted:
            cleanedPositions += i.split()[0]
            cleanedPositions += ' '

        cleanedPositions = cleanedPositions.split()
        cleanedPositions = sorted(cleanedPositions)
        cleanedPositions = ' '.join(cleanedPositions)

        row['Playing Positions (Position-Apps-Goals-Assists-Rating)'] = cleanedPositions
    data.to_csv(r'externals/WebScrapper/players_data/clean_data.csv')
    return data


#First things to do are those 2 lines below - so comment all the other stuff
#tu.get_teams_urls(0)
#pld.to_csv()

data = pd.read_csv('externals/WebScrapper/players_data/whoscored_data.csv')
data = cleanData(data)
print(data.head(20))
print(data.groupby('Playing Positions (Position-Apps-Goals-Assists-Rating)').size())
print(data.describe())