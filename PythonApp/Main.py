from pandas.plotting import scatter_matrix

import externals.WebScrapper.get_players_data as pld
import externals.WebScrapper.get_teams_urls as tu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


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

def deleteFakedRows(data):
    data = data[data.SpG != "SpG"]
    return data

def showPlt(data):
    data = data.astype(np.float)
    data.plot(kind='box', subplots=True, layout=(1, 6), sharex=False, sharey=False)
    plt.show()

def showHist(data):
    data = data.astype(np.float)
    data.hist()
    plt.show()

def showScatter(data):
    # Scatter plot matrix
    data = data.astype(np.float)
    scatter_matrix(data)
    plt.show()


#First things to do are those 2 lines below - so comment all the other stuff
#tu.get_teams_urls(0)
#pld.to_csv()

data = pd.read_csv('externals/WebScrapper/players_data/only_num_players.csv')
data = data[data.GoalsPerGame < 5]
data = data[data.AssistsPerGame < 5]
data = data[data.YelPerGame < 2]
data = data[data.RedPerGame < 1]


#data = cleanData(data)
#cols_to_use = ['Goals/90min', 'Assists/90min', 'Yel/90min', 'Red/90min', 'SpG', 'Rating']
#int_data = pd.read_csv('externals/WebScrapper/players_data/clean_data.csv', usecols= cols_to_use)
#data = deleteFakedRows(int_data)
#data.to_csv(r'externals/WebScrapper/players_data/only_num_players.csv')



print(data.head(100))
#print(data.iloc[1])

#Shows plot of data
#showPlt(data)

#Shows histogram of data
#showHist(data)


#Shows scatter histograms
#showScatter(data)

#print(modDfObj.describe())
#np_dfr = np.array(preprocessing(modDfObj))
#modDfObj.hist()
#print(data.groupby('Playing Positions (Position-Apps-Goals-Assists-Rating)').size())
#print(data.describe())
#data_standardized = preprocessing.scale(modDfObj)