from pandas.plotting import scatter_matrix

import externals.WebScrapper.get_players_data as pld
import externals.WebScrapper.get_teams_urls as tu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC


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
    #data = data.astype(np.float)
    scatter_matrix(data)
    plt.show()

#First things to do are those 2 lines below - so comment all the other stuff
#tu.get_teams_urls(0)
#pld.to_csv()

#data = pd.read_csv('externals/WebScrapper/players_data/only_num_players.csv')
#data = data[data.GoalsPerGame < 5]
#data = data[data.AssistsPerGame < 5]
#data = data[data.YelPerGame < 2]
#data = data[data.RedPerGame < 1]

#data = cleanData(data)
#cols_to_use = ['Goals/90min', 'Assists/90min', 'Yel/90min', 'Red/90min', 'SpG', 'Rating']
#int_data = pd.read_csv('externals/WebScrapper/players_data/clean_data.csv', usecols= cols_to_use)
#data = deleteFakedRows(int_data)
#data.to_csv(r'externals/WebScrapper/players_data/only_num_players.csv')



#print(data.head(100))
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















##########Using new dataset

#Individual changes for training dataset
training_dataset = pd.read_csv('externals/TrainingDataset/CompleteDataset.csv')
training_dataset.rename(columns={'Preferred Positions': 'Position'}, inplace=True)
training_dataset.rename(columns={'Heading accuracy': 'HeadingAccuracy'}, inplace=True)
training_dataset.rename(columns={'Short passing': 'ShortPassing'}, inplace=True)
training_dataset.rename(columns={'Free kick accuracy': 'FKAccuracy'}, inplace=True)
training_dataset.rename(columns={'Long passing': 'LongPassing'}, inplace=True)
training_dataset.rename(columns={'Ball control': 'BallControl'}, inplace=True)
training_dataset.rename(columns={'Sprint speed': 'SprintSpeed'}, inplace=True)
training_dataset.rename(columns={'Shot power': 'ShotPower'}, inplace=True)
training_dataset.rename(columns={'Long shots': 'LongShots'}, inplace=True)
training_dataset.rename(columns={'Sliding tackle': 'SlidingTackle'}, inplace=True)
training_dataset.rename(columns={'Standing tackle': 'StandingTackle'}, inplace=True)
training_dataset.rename(columns={'GK diving': 'GKDiving'}, inplace=True)
training_dataset.rename(columns={'GK handling': 'GKHandling'}, inplace=True)
training_dataset.rename(columns={'GK kicking': 'GKKicking'}, inplace=True)
training_dataset.rename(columns={'GK positioning': 'GKPositioning'}, inplace=True)
training_dataset.rename(columns={'GK reflexes': 'GKReflexes'}, inplace=True)
f = lambda x: (x["Position"].split()[0])
training_dataset["Position"] = training_dataset.apply(f, axis=1)
training_dataset["Crossing"] = training_dataset["Crossing"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Finishing"] = training_dataset["Finishing"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["HeadingAccuracy"] = training_dataset["HeadingAccuracy"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["ShortPassing"] = training_dataset["ShortPassing"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Volleys"] = training_dataset["Volleys"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Dribbling"] = training_dataset["Dribbling"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Curve"] = training_dataset["Curve"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["FKAccuracy"] = training_dataset["FKAccuracy"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["LongPassing"] = training_dataset["LongPassing"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["BallControl"] = training_dataset["BallControl"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Acceleration"] = training_dataset["Acceleration"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["SprintSpeed"] = training_dataset["SprintSpeed"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Agility"] = training_dataset["Agility"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Reactions"] = training_dataset["Reactions"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Balance"] = training_dataset["Balance"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["ShotPower"] = training_dataset["ShotPower"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Jumping"] = training_dataset["Jumping"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Stamina"] = training_dataset["Stamina"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Strength"] = training_dataset["Strength"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["LongShots"] = training_dataset["LongShots"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Aggression"] = training_dataset["Aggression"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Interceptions"] = training_dataset["Interceptions"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Positioning"] = training_dataset["Positioning"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Vision"] = training_dataset["Vision"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Penalties"] = training_dataset["Penalties"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Composure"] = training_dataset["Composure"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Marking"] = training_dataset["Marking"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["StandingTackle"] = training_dataset["StandingTackle"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["SlidingTackle"] = training_dataset["SlidingTackle"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["GKDiving"] = training_dataset["GKDiving"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["GKHandling"] = training_dataset["GKHandling"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["GKKicking"] = training_dataset["GKKicking"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["GKPositioning"] = training_dataset["GKPositioning"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["GKReflexes"] = training_dataset["GKReflexes"].replace({'(\-[0-9])|(\+[0-9])':''}, regex = True)
training_dataset["Value"] = training_dataset["Value"].replace({'\.[0-9]':''}, regex = True)
training_dataset["Value"] = training_dataset["Value"].replace({'K':'000'}, regex = True)
training_dataset["Value"] = training_dataset["Value"].replace({'M':'000000'}, regex = True)
training_dataset["Value"] = training_dataset["Value"].replace({'\€':''}, regex = True)
#Get only Players Market Values
players_values = training_dataset["Value"]

#Individual changes for testing dataset
testing_dataset = pd.read_csv('externals/data.csv')
testing_dataset = testing_dataset.drop(columns="Release Clause")
testing_dataset = testing_dataset.drop(columns="Body Type")
testing_dataset = testing_dataset.drop(columns="Skill Moves")
testing_dataset = testing_dataset.drop(columns="Weak Foot")
testing_dataset = testing_dataset.drop(columns="International Reputation")
testing_dataset = testing_dataset.drop(columns="Preferred Foot")
testing_dataset = testing_dataset.drop(columns="Work Rate")
testing_dataset = testing_dataset.drop(columns="Real Face")
testing_dataset = testing_dataset.drop(columns="Jersey Number")
testing_dataset = testing_dataset.drop(columns="Joined")
testing_dataset = testing_dataset.drop(columns="Loaned From")
testing_dataset = testing_dataset.drop(columns="Contract Valid Until")
testing_dataset = testing_dataset.drop(columns="Height")
testing_dataset = testing_dataset.drop(columns="Weight")
testing_dataset['LS'] = testing_dataset['LS'].map(lambda x: str(x)[:-2])
testing_dataset['ST'] = testing_dataset['ST'].map(lambda x: str(x)[:-2])
testing_dataset['RS'] = testing_dataset['RS'].map(lambda x: str(x)[:-2])
testing_dataset['LW'] = testing_dataset['LW'].map(lambda x: str(x)[:-2])
testing_dataset['LF'] = testing_dataset['LF'].map(lambda x: str(x)[:-2])
testing_dataset['CF'] = testing_dataset['CF'].map(lambda x: str(x)[:-2])
testing_dataset['RF'] = testing_dataset['RF'].map(lambda x: str(x)[:-2])
testing_dataset['RW'] = testing_dataset['RW'].map(lambda x: str(x)[:-2])
testing_dataset['LAM'] = testing_dataset['LAM'].map(lambda x: str(x)[:-2])
testing_dataset['CAM'] = testing_dataset['CAM'].map(lambda x: str(x)[:-2])
testing_dataset['RAM'] = testing_dataset['RAM'].map(lambda x: str(x)[:-2])
testing_dataset['LM'] = testing_dataset['LM'].map(lambda x: str(x)[:-2])
testing_dataset['LCM'] = testing_dataset['LCM'].map(lambda x: str(x)[:-2])
testing_dataset['CM'] = testing_dataset['CM'].map(lambda x: str(x)[:-2])
testing_dataset['RCM'] = testing_dataset['RCM'].map(lambda x: str(x)[:-2])
testing_dataset['RM'] = testing_dataset['RM'].map(lambda x: str(x)[:-2])
testing_dataset['LWB'] = testing_dataset['LWB'].map(lambda x: str(x)[:-2])
testing_dataset['LDM'] = testing_dataset['LDM'].map(lambda x: str(x)[:-2])
testing_dataset['CDM'] = testing_dataset['CDM'].map(lambda x: str(x)[:-2])
testing_dataset['RDM'] = testing_dataset['RDM'].map(lambda x: str(x)[:-2])
testing_dataset['RWB'] = testing_dataset['RWB'].map(lambda x: str(x)[:-2])
testing_dataset['LB'] = testing_dataset['LB'].map(lambda x: str(x)[:-2])
testing_dataset['LCB'] = testing_dataset['LCB'].map(lambda x: str(x)[:-2])
testing_dataset['CB'] = testing_dataset['CB'].map(lambda x: str(x)[:-2])
testing_dataset['RB'] = testing_dataset['RB'].map(lambda x: str(x)[:-2])
testing_dataset['RCB'] = testing_dataset['RCB'].map(lambda x: str(x)[:-2])




#Changes for both datasets
testing_dataset = testing_dataset.drop(columns="ID")
testing_dataset = testing_dataset.drop(columns="Name")
testing_dataset = testing_dataset.drop(columns="Photo")
testing_dataset = testing_dataset.drop(columns="Nationality")
testing_dataset = testing_dataset.drop(columns="Flag")
testing_dataset = testing_dataset.drop(columns="Club Logo")
testing_dataset = testing_dataset.drop(columns="Club")
testing_dataset = testing_dataset.drop(columns="Position")
testing_dataset = testing_dataset.drop(columns="Value")
testing_dataset = testing_dataset.drop(columns="Wage")
testing_dataset = testing_dataset.drop(columns="Unnamed: 0")

training_dataset = training_dataset.drop(columns="ID")
training_dataset = training_dataset.drop(columns="Name")
training_dataset = training_dataset.drop(columns="Photo")
training_dataset = training_dataset.drop(columns="Nationality")
training_dataset = training_dataset.drop(columns="Flag")
training_dataset = training_dataset.drop(columns="Club Logo")
training_dataset = training_dataset.drop(columns="Club")
training_dataset = training_dataset.drop(columns="Position")
training_dataset = training_dataset.drop(columns="Value")
training_dataset = training_dataset.drop(columns="Wage")
training_dataset = training_dataset.drop(columns="Unnamed: 0")

#For testing purposes delete ratings at positions
training_dataset = training_dataset.drop(columns="LS")
training_dataset = training_dataset.drop(columns="ST")
training_dataset = training_dataset.drop(columns="RS")
training_dataset = training_dataset.drop(columns="LW")
training_dataset = training_dataset.drop(columns="CF")
training_dataset = training_dataset.drop(columns="RF")
training_dataset = training_dataset.drop(columns="RW")
training_dataset = training_dataset.drop(columns="LAM")
training_dataset = training_dataset.drop(columns="CAM")
training_dataset = training_dataset.drop(columns="RAM")
training_dataset = training_dataset.drop(columns="LM")
training_dataset = training_dataset.drop(columns="LCM")
training_dataset = training_dataset.drop(columns="CM")
training_dataset = training_dataset.drop(columns="RCM")
training_dataset = training_dataset.drop(columns="RM")
training_dataset = training_dataset.drop(columns="LWB")
training_dataset = training_dataset.drop(columns="LDM")
training_dataset = training_dataset.drop(columns="CDM")
training_dataset = training_dataset.drop(columns="RDM")
training_dataset = training_dataset.drop(columns="RWB")
training_dataset = training_dataset.drop(columns="LB")
training_dataset = training_dataset.drop(columns="LCB")
training_dataset = training_dataset.drop(columns="CB")
training_dataset = training_dataset.drop(columns="RB")
training_dataset = training_dataset.drop(columns="RCB")
training_dataset = training_dataset[['Age', 'Overall', 'Potential', 'Special', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]

#training_dataset = training_dataset[['Age', 'Overall', 'Potential', 'Special', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]



#Make list of colums
training_cols = list(training_dataset.columns.values)
testing_cols = list(testing_dataset.columns.values)

testing_dataset.to_csv(r'externals/ready_testing.csv')
training_dataset.to_csv(r'externals/ready_training.csv')

#showScatter(training_dataset)

#print(testing_dataset.dtypes)
print(training_cols)
print(testing_cols)

print(training_dataset.shape)
print(testing_dataset.shape)

print(players_values)
X = np.array(training_dataset.values)
Y = np.array(players_values.values)
clf = SVC(gamma='auto')
clf.fit(X, Y)

print(clf.predict([[18, 46, 64, 1031, 19, 20, 48, 31, 19, 23, 17, 17, 24, 32, 48, 49, 49, 40, 47, 21, 60, 55, 67, 17, 52, 38, 20, 22, 21, 33, 38, 44, 43, 15, 8, 10, 10, 7]]))