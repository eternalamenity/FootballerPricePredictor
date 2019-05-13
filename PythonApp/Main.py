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















#Using new dataset



def makeOnlyOnePosition(data):
    for index, row in data.iterrows():
        positions = row['Preferred Positions']
        splitted = positions.split()
        row['Preferred Positions'] = splitted[0]
    return data

def makeOnlyOneRatingAtPosition(data):
    for index, row in data.iterrows():
        positions = str(row['CAM'])
        splitted = positions.split('+')
        row['CAM'] = splitted[0]

        positions = str(row['CB'])
        splitted = positions.split('+')
        row['CB'] = splitted[0]

        positions = str(row['CDM'])
        splitted = positions.split('+')
        row['CDM'] = splitted[0]

        positions = str(row['CF'])
        splitted = positions.split('+')
        row['CF'] = splitted[0]

        positions = str(row['CM'])
        splitted = positions.split('+')
        row['CM'] = splitted[0]

        positions = str(row['LAM'])
        splitted = positions.split('+')
        row['LAM'] = splitted[0]

        positions = str(row['LB'])
        splitted = positions.split('+')
        row['LB'] = splitted[0]

        positions = str(row['LCB'])
        splitted = positions.split('+')
        row['LCB'] = splitted[0]

        positions = str(row['LCM'])
        splitted = positions.split('+')
        row['LCM'] = splitted[0]

        positions = str(row['LDM'])
        splitted = positions.split('+')
        row['LDM'] = splitted[0]

        positions = str(row['LF'])
        splitted = positions.split('+')
        row['LF'] = splitted[0]

        positions = str(row['LM'])
        splitted = positions.split('+')
        row['LM'] = splitted[0]

        positions = str(row['LS'])
        splitted = positions.split('+')
        row['LS'] = splitted[0]

        positions = str(row['LW'])
        splitted = positions.split('+')
        row['LW'] = splitted[0]

        positions = str(row['LWB'])
        splitted = positions.split('+')
        row['LWB'] = splitted[0]

        positions = str(row['RAM'])
        splitted = positions.split('+')
        row['RAM'] = splitted[0]

        positions = str(row['RB'])
        splitted = positions.split('+')
        row['RB'] = splitted[0]

        positions = str(row['RCB'])
        splitted = positions.split('+')
        row['RCB'] = splitted[0]

        positions = str(row['RCM'])
        splitted = positions.split('+')
        row['RCM'] = splitted[0]

        positions = str(row['RDM'])
        splitted = positions.split('+')
        row['RDM'] = splitted[0]

        positions = str(row['RF'])
        splitted = positions.split('+')
        row['RF'] = splitted[0]

        positions = str(row['RM'])
        splitted = positions.split('+')
        row['RM'] = splitted[0]

        positions = str(row['RS'])
        splitted = positions.split('+')
        row['RS'] = splitted[0]

        positions = str(row['RW'])
        splitted = positions.split('+')
        row['RW'] = splitted[0]

        positions = str(row['RWB'])
        splitted = positions.split('+')
        row['RWB'] = splitted[0]

        positions = str(row['ST'])
        splitted = positions.split('+')
        row['ST'] = splitted[0]

    return data


#Individual changes for training dataset
training_dataset = pd.read_csv('externals/TrainingDataset/CompleteDataset.csv')
#training_dataset = makeOnlyOnePosition(training_dataset) - this function does not work because it does not save changes made on dataframe
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
#testing_dataset = makeOnlyOneRatingAtPosition(testing_dataset) - this function does not work because it does not save changes made on dataframe

#Changes for both datasets
testing_dataset = testing_dataset.drop(columns="ID")
testing_dataset = testing_dataset.drop(columns="Name")
testing_dataset = testing_dataset.drop(columns="Photo")
testing_dataset = testing_dataset.drop(columns="Nationality")
testing_dataset = testing_dataset.drop(columns="Flag")
testing_dataset = testing_dataset.drop(columns="Club Logo")

training_dataset = training_dataset.drop(columns="ID")
training_dataset = training_dataset.drop(columns="Name")
training_dataset = training_dataset.drop(columns="Photo")
training_dataset = training_dataset.drop(columns="Nationality")
training_dataset = training_dataset.drop(columns="Flag")
training_dataset = training_dataset.drop(columns="Club Logo")





#testing_dataset.to_csv(r'externals/before_changes.csv')
training_dataset = training_dataset[['Unnamed: 0', 'Age', 'Overall', 'Potential', 'Club', 'Value', 'Wage', 'Special', 'Position', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
#testing_dataset.to_csv(r'externals/after_changes.csv')

#Make list of colums
training_cols = list(training_dataset.columns.values)
testing_cols = list(testing_dataset.columns.values)

testing_dataset.to_csv(r'externals/ready_testing.csv')
training_dataset.to_csv(r'externals/ready_training.csv')

#print(testing_dataset.dtypes)
print(training_cols)
print(testing_cols)

print(training_dataset.shape)
print(testing_dataset.shape)



