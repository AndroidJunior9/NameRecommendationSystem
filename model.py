import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

if __name__=="__main__":

    data = pd.read_csv("NationalNames.csv")

    # Create a new column with the length of each name
    data['NameLength'] = data['Name'].apply(len)

    # Create a new column with the first letter of each name
    data['FirstLetter'] = data['Name'].str[0]

    # Create a new column with the gender neutrality of each name
    data['GenderNeutral'] = data['Name'].isin(data[data['Gender'] != data['Gender'].shift()]['Name'])

    data.drop("Id",axis = 1,inplace=True)

    latest_year = data['Year'].max()
    popularity_change = data[data['Year'] >= latest_year - 10].groupby('Name')['Count'].sum() - data[data['Year'] < latest_year - 5].groupby('Name')['Count'].sum()

    # Create a new column with whether each name is trending or not
    data['Trending'] = data['Name'].apply(lambda x: popularity_change.get(x, 0) > 0)

    # Calculate the total count for each name
    name_counts = data.groupby('Name')['Count'].sum()

    # Define the thresholds for different levels of popularity
    very_high_threshold = name_counts.quantile(0.9)
    high_threshold = name_counts.quantile(0.7)
    medium_threshold = name_counts.quantile(0.5)
    low_threshold = name_counts.quantile(0.3)

    # Create a new column with the popularity of each name
    data['Popularity'] = pd.cut(data['Name'].map(name_counts), [0, low_threshold, medium_threshold, high_threshold, very_high_threshold, float('inf')], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])



    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['FirstLetter'] = le.fit_transform(data['FirstLetter'])
    popularity_map = {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    data['Popularity'] = data['Popularity'].map(popularity_map)
    X = data[['Gender','Trending','NameLength','FirstLetter','GenderNeutral','Popularity']]
    Y = data['Name']



    # Train a decision tree regressor on the data
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X.values, Y.values)

    #1.Gender
    #2.Trending
    #3.NameLength
    #4.First Letter
    #5.Gender Neutral
    #6.Popularity

    # Save the model as a pickle file
    with open('model.pkl','wb') as file:
        pickle.dump(model,file)