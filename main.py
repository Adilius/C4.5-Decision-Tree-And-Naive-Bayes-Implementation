
from scipy.io.arff import loadarff  # To load .arff files
import pandas
from rich.console import Console  # Rich used for pretty console printing
from rich.table import Table

from sklearn.model_selection import train_test_split  # Naive bayes
from sklearn import preprocessing  # Naive bayes
from sklearn.naive_bayes import GaussianNB  # Naive bayes

import matplotlib.pyplot as plt #Plotting tool
import numpy as np

# Prints table from dataset variable
def print_table(df):

    # Initialize table
    table = Table(show_header=True)

    # Add titles to table
    titles = []
    # titles.append("number")
    table.add_column("Index")
    for col in df.columns:
        table.add_column(col)
        titles.append(col)

    # Add rows to table
    for i, _ in enumerate(df.iterrows()):
        rowList = []
        rowList.append(str(i+1))
        for title in titles:
            # print(df[title][i])
            rowList.append(df[title][i])
        table.add_row(*rowList)

    # Print table
    console.print(table)

# Run the sklearn Naive bayes on dataframe
def sklearn_naive_bayes(df):
    #Encode string into incremental value
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])

    #Split feature part X, and label part Y, from dataset, into train and test parts
    X = df[['age', 'menopause', 'tumor-size', 'inv-nodes','node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']]
    Y = df[['Class']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    model = GaussianNB()
    model.fit(x_train, y_train.values.ravel())

    print(model.score(x_test, y_test))  

# Initialize console and table object for pretty printing
console = Console()

# Load in the dataset
data = loadarff('breast-cancer.arff')
df = pandas.DataFrame(data[0])

# Encode from byte literals to utf-8
for column in df:
    df[column] = df[column].apply(lambda s: s.decode("utf-8"))

# Clean dataframe from rows with missing values
df = df.replace('?', np.nan)  # Replace ? with np.nan
df.dropna(inplace=True)  # Remove rows
df.reset_index(drop=True, inplace=True)  # Reset index

with console.status("[bold green]Generating data...") as status:
    print_table(df)
    sklearn_naive_bayes(df)