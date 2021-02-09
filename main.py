
from scipy.io.arff import loadarff  # To load .arff files
import pandas
from rich.console import Console  # Rich used for pretty console printing
from rich.table import Table

from sklearn.model_selection import train_test_split  # Naive bayes
from sklearn import preprocessing  # Naive bayes & C4.5 decision tree
from sklearn.naive_bayes import GaussianNB  # Naive bayes

from sklearn.tree import DecisionTreeClassifier #C4.5 decision tree
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt #Plotting tool
import numpy as np

#Load, encode, and clean dataframe from .arff file
def arff_to_dataframe(arff):

    # Load in the dataset
    data = loadarff(arff)
    df = pandas.DataFrame(data[0])

    # Encode from byte literals to utf-8
    for column in df:
        df[column] = df[column].apply(lambda s: s.decode("utf-8"))

    # Clean dataframe from rows with missing values
    df = df.replace('?', np.nan)  # Replace ? with np.nan
    df.dropna(inplace=True)  # Remove rows
    df.reset_index(drop=True, inplace=True)  # Reset index

    return df

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

#Run sklearn C4.5 tree decision algorithm on dataframe
def sklearn_c45_tree_decision(df):
    X, y = [df.iloc[:, list(range(0, len(df.columns) - 2))].values, df.iloc[:, len(df.columns) - 1].values]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


    sc = preprocessing.StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # fit method is used to train data
    classifier = DecisionTreeClassifier(criterion= 'entropy', random_state = 45)
    classifier.fit(X_train, y_train)


    # predict
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Testsize: ' + str(0.25 * 100) + '%')
    print('Random state: ' + str(45) + '%')
    print('Success rate: ' + str((cm[0][0] + cm[1][1]) / sum(map(sum, cm))))
    #print('Kappa success rate: ' + str(kappaSuccesRate(cm)))

# Initialize console and table object for pretty printing
console = Console()

with console.status("[bold green]Processing data...") as status:
    df = arff_to_dataframe("breast-cancer.arff")
    print_table(df)
    sklearn_naive_bayes(df)
    sklearn_c45_tree_decision(df)