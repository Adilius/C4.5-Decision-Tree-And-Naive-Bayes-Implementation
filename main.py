
from scipy.io.arff import loadarff #To load .arff files
import pandas
from rich.console import Console #Rich used for pretty console printing
from rich.table import Table
from sklearn import tree
import numpy as np

#Prints table from dataset variable
def print_table(df):

    #Initialize table
    table = Table(show_header=True)

    #Add titles to table
    titles = []
    #titles.append("number")
    table.add_column("Index")
    for col in df.columns:
        table.add_column(col)
        titles.append(col)

    #Add rows to table
    for i, _ in enumerate(df.iterrows()):
        rowList = []
        rowList.append(str(i+1))
        for title in titles:
            #print(df[title][i])
            rowList.append(df[title][i])
        table.add_row(*rowList)
        
    #Print table
    console.print(table)

#Initialize console and table object for pretty printing
console = Console()

#Load in the dataset
data = loadarff('breast-cancer.arff')
df = pandas.DataFrame(data[0])

#Encode from byte literals to utf-8
for i in range(0, len(df.columns)):
    title = list(df.columns)[i]
    df[title] = df[title].apply(lambda s: s.decode("utf-8"))

df = df.replace('?', np.nan)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print_table(df)