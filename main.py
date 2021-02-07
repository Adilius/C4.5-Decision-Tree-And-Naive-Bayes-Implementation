
from scipy.io.arff import loadarff #To load .arff files
import pandas
from rich.console import Console
from rich.table import Table

#Load in the dataset
data = loadarff('breast-cancer.arff')
df = pandas.DataFrame(data[0])

#Encode from byte literals to utf-8
for i in range(0, len(df.columns)):
    title = list(df.columns)[i]
    df[title] = df[title].apply(lambda s: s.decode("utf-8"))

#Prints head of dataset
#print(df.head())

console = Console()
table = Table(show_header=True)
titles = []
for col in df.columns:
    table.add_column(col)
    titles.append(col)

#print("titles:", titles)

for k, row in enumerate(df.iterrows()):
    rowList = []
    for title in titles:
        rowList.append(df[title][k])
    #print(rowList)
    table.add_row(*rowList)

console.print(table)
#print(df.to_string())
#print(data)
