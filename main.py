
from scipy.io.arff import loadarff #To load .arff files
import pandas

#Load in the dataset
data = loadarff('breast-cancer.arff')
df = pandas.DataFrame(data[0])

print(df.head())
#print(data)
