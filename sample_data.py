import pandas as pd 
import sys

data = pd.read_csv('./clean_train2.csv', sep=',', header=0, engine='python')
sample_data = data.sample(frac=1, replace=True)

sample_data.to_csv(sys.argv[1], sep=',', index=False)