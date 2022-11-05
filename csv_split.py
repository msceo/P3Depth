from sklearn.model_selection import train_test_split
import pandas as pd
import sys

data = pd.read_csv(sys.argv[1])
train,test = train_test_split(data, test_size=0.2, shuffle=True)
train.to_csv('ilsr_train.csv', index=False)
test.to_csv('ilsr_test.csv', index=False)