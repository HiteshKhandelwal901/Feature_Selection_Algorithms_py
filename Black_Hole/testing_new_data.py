import pandas as pd
"""
data = pd.read_csv('birds.csv')

print("data = \n", data)
print("data.shape  = \n", data.shape)

Y = data.iloc[:, -19:]

X = data.iloc[:, 1:-19]
print("X = ", X)
print("Y = \n", Y)
"""

"""
data = pd.read_csv('CAL500.csv')
print("data = \n", data)
print("data.shape  = \n", data.shape)

Y = data.iloc[:, -174:]

X = data.iloc[:, 1:-174]
print("X = ", X)
print("Y = \n", Y)
"""


data = pd.read_csv('flags.csv')
print("data = \n", data)
print("data.shape  = \n", data.shape)

Y = data.iloc[:, -7:]

X = data.iloc[:, 1:-7]
print("X = ", X)
print("Y = \n", Y)
