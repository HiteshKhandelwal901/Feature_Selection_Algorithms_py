import pandas as pd

def get_index_sum(X, cols):
    index_sum = []
    for col in cols:
        print("col = ", col)
        index_sum.append(X.columns.get_loc(col))
        print("index sum = ", index_sum)
    return sum(index_sum)


record = {'Math': [10, 20, 30, 40, 70],
          'Science': [40, 50, 60, 90, 50], 
          'English': [70, 80, 66, 75, 88],
          'English2': [70, 80, 66, 75, 88],
          'English3': [70, 80, 66, 75, 88]}

df = pd.DataFrame(record)
print(df)

print(get_index_sum(df, ['Math', 'English2']))