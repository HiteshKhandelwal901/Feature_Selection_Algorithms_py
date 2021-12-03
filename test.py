from typing import DefaultDict

a = ['a', 'b', 'c']
b = [1,2,3]
c = zip(a,b)

s = sorted(c)
print(s)
remove = s[-1]
print("remove = ", remove)
feat = remove[0]
print("feat = ", feat)