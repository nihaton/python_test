import numpy as np

list_s = np.array(['1', '2', '3', '4', '5'])

list_t = list_s.astype(np.int) + 100

print (list_s)
print (list_s.dtype)
print (list_t)
print (list_t.dtype)



str = ['1', '2', '3', '4', '5']

list_i = np.array(str, dtype=np.int)

print (list_i)
print (list_i.dtype)
