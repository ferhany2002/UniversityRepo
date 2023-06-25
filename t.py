import operator

import numpy as np

a = [0,0,0,1]
b = [1,0,1,1]

temp = list(map(operator.add, a, b))
all_label = [1 if x > 0 else 0 for x in temp]

print(all_label)
print(np.count_nonzero(a == 1), " ", np.count_nonzero(b == 1), "  ", np.count_nonzero(all_label == 1))
