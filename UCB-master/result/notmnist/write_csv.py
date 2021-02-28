import numpy as np 



def write(array, file_name):
    result = ""
    for i in range(array.shape[0]):
        result+= ",".join([str(t) for t in array[i]]) +"\n"
    with open(file_name+".csv" , "w") as f:
        f.write(result)
import sys
file_name = sys.argv[1]
array = np.loadtxt(file_name)
write(array , file_name)