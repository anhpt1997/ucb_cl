import os
import pandas as pd 
import numpy as np
arr = os.listdir()
result_name = "result_overall.csv"
overall_result = ""
for f in arr:
	if os.path.isfile(f)==True and f.split(".")[-1]=='csv':
		data = pd.read_csv(f , header=None , delimiter=",").values
		list_name = f.split(".")
		name = ".".join(list_name[:-1])
		data_mean = np.mean(data , axis = 0)
		overall_result += ",".join([name , ",".join([str(t) for t in data_mean])]) +"\n"
		print(overall_result)

with open(result_name , "w") as f:
	f.write(overall_result)