import numpy as np 
import sys,os
import pandas as pd
# seed = sys.argv[1]
# samples = sys.argv[2]
# file_name = "../checkpoints/mnist_5_singlehead_ucb_singlehead_mixture_numgauss3/mnist_5_singlehead_ucb_singlehead_mixture_"+seed+"_"+samples+"_0.0_6.0_0.25.txt"
# data = np.loadtxt(fname=file_name)
# print(data)
# data = np.where(data==0 , np.nan ,data)
# print(data)
# print(np.nanmean(data.T, axis = 0))

def create_path_result(samples , pi, data , num_gauss):
	path_result_root = "../result"
	if not os.path.exists(path_result_root):
		os.mkdir(path_result_root)
	path_data = path_result_root+"/"+data+"/"
	if not os.path.exists(path_data):
		os.mkdir(path_data)
	return path_data+"ucb_mixture_"+str(num_gauss) +"_sample_"+str(samples)+"pi_"+str(pi)

result = []
seeds = [ 0,  1, 2, 3, 4]
samples = sys.argv[1]
pi= sys.argv[2]
data_name = sys.argv[3]
num_gauss = sys.argv[4]
for seed in seeds:
	file_name = "../checkpoints/mnist_5_singlehead_ucb_singlehead_mixture/mnist_5_singlehead_ucb_singlehead_mixture_"+str(seed)+"_"+samples+"_0.0_6.0_"+pi+".csv"
	data = pd.read_csv(file_name , header=None)
	# data = np.loadtxt(file_name)
	data = np.where(data==0, np.nan ,data)
	data_mean = np.nanmean(data.T, axis = 0)
	print("da" , data_mean)
	result.append(data_mean)
result = np.array(result)
np.savetxt(fname=create_path_result(samples , pi,data_name,num_gauss) , X = result)



