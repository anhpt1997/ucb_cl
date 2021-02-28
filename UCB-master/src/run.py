import sys,os,argparse,time
import numpy as np
import torch
import utils
from datetime import datetime
import pickle
tstart=time.time()


def convert_array_tostring(array):
	result = ""
	for i in range(array.shape[0]):
		result+= ",".join([str(t) for t in array[i]])  +"\n"
	return result
# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',               default=0,              type=int,   help='(default=%(default)d)')
parser.add_argument('--device',             default='cuda:0',       type=str,   help='gpu id')
parser.add_argument('--experiment',         default='cifar',       type=str,
											choices=['mnist2','mnist5','pmnist','cifar','mixture'])
parser.add_argument('--approach',           default='ucb',            type=str,   help='acl')
parser.add_argument('--data_path',          default='../data/',            type=str,   help='gpu id')

# Training parameters
parser.add_argument('--output',             default='',                     type=str,   help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints/',    type=str,   help='')
parser.add_argument('--nepochs',            default=10,            type=int,   help='')
parser.add_argument('--sbatch',             default=64,             type=int,   help='')
parser.add_argument('--lr',                 default=0.01,           type=float, help='')  # use 0.3 for non-mnist datasets
parser.add_argument('--nlayers',            default=1,              type=int,   help='')
parser.add_argument('--nhid',               default=1200,           type=int, help='')
parser.add_argument('--parameter',          default='',             type=str,   help='')

# UCB HYPER-PARAMETERS
parser.add_argument('--samples',            default='1',           type=int,     help='Number of Monte Carlo samples')
parser.add_argument('--rho',                default='-3',           type=float,   help='Initial rho')
parser.add_argument('--sig1',               default='0.0',          type=float,   help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sig2',               default='6.0',          type=float,   help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi',                 default='0.25',         type=float,   help='weighting factor for prior')
parser.add_argument('--arch',               default='mlp',          type=str,     help='Bayesian Neural Network architecture')

parser.add_argument('--resume',          default='no',            type=str,   help='resume?')
parser.add_argument('--sti',             default=0,               type=int,   help='starting task?')

args=parser.parse_args()
utils.print_arguments(args)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
	# torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


print('Using device:', args.device)
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()

# Args -- Experiment
if args.experiment=='mnist2':
	from dataloaders import mnist2 as dataloader
elif args.experiment=='mnist5':
	from dataloaders import mnist5 as dataloader
elif args.experiment=='fashionmnist2':
	from dataloaders import fashionmnist2 as dataloader
elif args.experiment=='pmnist':
	from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
	from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
	from dataloaders import mixture as dataloader
elif args.experiment=='fmnist_5_singlehead':
	from dataloaders import fmnist_5_singlehead as dataloader
elif args.experiment=='mnist2_singlehead':
	from dataloaders import mnist2_singlehead as dataloader
elif args.experiment=='mnist2_singlehead_ver1':
	from dataloaders import mnist2_singlehead_ver1 as dataloader
# elif args.experiment =='mnist_5_singlehead':
#     from dataloaders import mnist_5_singlehead as dataloader
elif args.experiment =='notmnist_singlehead':
	from dataloaders import notmnist_singlehead as dataloader

# Args -- Approach
if args.approach=='ucb':
	from approaches import ucb as approach
elif args.approach == 'ucb_singlehead':
	from approaches import ucb_singlehead as approach
elif args.approach == 'ucb_singlehead_2':
	from approaches import ucb_singlehead_2 as approach	
elif args.approach == 'ucb_singlehead_mixture':
	from approaches import ucb_singlehead_mixture as approach
elif args.approach == 'ucb_single_ver1':
	from approaches import ucb_single_ver1 as approach
# Args -- Network

if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment == 'mnist5' or args.experiment =='fashionmnist2' :
	from networks import mlp_ucb as network

# elif args.experiment =='fmnist5_singlehead':
#     from networks import mlp_ucb_mixture as network
elif args.experiment =='mnist2_singlehead' or args.experiment == 'mnist2_singlehead_ver1' or args.experiment =='mnist_5_singlehead' or args.experiment =='notmnist_singlehead':
	from networks import mlp_ucb_singlehead as network
else:
	from networks import resnet_ucb as network


########################################################################################################################
print()
print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(data_path=args.data_path, seed=args.seed)
# data,taskcla,inputsize=pickle.load(open('mnist5_singlehead','rb'))
pickle.dump(dataloader.get(data_path=args.data_path, seed=args.seed) , open("data_cifar", "wb"))
print('Input size =',inputsize,'\nTask info =',taskcla)
args.num_tasks=len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla
#
# t = data[0]
# t_train = t['train']['x']
# print(type(t_train), t_train.shape)
# t_y= t['train']['y']
# Inits
print('Inits...')
model=network.Net(args).to(args.device)


print('-'*100)
appr=approach.Appr(model,args=args)
print('-'*100)

# args.output=os.path.join(args.results_path, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
print('-'*100)

if args.resume == 'yes':
  checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.to(device=args.device)
else:
  args.sti = 0


# Loop tasks
acc=np.full(shape = (len(taskcla),len(taskcla)) , fill_value=np.nan,dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
list_valid_x , list_valid_y  = [] , []
for t,ncla in taskcla[args.sti:]:
  print('*'*100)
  print('Task {:2d} ({:s})'.format(t,data[t]['name']))
  print('*'*100)

  if args.approach == 'joint':
	   # Get data. We do not put it to GPU
	   if t==0:
		   xtrain=data[t]['train']['x']
		   ytrain=data[t]['train']['y']
		   xvalid=data[t]['valid']['x']
		   yvalid=data[t]['valid']['y']
		   task_t=t*torch.ones(xtrain.size(0)).int()
		   task_v=t*torch.ones(xvalid.size(0)).int()
		   task=[task_t,task_v]
	   else:
		   xtrain=torch.cat((xtrain,data[t]['train']['x']))
		   ytrain=torch.cat((ytrain,data[t]['train']['y']))
		   xvalid=torch.cat((xvalid,data[t]['valid']['x']))
		   yvalid=torch.cat((yvalid,data[t]['valid']['y']))
		   task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
		   task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
		   task=[task_t,task_v]
  else:
	   # Get data
	   xtrain=data[t]['train']['x'].to(args.device)
	   ytrain=data[t]['train']['y'].to(args.device)
	   xvalid=data[t]['valid']['x'].to(args.device)
	   yvalid=data[t]['valid']['y'].to(args.device)
	   list_valid_x.append(xvalid)
	   list_valid_y.append(yvalid)
	   task=t

  # Train
  appr.train(task,xtrain,ytrain,list_valid_x,list_valid_y)
  print('-'*100)

  # Test
  for u in range(t+1):
	   xtest=data[u]['test']['x'].to(args.device)
	   ytest=data[u]['test']['y'].to(args.device)
	#    test_acc=appr.test_mixture(u , xtest , ytest , num_sample= 200 , bs_test=5000)
	   test_loss, test_acc=appr.eval(u , xtest , ytest)

	   print('>>> Test on task {:2d} - {:15s}: , acc={:5.3f}% <<<'.format(u,data[u]['name'],100*test_acc))
	   acc[t,u]=test_acc
	#    lss[t,u]=test_loss

  # Save
  print('Save at '+args.checkpoint)
  print("acc ",acc)
#   np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}_{}_{}_{}_{}.txt'.format(args.experiment,args.approach,args.seed,args.samples,args.sig1,args.sig2,args.pi)),acc,'%.5f')
  file_name = os.path.join(args.checkpoint,'{}_{}_{}_{}_{}_{}_{}.csv'.format(args.experiment,args.approach,args.seed,args.samples,args.sig1,args.sig2,args.pi))
  with open(file_name , "w") as f:
	  f.write(convert_array_tostring(acc))

utils.print_log_acc_bwt(args, acc, lss)
print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

