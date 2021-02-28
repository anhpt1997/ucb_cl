import torch
import numpy as np
from .FC import BayesianLinear


class BayesianMLP(torch.nn.Module):
	
	def __init__(self, args):
		super(BayesianMLP, self).__init__()

		ncha,size,_= args.inputsize
		self.taskcla= args.taskcla
		self.samples = args.samples
		self.device = args.device
		self.sbatch = args.sbatch
		self.init_lr = args.lr
		dim=args.nhid
		nlayers=args.nlayers

		self.fc1 = BayesianLinear(ncha*size*size, dim, args)

		#header
		self.classifier = torch.nn.ModuleList()
		n = self.taskcla[0][1]
		print("num classesifier ", n)
		self.classifier.append(BayesianLinear(dim, n, args))  # List chi gom 1 phan tu

	def prune(self,mask_modules):
		for module, mask in mask_modules.items():
			module.prune_module(mask)

	def forward(self, x, sample=False):
		x = x.view(x.size(0),-1)
		x = torch.nn.functional.relu(self.fc1(x, sample))
		return self.classifier[0](x, sample)


def Net(args):
	return BayesianMLP(args)

