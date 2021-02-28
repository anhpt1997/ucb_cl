import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
from .utils import BayesianSGD
from torch.autograd import Variable

class Appr(object):

	def __init__(self,model,args,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=1000):
		self.model=model
		self.device = args.device
		self.lr_min=lr_min
		self.lr_factor=lr_factor
		self.lr_patience=lr_patience
		self.clipgrad=clipgrad

		self.init_lr=args.lr
		self.sbatch=args.sbatch
		self.nepochs=args.nepochs

		self.arch=args.arch
		self.samples=args.samples
		self.lambda_=1.

		self.output=args.output
		self.checkpoint = args.checkpoint
		self.experiment=args.experiment
		self.num_tasks=args.num_tasks

		self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
		self.modules_names_without_cls = self.find_modules_names(with_classifier=False)

		print("module name with cls " , self.modules_names_with_cls)
		print("module name not with cls ", self.modules_names_without_cls)
		print(" PRINT PARAMETER")
		for name, param in model.named_parameters():
			if param.requires_grad:
				print (name)

	def train(self,t,xtrain,ytrain,list_xvalid,list_yvalid):
	

		# Update the next learning rate for each parameter based on their uncertainty
		params_dict = self.update_lr(t)
		self.optimizer = BayesianSGD(params=params_dict)

		best_loss=np.inf

		# best_model=copy.deepcopy(self.model)
		best_model = copy.deepcopy(self.model.state_dict())
		lr = self.init_lr
		patience = self.lr_patience

		# Loop epochs
		try:
			for e in range(self.nepochs):

				valid_loss,valid_acc=self.eval_list_valid(list_xvalid , list_yvalid)
				print( e , 'valid loss ' ,  valid_loss, 'sum valid loss ', valid_loss , 'valid acc ' , valid_acc)

				# for name, param in self.model.named_parameters():
				#     if name == 'fc1.bias_mu':
				#         print (e , "bias_mu " , param.data)
				if math.isnan(valid_loss[-1]):
					print("saved best model and quit because loss became nan")
					break
				# Adapt lr
				if valid_loss[-1]<best_loss:
					best_loss=valid_loss[-1]
					print("best loss " , best_loss)
					best_model=copy.deepcopy(self.model.state_dict())
					patience=self.lr_patience
					print(' *',end='')
				else:
					patience-=1
					if patience<=0:
						lr/=self.lr_factor
						print(' lr={:.1e}'.format(lr),end='')
						if lr<self.lr_min:
							print()
							break
						patience=self.lr_patience

						params_dict = self.update_lr(t, adaptive_lr=True, lr=lr)
						self.optimizer=BayesianSGD(params=params_dict)
				s_t = time.time()
				self.train_epoch(t,xtrain,ytrain)
				e_time = time.time()
				print("total time train ", e_time- s_t)
				print()
		except KeyboardInterrupt:
			print()

		# Restore best
		self.model.load_state_dict(copy.deepcopy(best_model))
		self.save_model(t)

	def eval_list_valid(self, list_xvalid , list_yvalid):
		result_loss, result_acc = [] , []
		for i in range(len(list_xvalid)):
			loss ,acc =self.eval(i , list_xvalid[i] , list_yvalid[i])
			result_loss.append(loss)
			result_acc.append(acc)
		return result_loss , result_acc
	
	def test(self ,t , x , y , num_sample = 1):
		total_acc=0
		total_num=0
		self.model.eval()

		r=np.arange(x.size(0))
		r=torch.as_tensor(r, device=self.device, dtype=torch.int64)
		with torch.no_grad():
			num_batches = len(x)//self.sbatch
			# Loop batches
			for i in range(0,len(r),self.sbatch):
				if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
				else: b=r[i:]
				images, targets = x[b].to(self.device), y[b].to(self.device)

				list_output  = [] 
				# Forward
				for i in range(num_sample):
					output = torch.nn.functional.log_softmax(self.model(images,sample=False), dim = 1).cpu().numpy()
					list_output.append(output)

				output_mean = np.mean( np.array(list_output), axis = 0)
				pred=np.argmax(output_mean, axis = 1)

				total_acc += np.sum(pred == targets.cpu().numpy())
				total_num += len(b) 
		return  total_acc/total_num       

	def update_lr(self,t, lr=None, adaptive_lr=False):
		print("update lr " , t)
		params_dict = []
		if t==0:
			params_dict.append({'params': self.model.parameters(), 'lr': self.init_lr})
		else:
			for name in self.modules_names_without_cls:
				n = name.split('.')
				print("n ",n)
				if len(n) >= 1:
					if len(n) == 1:
						m = self.model._modules[n[0]]
					elif len(n) == 2:
						m = self.model._modules[n[0]]._modules[n[1]]
					elif len(n) == 3:
						m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
					elif len(n) == 4:
						m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
					else:
						print (name)

					if adaptive_lr is True:
						params_dict.append({'params': m.weight_rho, 'lr': lr})
						_params_dict.append({'params': m.bias_rho, 'lr': lr})

					else:
						w_unc = torch.log1p(torch.exp(m.weight_rho.data))
						b_unc = torch.log1p(torch.exp(m.bias_rho.data))

						params_dict.append({'params': m.weight_mu, 'lr': self.init_lr})
						params_dict.append({'params': m.bias_mu, 'lr': self.init_lr})
						params_dict.append({'params': m.weight_rho, 'lr':self.init_lr})
						params_dict.append({'params': m.bias_rho, 'lr':self.init_lr})
						print("name " ,name)

			print(" moduls with cls")
			for name in self.modules_names_with_cls:
				n = name.split('.')
				print("n ",n)
				if len(n) > 1:
					if len(n) == 1:
						m = self.model._modules[n[0]]
					elif len(n) == 2:
						m = self.model._modules[n[0]]._modules[n[1]]
					elif len(n) == 3:
						m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
					elif len(n) == 4:
						m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
					else:
						print (name)

					if adaptive_lr is True:
						params_dict.append({'params': m.weight_rho, 'lr': lr})
						params_dict.append({'params': m.bias_rho, 'lr': lr})

					else:
						w_unc = torch.log1p(torch.exp(m.weight_rho.data))
						b_unc = torch.log1p(torch.exp(m.bias_rho.data))

						params_dict.append({'params': m.weight_mu, 'lr': self.init_lr})
						params_dict.append({'params': m.bias_mu, 'lr': self.init_lr})
						params_dict.append({'params': m.weight_rho, 'lr':self.init_lr})
						params_dict.append({'params': m.bias_rho, 'lr':self.init_lr})

						print("name " ,name)
		return params_dict

	def find_modules_names(self, with_classifier=False):
		modules_names = []
		for name, p in self.model.named_parameters():
			if with_classifier is False:
				if not name.startswith('classifier'):
					n = name.split('.')[:-1]
					modules_names.append('.'.join(n))
			else:
				n = name.split('.')[:-1]
				modules_names.append('.'.join(n))

		modules_names = set(modules_names)

		return modules_names

	def logs(self,t):

		lp, lvp = 0.0, 0.0
		for name in self.modules_names_without_cls:
			n = name.split('.')
			if len(n) == 1:
				m = self.model._modules[n[0]]
			elif len(n) == 3:
				m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
			elif len(n) == 4:
				m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

			lp += m.log_prior
			lvp += m.log_variational_posterior

		#t la index cua header
		lp += self.model.classifier[t].log_prior
		lvp += self.model.classifier[t].log_variational_posterior

		return lp, lvp

	def train_epoch(self,t,x,y):

		self.model.train()

		r=np.arange(x.size(0))
		np.random.shuffle(r)
		r=torch.LongTensor(r).to(self.device)

		num_batches = len(x)//self.sbatch
		j=0
		# Loop batches
		for i in range(0,len(r),self.sbatch):

			if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
			else: b=r[i:]
			images, targets = x[b].to(self.device), y[b].to(self.device)

			# Forward
			loss=self.elbo_loss(images,targets,t,num_batches,sample=True).to(self.device)
			# print("loss ", loss)
			# Backward
			self.model.cuda()
			self.optimizer.zero_grad()
			loss.backward(retain_graph=True)
			self.model.cuda()

			# Update parameters
			self.optimizer.step()
		return

	def eval(self,t,x,y,debug=False):
		offset1 , offset2 = self.compute_offset(t)
		total_loss=0
		total_acc=0
		total_num=0
		self.model.eval()

		r=np.arange(x.size(0))
		r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

		with torch.no_grad():
			num_batches = len(x)//self.sbatch
			# Loop batches
			for i in range(0,len(r),self.sbatch):
				if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
				else: b=r[i:]
				images, targets = x[b].to(self.device), y[b].to(self.device)

				# Forward
				output=torch.nn.functional.log_softmax(self.model(images,sample=False), dim = 1)[: ,offset1 : offset2]
				# print(output.shape)
				loss = self.elbo_loss(images, targets, t, num_batches,sample=False,debug=debug)

				_, pred=output.max(1, keepdim=True)

				total_loss += loss.detach()*len(b)
				total_acc += pred.eq( (targets- offset1).view_as(pred)).sum().item() 
				total_num += len(b)           

		return total_loss/total_num, total_acc/total_num

	def compute_acc(self, t , x , y, num_sample):
		end = min(5*(t+1) , 10)
		total_loss=0
		total_acc=0
		total_num=0
		self.model.eval()

		r=np.arange(x.size(0))
		r=torch.as_tensor(r, device=self.device, dtype=torch.int64)
		with torch.no_grad():
			num_batches = len(x)//self.sbatch
			# Loop batches
			for i in range(0,len(r),self.sbatch):
				if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
				else: b=r[i:]
				images, targets = x[b].to(self.device), y[b].to(self.device)

				list_output , list_loss = [] , []
				# Forward
				for i in range(num_sample):
					output=self.model(images,sample=True)[: , (5*t):end ].cpu().numpy()
					loss = self.elbo_loss(images, targets, t, num_batches,sample=True)
					list_output.append(output)
					list_loss.append(loss.cpu().numpy())

				output_mean = np.mean( np.array(list_output), axis = 0)
				loss_mean = np.mean(np.array(list_loss))
				pred=np.argmax(output_mean, axis = 1)

				total_loss += loss_mean*len(b)
				total_acc += np.sum(pred == targets.cpu().numpy())
				total_num += len(b) 
		return total_loss/total_num, total_acc/total_num

	def eval_overall(self,t,x,y,debug=False):
		offset1 , offset2 = self.compute_offset(t)
		total_loss=0
		total_acc=0
		total_num=0
		self.model.eval()

		r=np.arange(x.size(0))
		r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

		with torch.no_grad():
			num_batches = len(x)//self.sbatch
			# Loop batches
			for i in range(0,len(r),self.sbatch):
				if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
				else: b=r[i:]
				images, targets = x[b].to(self.device), y[b].to(self.device)

				# Forward
				output=torch.nn.functional.log_softmax(self.model(images,sample=False), dim = 1)
				# print(output.shape)
				loss = self.elbo_loss(images, targets, t, num_batches,sample=False,debug=debug)

				_, pred=output.max(1, keepdim=True)

				total_loss += loss.detach()*len(b)
				total_acc += pred.eq( (targets).view_as(pred)).sum().item() 
				total_num += len(b)           

		return total_loss/total_num, total_acc/total_num

	def set_model_(model, state_dict):
		model.model.load_state_dict(copy.deepcopy(state_dict))

	def elbo_loss(self, input, target, t, num_batches, sample,debug=False):
		offset1, offset2 = self.compute_offset(t)
		if sample:
			# print("sample ", sample)
			predictions = []
			log_priors = torch.zeros(self.samples).to(self.device)
			log_variational_posterior = torch.zeros(self.samples).to(self.device)
			for i in range(self.samples):
				output = self.model(input,sample=sample)
				pred = torch.nn.functional.log_softmax(output, dim = 1)
				predictions.append(pred)
				log_priors[i], log_variational_posterior[i] = self.logs(0)

			# hack
			w1 = 1e-3
			w2 = 1e-3
			w3 = 5e-2

			outputs = torch.stack(predictions,dim=0).to(self.device)
			log_var = w1*log_variational_posterior.mean()
			log_p = w2*log_priors.mean()
			nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target , reduction='sum')
			return (log_var - log_p)/num_batches + nll

		else:
			predictions = []
			for i in range(self.samples):
				pred = torch.nn.functional.log_softmax(self.model(input,sample=sample), dim = 1)[: , offset1 : offset2]
				predictions.append(pred)
			w3 = 1.0

			outputs = torch.stack(predictions,dim=0).to(self.device)
			nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target - offset1, reduction='mean')

			return nll
	
	def re_compute_output(self, output , offset1 , offset2 ):
		if offset1 ==0:
			output[: , offset2 :] = -10e-10
		elif offset2 == 10:
			output[: , :offset1] = -10e-10
		return output

	def compute_offset(self, t ):
		end = min(2*(t+1) ,10)
		return (2*t , end)

	def test_list_valid(self, list_xvalid , list_yvalid):
		result_acc = []
		for i in range(len(list_xvalid)):
			acc =self.test(i , list_xvalid[i] , list_yvalid[i])
			result_acc.append(acc)
		return result_acc

	def save_model(self,t):
		torch.save({'model_state_dict': self.model.state_dict(),
		}, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))
