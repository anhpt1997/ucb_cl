import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import VariationalPosterior, Prior , GaussMixtureVariationalPosterior
import numpy as np

class BayesianLinear(nn.Module):

    '''
    Applies a linear Bayesian transformation to the incoming data: :math:`y = Ax + b`
    '''

    def __init__(self, in_features, out_features, args, use_bias=True):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.device = args.device
        self.num_component = 2
        self.rho = args.rho

        # Variational Posterior Distributions
        self.weight_mu = nn.Parameter(torch.empty(( self.num_component , out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        self.weight_rho = nn.Parameter(self.rho + torch.empty((self.num_component , out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        self.weight_alpha = nn.Parameter(torch.empty(( self.num_component , out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        
        print("shape ", self.weight_mu.shape)
        self.weight = GaussMixtureVariationalPosterior(self.weight_mu , self.weight_rho , self.weight_alpha , self.device)

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((self.num_component , out_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
            self.bias_rho = nn.Parameter(self.rho + nn.Parameter(torch.empty(self.num_component , out_features,
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True))
            self.bias_alpha = nn.Parameter(torch.empty((self.num_component , out_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True) 


            # self.bias_alpha = nn.Parameter(torch.cat((weight_alpha_component_0 , weight_alpha_component_1), dim = 0))
            
            self.bias = GaussMixtureVariationalPosterior(self.bias_mu , self.bias_rho , self.bias_alpha, self.device)
        else:
            self.register_parameter('bias', None)      

        # Prior Distributions
        self.weight_prior = Prior(args)
        if self.use_bias:      
            self.bias_prior = Prior(args)

        # Initialize log prior and log posterior
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.mask_flag = False

    def prune_module(self, mask):
        self.mask_flag = True 
        self.pruned_weight_mu=self.weight_mu.data.clone().mul_(mask).to(self.device)
        self.pruned_weight_rho=self.weight_rho.data.clone().mul_(mask).to(self.device)


    def forward(self, input, sample=False, calculate_log_probs=False):
        # if self.mask_flag:
        #     self.weight = VariationalPosterior(self.pruned_weight_mu, self.pruned_weight_rho, self.device)
            # if self.use_bias:
            #     self.bias = VariationalPosterior(self.pruned_bias_mu, self.pruned_bias_rho)

        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample() if self.use_bias else None
            
        else:
            print("sam ple test")
            weight = self.weight.sample_test()
            bias = self.bias.sample_test()
                
        # print("self. training ", self.training , calculate_log_probs)
        if self.training or calculate_log_probs:

            #trong luc feed forward thi tinh log posterior luon
            if self.use_bias:
                self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
                self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            else:
                self.log_prior = self.weight_prior.log_prob(weight)
                self.log_variational_posterior = self.weight.log_prob(weight)
            
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        
        # print("log prior ",self.log_prior)
        return F.linear(input, weight, bias)

