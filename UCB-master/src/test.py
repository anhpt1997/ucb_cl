# -*- coding: utf-8 -*-

import torch

a = torch.tensor([1., 2.])
print(torch.nn.functional.softmax(a, dim= 0))