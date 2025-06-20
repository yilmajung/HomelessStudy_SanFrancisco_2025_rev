import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
import joblib
from torch.distributions import NegativeBinomial, constraints
import torch.nn.functional as F

r = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
mu = torch.ones(10, requires_grad=True)
logits = torch.log(mu + 1e-6) - torch.log(F.softplus(r) + 1e-6)
dist = torch.distributions.NegativeBinomial(total_count=F.softplus(r).expand_as(logits), logits=logits)
target = torch.poisson(mu)
loss = -dist.log_prob(target).mean()
loss.backward()
print("Grad wrt r:", r.grad)
