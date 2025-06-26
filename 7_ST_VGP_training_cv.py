# Set up OPENBLAS and other threading environments
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]    = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"

import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler



# Load and preprocess the dataset
print("Loading dataset...")
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')

print("Preprocessing dataset...")
df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Separate training data
df_training = df.dropna(subset=['ground_truth'])
#df_test = df[df['ground_truth'].isna()]

# Compute average counts per bounding box
bbox_counts = df_training.groupby('bboxid')['ground_truth'].mean().reset_index()

# Extract coordinates and covariates
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_secs = ((df_training['timestamp'] - pd.Timestamp("1970-01-01")) 
                   // pd.Timedelta("1s")).values
X_covariates = df_training[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']]
y_counts = df_training['ground_truth'].values.astype(np.float32)
bboxids = df_training['bboxid'].values

# Prepare training tensors
print("Preparing training tensors...")
train_x_np = np.hstack((spatial_coords, temporal_secs[:,None], X_covariates))
train_y_np = y_counts

# Normalize the data
train_x_np = StandardScaler().fit_transform(train_x_np)
train_y_np = train_y_np.reshape(-1, 1)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_x = torch.from_numpy(train_x_np)
all_y = torch.from_numpy(train_y_np).float().squeeze(1)
print(f"all_x shape: {all_x.shape}, all_y shape: {all_y.shape}")

# Build CV splits
# Temporal splits
def make_time_forward_splits(timestamps, n_splits=3, horizon_days=365):
    unique_times = np.sort(np.unique(timestamps))
    splits = []
    max_idx = len(unique_times) - horizon_days
    block = max_idx // n_splits
    for k in range(n_splits):
        train_end = unique_times[(k+1)*block - 1]
        val_start = unique_times[(k+1)*block]
        val_end   = val_start + horizon_days * 24*3600
        train_idx = np.where(timestamps <= train_end)[0]
        val_idx   = np.where((timestamps > train_end) & (timestamps <= val_end))[0]
        splits.append((train_idx, val_idx))
    return splits

# Spatial block splits via Kmeans over lat/lon
def make_spatial_splits(spatial_coords, n_blocks=5):
    km = KMeans(n_clusters=n_blocks, random_state=42).fit(spatial_coords)
    labels = km.labels_
    splits = []

    for b in range(n_blocks):
        val_idx = np.where(labels == b)[0]
        train_idx = np.where(labels != b)[0]
        splits.append((train_idx, val_idx))

    return splits

time_splits = make_time_forward_splits(temporal_secs, n_splits=3, horizon_days=365)
spatial_splits = make_spatial_splits(spatial_coords, n_blocks=5)

# Combine time and spatial splits
combined_splits = []
for train_t, val_t in time_splits:
    for train_s, val_s in spatial_splits:
        # train only on data in BOTH train-time AND train-space
        train_idx = np.intersect1d(train_t, train_s)
        # validate on anything in the held-out time OR the held-out space
        val_idx   = np.union1d(val_t, val_s)
        combined_splits.append((train_idx, val_idx))

# STVGP + NB likelihood
def make_stvgp_model(inducing_points, outputscale):
    class STVGP(gpytorch.models.ApproximateGP):
        def __init__(self, Z):
            variational_dist = gpytorch.variational.CholeskyVariationalDistribution(Z.size(0))
            variational_strat= gpytorch.variational.VariationalStrategy(
                self, Z, variational_dist, learn_inducing_locations=True)
            super().__init__(variational_strat)
            self.spatial_kernel   = gpytorch.kernels.ScaleKernel(
                                      gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
            self.temporal_kernel  = gpytorch.kernels.ScaleKernel(
                                      gpytorch.kernels.MaternKernel(nu=1.5))
            self.covariate_kernel = gpytorch.kernels.ScaleKernel(
                                      gpytorch.kernels.RBFKernel(ard_num_dims=7))
            self.mean_module      = gpytorch.means.ZeroMean()
            # set outputscales
            for kern in (self.spatial_kernel, self.temporal_kernel, self.covariate_kernel):
                kern.outputscale = outputscale

        def forward(self, x):
            s, t, c = x[:, :2], x[:, 2:3], x[:, 3:]
            mean_x = self.mean_module(c).clamp(-10, 10)
            Ks = self.spatial_kernel(s)
            Kt = self.temporal_kernel(t)
            Kc = self.covariate_kernel(c)
            covar = Ks * Kt * Kc + Ks + Kt + Kc + 1e-3 * torch.eye(x.size(0), device=x.device)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    # Negative-Binomial likelihood with correct p = r/(r+μ)
    class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
        def __init__(self, init_disp=1.0):
            super().__init__()
            raw = torch.tensor(np.log(np.exp(init_disp)-1), dtype=torch.float32)
            self.register_parameter("raw_disp", torch.nn.Parameter(raw))

        @property
        def dispersion(self):
            return torch.nn.functional.softplus(self.raw_disp) + 1e-5

        def forward(self, f_samples, **kwargs):
            log_mu = f_samples.clamp(-10,10)
            mu     = log_mu.exp().clamp(1e-3,1e3)
            r      = self.dispersion
            probs  = r / (r + mu)
            return torch.distributions.NegativeBinomial(
                      total_count=r.expand_as(mu), probs=probs)

        def expected_log_prob(self, target, f_dist, **kwargs):
            log_mu = f_dist.mean.clamp(-10,10)
            mu     = log_mu.exp().clamp(1e-3,1e3)
            r      = self.dispersion
            probs  = r / (r + mu)
            dist   = torch.distributions.NegativeBinomial(
                        total_count=r.expand_as(mu), probs=probs)
            return dist.log_prob(target)

    return STVGP(inducing_points), NBLikelihood()


# Single hyper‐param evaluation on one combined CV split
def evaluate_single_split(params, train_idx, val_idx):
    # unpack
    num_inducing_density = params["num_inducing_density"]
    num_inducing_random = params["num_inducing_random"]
    lr = params["lr"]
    outputscale = params["outputscale"]

    # prepare fold data
    X_tr = all_x[train_idx].to(device)
    y_tr = all_y[train_idx].long().to(device)
    X_va = all_x[val_idx].to(device)
    y_va = all_y[val_idx].long().to(device)
    

    # Pick inducing points by density-based + random subset of X_tr
    # Compute average counts per bounding box
    train_bids = bboxids[train_idx]
    train_vals = y_counts[train_idx]
    fold_df = pd.DataFrame({
        'bboxid': train_bids,
        'count': train_vals
    })

    fold_counts = (fold_df.groupby('bboxid')['count'].mean().reset_index())

    dens_bids = fold_counts.nlargest(num_inducing_density, 'count')['bboxid'].values
    rem_bids = fold_counts[~fold_counts['bboxid'].isin(dens_bids)]
    rand_bids = rem_bids.sample(n=num_inducing_random, random_state=42)['bboxid'].values
    inducing_ids = np.concatenate([dens_bids, rand_bids])

    mask = np.isin(train_bids, inducing_ids)
    local_idx = np.where(mask)[0]
    Z = X_tr[local_idx].clone()

    # build model & likelihood
    model, lik = make_stvgp_model(Z, outputscale)
    model, lik = model.to(device), lik.to(device)

    # optimizer & mll
    opt = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': lik.parameters()},
    ], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(train_idx))

    # Train with batching
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=256, shuffle=True, drop_last=True
    )
    
    # train for a small number of iters
    scaler = GradScaler()
    model.train(); lik.train()
    for _ in range(params.get("train_iters", 200)):
        total_loss = 0
        for x_b, y_b in train_loader:
            opt.zero_grad()
            with autocast():
                out = model(x_b)
                loss = -mll(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()


    # evaluate
    val_loader = DataLoader(
        TensorDataset(X_va, y_va),batch_size=256, shuffle=False
    )
    model.eval(); lik.eval()

    preds, logps = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_b, y_b in val_loader:
            f_dist = model(x_b)
            p_dist = lik(f_dist)
            preds.append(p_dist.mean.cpu())  
            logps.append(lik.expected_log_prob(y_b, f_dist).cpu())

    preds = torch.cat(preds).numpy()    # shape [n_val,]
    logps = torch.cat(logps).numpy()    # shape [n_val,]

    true = y_va.cpu().numpy().ravel()       # flatten [n_val,1] → [n_val]
    rmse = np.sqrt(np.mean((preds-true)**2))    
    nlpd = -np.mean(logps)
    
    del model, lik, opt, mll, preds, logps, f_dist, p_dist
    torch.cuda.empty_cache()

    return rmse, nlpd


# Wrap over ALL splits to get avg RMSE & avg NLPD for one hyper‐param combo
def evaluate_params(params):
    rmse_list, nlpd_list = [], []
    for train_idx, val_idx in tqdm(combined_splits, desc=f"CV folds ({params['num_inducing_density']} pts)",
                                  leave=False):
        rmse, nlpd = evaluate_single_split(params, train_idx, val_idx)
        rmse_list.append(rmse)
        nlpd_list.append(nlpd)
    return {
        **params,
        "avg_rmse": np.mean(rmse_list),
        "avg_nlpd": np.mean(nlpd_list),
    }


# Define grid & run in parallel
print("Starting cross-validation...")
param_grid = {
    "num_inducing_density": [100, 250, 500],
    "num_inducing_random": [100, 250, 500],
    "lr":            [1e-2, 1e-3],
    "outputscale":  [0.01, 0.1],
    "train_iters":   [300],
}

grid = list(ParameterGrid(param_grid))
print(f"Total combinations: {len(grid)}")

with tqdm_joblib(tqdm(desc="Grid search", total=len(grid))):
    results = Parallel(n_jobs=1)(
        delayed(evaluate_params)(p) 
        for p in grid
)

# Sort & inspect
df_res = pd.DataFrame(results)
df_res = df_res.sort_values(["avg_rmse", "avg_nlpd"])
print(df_res)

# Save results
df_res.to_csv("stvgp_cv_results.csv", index=False)
print("Cross-validation results saved to 'stvgp_cv_results.csv'.")