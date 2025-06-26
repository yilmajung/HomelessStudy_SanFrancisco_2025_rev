import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
print("Loading dataset...")
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')

print("Preprocessing dataset...")
df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['timestamp_sec'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Separate training data
df_training = df.dropna(subset=['ground_truth'])
#df_test = df[df['ground_truth'].isna()]

# Compute average counts per bounding box
bbox_counts = df_training.groupby('bboxid')['ground_truth'].mean().reset_index()

# # Top k bounding boxes by density (tent count)
# top_density_bboxes = bbox_counts.nlargest(num_inducing_density, 'ground_truth')['bboxid'].values

# # Exclude already selected points and randomly choose 100 bounding boxes
# remaining_bboxes = bbox_counts[~bbox_counts['bboxid'].isin(top_density_bboxes)]
# random_bboxes = remaining_bboxes.sample(n=num_random_points, random_state=42)['bboxid'].values

# # Combine both sets to form final inducing set (500 total)
# inducing_bbox_ids = np.concatenate([top_density_bboxes, random_bboxes])
# inducing_df = df_training[df_training['bboxid'].isin(inducing_bbox_ids)].drop_duplicates('bboxid')

# # Extract coordinates
# Z_spatial = inducing_df[['latitude','longitude']].values
# Z_temporal = inducing_df[['timestamp_sec']].values
# Z_covariates = inducing_df[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

# Extract coordinates and covariates
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_coords = df_training[['timestamp_sec']].values
X_covariates = df_training[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']]
y_counts = df_training['ground_truth'].values
bboxids = df_training['bboxid'].values

# Prepare training tensors
print("Preparing training tensors...")
train_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
train_y_np = y_counts

# Normalize the data
scaler = StandardScaler()
train_x_np = scaler.fit_transform(train_x_np)
train_y_np = train_y_np.reshape(-1, 1)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_x = torch.from_numpy(train_x_np).to(device)
all_y = torch.from_numpy(train_y_np).to(device)
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

time_splits = make_time_forward_splits(temporal_coords, n_splits=3, horizon_days=365)
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
    X_tr = all_x[train_idx]
    y_tr = all_y[train_idx]
    X_va = all_x[val_idx]
    y_va = all_y[val_idx]

    # pick inducing points by density-based + random subset of X_tr

    # Compute average counts per bounding box
    bbox_counts = df_training.groupby('bboxid')['ground_truth'].mean().reset_index()

    top_density_bboxes = bbox_counts.nlargest(num_inducing_density, 'ground_truth')['bboxid'].values
    remaining_bboxes = bbox_counts[~bbox_counts['bboxid'].isin(top_density_bboxes)]
    random_bboxes = remaining_bboxes.sample(n=num_inducing_random, random_state=42)['bboxid'].values
    
    idx = np.where(np.isin(bboxids, np.concatenate([top_density_bboxes, random_bboxes])))[0]
    Z   = X_tr[idx].clone()

    # build model & likelihood
    model, lik = make_stvgp_model(Z, outputscale)
    model, lik = model.to(device), lik.to(device)

    # optimizer & mll
    opt = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': lik.parameters()},
    ], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(train_idx))

    # train for a small number of iters
    model.train(); lik.train()
    for _ in range(params.get("train_iters", 200)):
        opt.zero_grad()
        out = model(X_tr)
        loss = -mll(out, y_tr)
        loss.backward()
        opt.step()

    # evaluate
    model.eval(); lik.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_dist   = model(X_va)
        p_dist   = lik(f_dist)
        mean_va  = p_dist.mean.cpu().numpy()
        rmse     = np.sqrt(mean_squared_error(y_va.cpu().numpy(), mean_va))
        logp_val = lik.expected_log_prob(y_va, f_dist)
        nlpd     = -logp_val.mean().item()

    return rmse, nlpd


# Wrap over ALL splits to get avg RMSE & avg NLPD for one hyper‐param combo
def evaluate_params(params):
    rmse_list, nlpd_list = [], []
    for train_idx, val_idx in combined_splits:
        rmse, nlpd = evaluate_single_split(params, train_idx, val_idx)
        rmse_list.append(rmse)
        nlpd_list.append(nlpd)
    return {
        **params,
        "avg_rmse": np.mean(rmse_list),
        "avg_nlpd": np.mean(nlpd_list),
    }


# Define grid & run in parallel
param_grid = {
    "num_inducing_density": [100, 250, 500],
    "num_inducing_random": [100, 250, 500],
    "lr":            [1e-2, 1e-3],
    "outputscale":  [0.01, 0.1],
    "train_iters":   [300],
}

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_params)(p) 
    for p in ParameterGrid(param_grid)
)

# Sort & inspect
df_res = pd.DataFrame(results)
df_res = df_res.sort_values(["avg_rmse", "avg_nlpd"])
print(df_res)

# Save results
df_res.to_csv("stvgp_cv_results.csv", index=False)
print("Cross-validation results saved to 'stvgp_cv_results.csv'.")