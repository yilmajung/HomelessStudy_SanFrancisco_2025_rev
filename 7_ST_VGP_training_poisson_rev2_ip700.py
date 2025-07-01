# train_stvgp_nb_constant_mean.py
import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn.functional as F

# Load and preprocess the dataset
print("Loading dataset...")
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')

# parse lat/lon, timestamp
print("Preprocessing dataset...")
df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Separate training data
df_training = df.dropna(subset=['ground_truth'])
#df_test = df[df['ground_truth'].isna()]

# Extract coordinates and covariates
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_coords = df_training[['timestamp']].values
X_covariates = df_training[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']]
y_counts = df_training['ground_truth'].values

# Inducing Points Strategy (Density-based + Random)
print("Selecting inducing points...")
# Number of inducing points
num_density_points = 400
num_random_points = 300

# Compute average counts per bounding box
bbox_counts = df_training.groupby('bboxid')['ground_truth'].mean().reset_index()

# Top 300 bounding boxes by density (tent count)
top_density_bboxes = bbox_counts.nlargest(num_density_points, 'ground_truth')['bboxid'].values

# Exclude already selected points and randomly choose 200 bounding boxes
remaining_bboxes = bbox_counts[~bbox_counts['bboxid'].isin(top_density_bboxes)]
random_bboxes = remaining_bboxes.sample(n=num_random_points, random_state=42)['bboxid'].values

# Combine both sets to form final inducing set (500 total)
inducing_bbox_ids = np.concatenate([top_density_bboxes, random_bboxes])
inducing_df = df_training[df_training['bboxid'].isin(inducing_bbox_ids)].drop_duplicates('bboxid')

# Extract coordinates
Z_spatial = inducing_df[['latitude','longitude']].values
Z_temporal = inducing_df[['timestamp']].values
Z_covariates = inducing_df[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

# Quick check for NaN values
assert not np.isnan(Z_spatial).any()
assert not np.isnan(Z_temporal).any()
assert not np.isnan(Z_covariates).any()

# Stack and scale
print("Preparing training tensors...")
t = np.hstack((spatial_coords, temporal_coords, X_covariates))
y = y_counts.astype(np.float32)
scaler = StandardScaler().fit(t)
x_scaled = scaler.transform(t).astype(np.float32)
joblib.dump(scaler, 'scaler_pois_constmean_ip700.joblib')

# Compute log-mean for ConstantMean initialization
log_y_mean = np.log(y.mean() + 1e-3).astype(np.float32)
joblib.dump(log_y_mean, 'constant_mean_pois_ip700.pkl')

# Prepare tensors
train_x = torch.tensor(x_scaled, dtype=torch.float32)
train_y = torch.tensor(y, dtype=torch.float32)

# Z_np defined from selected bboxids
Z_np = np.hstack((Z_spatial, Z_temporal, Z_covariates))
Z_scaled = scaler.transform(Z_np).astype(np.float32)
inducing_points = torch.tensor(Z_scaled, dtype=torch.float32)

# DataLoader
train_ds = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

# Define STVGP with ConstantMean
torch.manual_seed(0)
class STVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, constant_mean):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat= gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat)

        self.mean_module = gpytorch.means.ConstantMean()
        # initialize constant to log mean of counts
        self.mean_module.constant.data.fill_(constant_mean)

        self.spatial_kernel   = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
        self.temporal_kernel  = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5))
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=X_covariates.shape[1]))
        self.const_kernel     = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ConstantKernel())

    def forward(self, x):
        s, t, c = x[:, :2], x[:, 2:3], x[:, 3:]
        # Constant mean uses covariates to determine batch shape
        mean_x = self.mean_module(c)
        mean_x = mean_x.clamp(min=-3.0, max=3.0)
        Ks = self.spatial_kernel(s)
        Kt = self.temporal_kernel(t)
        Kc = self.covariate_kernel(c)
        Kconst = self.const_kernel(s)

        covar = Ks * Kt * Kc + Ks + Kt + Kc + Kconst
        covar = covar + torch.eye(covar.size(-1), device=x.device) * 1e-3  # jitter
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class PoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()
        # No parameters for vanilla Poisson

    def forward(self, function_samples, **kwargs):
        # The function_samples should be on log-scale
        rate = function_samples.exp()
        rate = torch.nan_to_num(rate, nan=1e-6, posinf=1e6, neginf=1e-6)
        rate = rate.clamp(min=1e-6, max=1e6)  # Ensure rate is positive
        return torch.distributions.Poisson(rate)
    
    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean
        rate = mean.exp()
        dist = torch.distributions.Poisson(rate)
        return dist.log_prob(target)


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STVGPModel(inducing_points.to(device), constant_mean=log_y_mean).to(device)
likelihood = PoissonLikelihood().to(device)

# Optimizer & MLL
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.005)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

# Training loop
print("Starting training...")
model.train(); likelihood.train()

for epoch in tqdm(range(700)):
    total_loss = 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        output = model(x_b)
        loss = -mll(output, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {total_loss:.3f}")

# Save
torch.save(model.state_dict(), 'stvgp_pois_constmean_ip700.pth')
torch.save(likelihood.state_dict(), 'likelihood_pois_constmean_ip700.pth')
torch.save(inducing_points, 'inducing_points_pois_constmean_ip700.pt')
