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

# Load and preprocess the dataset
print("Loading dataset...")
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')

print("Preprocessing dataset...")
df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Separate training data
df_training = df.dropna(subset=['ground_truth'])
df_test = df[df['ground_truth'].isna()]

# Extract coordinates and covariates
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_coords = df_training[['timestamp']].values
X_covariates = df_training[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']]
y_counts = df_training['ground_truth'].values

# Inducing Points Strategy (Density-based + Random)
print("Selecting inducing points...")
# Number of inducing points
num_density_points = 250
num_random_points = 50

# Compute average counts per bounding box
bbox_counts = df_training.groupby('bboxid')['ground_truth'].mean().reset_index()

# Top 400 bounding boxes by density (tent count)
top_density_bboxes = bbox_counts.nlargest(num_density_points, 'ground_truth')['bboxid'].values

# Exclude already selected points and randomly choose 100 bounding boxes
remaining_bboxes = bbox_counts[~bbox_counts['bboxid'].isin(top_density_bboxes)]
random_bboxes = remaining_bboxes.sample(n=num_random_points, random_state=42)['bboxid'].values

# Combine both sets to form final inducing set (500 total)
inducing_bbox_ids = np.concatenate([top_density_bboxes, random_bboxes])
inducing_df = df_training[df_training['bboxid'].isin(inducing_bbox_ids)].drop_duplicates('bboxid')

# Extract coordinates
Z_spatial = inducing_df[['longitude', 'latitude']].values
Z_temporal = inducing_df[['timestamp']].values
Z_covariates = inducing_df[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

# Quick check for NaN values
assert not np.isnan(Z_spatial).any()
assert not np.isnan(Z_temporal).any()
assert not np.isnan(Z_covariates).any()

# # Final inducing points tensor
# inducing_points = torch.tensor(np.hstack((Z_spatial, np.full((Z_spatial.shape[0], 1), Z_temporal), Z_covariates)), dtype=torch.float32)

# Prepare training tensors
print("Preparing training tensors...")
train_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
train_y_np = y_counts
scaler = StandardScaler()
train_x = torch.tensor(scaler.fit_transform(train_x_np), dtype=torch.float32)
train_y = torch.tensor(train_y_np, dtype=torch.float32)
inducing_points_np = np.hstack((Z_spatial, Z_temporal, Z_covariates))
inducing_points = torch.tensor(scaler.transform(inducing_points_np), dtype=torch.float32)

# Dataset and DataLoader for batching
batch_size = 1024
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model with MiniBatchVariationalStrategy
class STVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(STVGPModel, self).__init__(variational_strategy)

        # self.spatial_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
        # self.temporal_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=7))
        self.mean_module = gpytorch.means.LinearMean(input_size=7)
        # self.spatial_kernel.outputscale = 0.1
        # self.spatial_kernel.base_kernel.lengthscale = 1.0
        # self.temporal_kernel.outputscale = 0.1
        # self.temporal_kernel.base_kernel.lengthscale = 1.0
        self.covariate_kernel.outputscale = 0.1
        self.covariate_kernel.base_kernel.lengthscale = 5.0


    def forward(self, x):
        spatial_x = x[:, :2]
        temporal_x = x[:, 2:3]
        covariate_x = x[:, 3:]
        mean_x = self.mean_module(covariate_x)
        covar_x = self.covariate_kernel(covariate_x) #self.spatial_kernel(spatial_x) * self.temporal_kernel(temporal_x) * 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Negative Binomial Likelihood
from torch.distributions import NegativeBinomial

class NegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, dispersion=1.0):
        super().__init__()
        self.register_parameter(name="raw_dispersion", parameter=torch.nn.Parameter(torch.tensor(dispersion)))
        self.register_constraint("raw_dispersion", gpytorch.constraints.Positive())

    @property
    def dispersion(self):
        return self.raw_dispersion_constraint.transform(self.raw_dispersion)

    def forward(self, function_samples, **kwargs):
        mu = function_samples.exp()
        total_count = self.dispersion
        probs = total_count / (total_count + mu)
        return NegativeBinomial(total_count=total_count, probs=probs)

    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean.exp()
        total_count = self.dispersion
        probs = total_count / (total_count + mean)
        dist = NegativeBinomial(total_count=total_count, probs=probs)
        return dist.log_prob(target).sum(-1)

    def log_marginal(self, observations, function_dist, **kwargs):
        mean = function_dist.mean.exp()
        total_count = self.dispersion
        probs = total_count / (total_count + mean)
        dist = NegativeBinomial(total_count=total_count, probs=probs)
        return dist.log_prob(observations).sum(-1)

# Move model and likelihood to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
likelihood = NegativeBinomialLikelihood(dispersion=1.0).to(device)
model = STVGPModel(inducing_points.to(device)).to(device)

# Quick diagnose for kernel matrix

with torch.no_grad():
    # Extract only the covariate part: columns 3 onward
    inducing_covariates = inducing_points[:, 3:]
    cov = model.covariate_kernel(inducing_covariates.to(device)).evaluate()
    print("Cov matrix stats:")
    print("  min:", cov.min().item())
    print("  max:", cov.max().item())
    print("  mean:", cov.mean().item())
    print("  diag min:", cov.diag().min().item())
    print("  diag max:", cov.diag().max().item())
    print("  Is NaN:", torch.isnan(cov).any().item())
    print("  Is Inf:", torch.isinf(cov).any().item())
    print("  Is Symmetric:", torch.allclose(cov, cov.T, atol=1e-5))

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.005)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

scaler2 = GradScaler()
training_iterations = 500

# Quick check for NaN values
assert not torch.isnan(train_x).any(), "train_x contains NaNs"
assert not torch.isinf(train_x).any(), "train_x contains Infs"
assert not torch.isnan(inducing_points).any(), "inducing_points contain NaNs"
assert not torch.isinf(inducing_points).any(), "inducing_points contain Infs"

# Training loop with AMP and mini-batching
print("Starting training...")
for i in tqdm(range(training_iterations)):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with autocast(), gpytorch.settings.max_cholesky_size(1000), gpytorch.settings.fast_computations(True):#, gpytorch.settings.cholesky_jitter(1e-3):
            output = model(x_batch)
            loss = -mll(output, y_batch)
        scaler2.scale(loss).backward()
        scaler2.step(optimizer)
        scaler2.update()
        total_loss += loss.item()
    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}/{training_iterations}: Avg Loss = {total_loss:.3f}")

# Prediction
print("Starting prediction...")
model.eval()
likelihood.eval()

test_x = torch.tensor(df_test[['latitude', 'longitude', 'timestamp', 'max','min','precipitation',
                               'total_population','white_ratio','black_ratio','hh_median_income']].values, dtype=torch.float32)

test_dataset = TensorDataset(test_x)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_means = []
all_stddevs = []

with torch.no_grad(), gpytorch.settings.fast_pred_var(), autocast():
    for (x_batch,) in tqdm(test_loader):
        preds = likelihood(model(x_batch.cuda()))
        mean_batch = preds.mean.cpu().numpy()
        stddev_batch = preds.stddev.cpu().numpy()

        all_means.append(mean_batch)
        all_stddevs.append(stddev_batch)

predicted_counts = np.concatenate(all_means)
predicted_std = np.concatenate(all_stddevs)

df_test['predicted_counts'] = predicted_counts
df_test['predicted_std'] = predicted_std

# Save predictions with uncertainty
df_test[['bboxid', 'predicted_counts', 'predicted_std']].to_csv(
    '~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/predictions_st_vgp.csv', 
    index=False
)