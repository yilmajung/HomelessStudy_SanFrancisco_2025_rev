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
num_density_points = 400
num_random_points = 100

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
Z_spatial = inducing_df[['latitude','longitude']].values
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
batch_size = 512
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define ST-VGP model
class STVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(STVGPModel, self).__init__(variational_strategy)

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=7))
        self.mean_module = gpytorch.means.LinearMean(input_size=7)
        self.spatial_kernel.outputscale = 0.1
        self.spatial_kernel.base_kernel.lengthscale = 1.0
        self.temporal_kernel.outputscale = 0.1
        self.temporal_kernel.base_kernel.lengthscale = 1.0
        self.covariate_kernel.outputscale = 0.1
        self.covariate_kernel.base_kernel.lengthscale = 1.0
        self.const_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel())

    def forward(self, x):
        spatial_x = x[:, :2]
        temporal_x = x[:, 2:3]
        covariate_x = x[:, 3:]
        mean_x = self.mean_module(covariate_x)
        mean_x = mean_x.clamp(min=-10.0, max=10.0)  # avoids very large exp()
        # Combine kernels
        Ks = self.spatial_kernel(spatial_x)
        Kt = self.temporal_kernel(temporal_x)
        Kc = self.covariate_kernel(covariate_x)
        const_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel())
        Kconst = self.const_kernel(spatial_x)

        covar_x = Ks * Kt * Kc + Ks + Kt + Kc + Kconst
        covar_x = covar_x + torch.eye(covar_x.size(-1), device=x.device) * 1e-3 # add jitter to avoid numerical issues

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Negative Binomial Likelihood
class StableNegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, init_dispersion=1.0):
        super().__init__()
        # Use log for better initialization, and ensure float32 for PyTorch
        raw_disp = torch.tensor(np.log(np.exp(init_dispersion) - 1), dtype=torch.float32)
        self.register_parameter(name="raw_log_dispersion", parameter=torch.nn.Parameter(raw_disp))

    @property
    def dispersion(self):
        return F.softplus(self.raw_log_dispersion) + 1e-5  # Ensure strictly positive

    def forward(self, function_samples, **kwargs):
        function_samples = function_samples.clamp(min=-10, max=10)
        mu = function_samples.exp().clamp(min=1e-3, max=1e3)
        r = self.dispersion
        logits = torch.log(mu + 1e-6) - torch.log(r + 1e-6)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN/Inf detected in logits!", logits)
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("NaN/Inf detected in mu!", mu)
        if torch.isnan(r).any() or torch.isinf(r).any():
            print("NaN/Inf detected in dispersion!", r)
        return torch.distributions.NegativeBinomial(total_count=r.expand_as(logits), logits=logits)

    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean.clamp(min=-10, max=10)
        mu = mean.exp().clamp(min=1e-3, max=1e3)
        r = self.dispersion
        logits = torch.log(mu + 1e-6) - torch.log(r + 1e-6)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN/Inf detected in logits!", logits)
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("NaN/Inf detected in mu!", mu)
        if torch.isnan(r).any() or torch.isinf(r).any():
            print("NaN/Inf detected in dispersion!", r)
        dist = torch.distributions.NegativeBinomial(total_count=r.expand_as(logits), logits=logits)
        return dist.log_prob(target)

# class StableNegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
#     def __init__(self):
#         super().__init__()
#         raw_disp = torch.tensor(np.log(np.exp(1.0) - 1)).unsqueeze(0)  # inverse softplus(3.0)
#         self.register_parameter(name="raw_log_dispersion", parameter=torch.nn.Parameter(raw_disp))


#     @property
#     def dispersion(self):
#         return torch.nn.functional.softplus(self.raw_log_dispersion)

#     def forward(self, function_samples, **kwargs):
#         mu = function_samples.exp().clamp(min=1e-3, max=1e3)
#         r = self.dispersion
#         logits = torch.log(mu + 1e-6) - torch.log(r + 1e-6)
#         return torch.distributions.NegativeBinomial(total_count=r.expand(mu.shape), logits=logits)

#     def expected_log_prob(self, target, function_dist, **kwargs):
#         mu = function_dist.mean.exp().clamp(min=1e-3, max=1e3)
#         r = self.dispersion
#         logits = torch.log(mu + 1e-6) - torch.log(r + 1e-6)

#         # r = self.dispersion + 0.0 * function_dist.mean.mean()  # force autograd to retain connection
#         # r = r.expand_as(logits).float()

#         dist = torch.distributions.NegativeBinomial(total_count=r.expand(mu.shape), logits=logits)
#         return dist.log_prob(target)

#     def log_marginal(self, observations, function_dist, **kwargs):
#         return self.expected_log_prob(observations, function_dist, **kwargs)

# class StableNegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
#     def __init__(self, init_dispersion=1.0):
#         super().__init__()
#         raw_disp = torch.tensor(init_dispersion).log().unsqueeze(0)
#         self.register_parameter(name="raw_log_dispersion", parameter=torch.nn.Parameter(raw_disp))

#     @property
#     def dispersion(self):
#         return F.softplus(self.raw_log_dispersion)

#     def forward(self, function_samples, **kwargs):
#         # Convert GP output to mean
#         mu = function_samples.exp()
#         mu = mu.clamp(min=1e-3, max=1e3)  # avoid overflows

#         # Convert to total_count and probs
#         total_count = self.dispersion
#         probs = total_count / (total_count + mu)
#         probs = probs.clamp(min=1e-4, max=1 - 1e-4)

#         return NegativeBinomial(total_count=total_count, probs=probs)

#     def expected_log_prob(self, target, function_dist, **kwargs):
#         print("GP mean stats:")
#         print("  min:", function_dist.mean.min().item())
#         print("  max:", function_dist.mean.max().item())
#         print("  any NaN?", torch.isnan(function_dist.mean).any().item())

#         mean = function_dist.mean.exp()
#         print("Exp(mean) stats:")
#         print("  min:", mean.min().item())
#         print("  max:", mean.max().item())
#         print("  any NaN?", torch.isnan(mean).any().item())

#         mean = mean.clamp(min=1e-3, max=1e3)

#         total_count = self.dispersion
#         probs = total_count / (total_count + mean)
#         probs = probs.clamp(min=1e-4, max=1 - 1e-4)

#         print("Probs stats:")
#         print("  min:", probs.min().item())
#         print("  max:", probs.max().item())
#         print("  any NaN?", torch.isnan(probs).any().item())

#         dist = NegativeBinomial(total_count=total_count, probs=probs)
#         return dist.log_prob(target)

#     def log_marginal(self, observations, function_dist, **kwargs):
#         return self.expected_log_prob(observations, function_dist, **kwargs)


# class NegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
#     def __init__(self, init_log_dispersion=0.0):
#         super().__init__()
#         raw_log_disp = torch.tensor(init_log_dispersion).float()
#         self.register_parameter("raw_log_dispersion", torch.nn.Parameter(raw_log_disp))
#         self.register_constraint("raw_log_dispersion", gpytorch.constraints.GreaterThan(-6.0))  # softer constraint

#     @property
#     def dispersion(self):
#         # Use softplus to ensure positivity and prevent NaNs
#         return F.softplus(self.raw_log_dispersion)

#     def forward(self, function_samples, **kwargs):
#         mu = function_samples.exp()
#         total_count = self.dispersion
#         probs = total_count / (total_count + mu)
#         probs = probs.clamp(min=1e-4, max=1 - 1e-4)  # avoid NaNs
#         return NegativeBinomial(total_count=total_count, probs=probs)

#     def expected_log_prob(self, target, function_dist, **kwargs):
#         mean = function_dist.mean.exp()
#         total_count = self.dispersion
#         probs = total_count / (total_count + mean)
#         probs = probs.clamp(min=1e-4, max=1 - 1e-4)
#         dist = NegativeBinomial(total_count=total_count, probs=probs)
#         return dist.log_prob(target)

#     def log_marginal(self, observations, function_dist, **kwargs):
#         mean = function_dist.mean.exp()
#         total_count = self.dispersion
#         probs = total_count / (total_count + mean)
#         probs = probs.clamp(min=1e-4, max=1 - 1e-4)
#         dist = NegativeBinomial(total_count=total_count, probs=probs)
#         return dist.log_prob(observations)

# class NegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
#     def __init__(self, dispersion=1.0):
#         super().__init__()
#         self.raw_dispersion = torch.nn.Parameter(torch.tensor(float(dispersion)))
#         self.register_parameter(name="raw_dispersion", parameter=self.raw_dispersion)
#         self.register_constraint("raw_dispersion", gpytorch.constraints.Positive())

#     @property
#     def dispersion(self):
#         return self.raw_dispersion_constraint.transform(self.raw_dispersion)

#     def forward(self, function_samples, **kwargs):
#         mu = function_samples.exp()
#         total_count = self.dispersion.float()
#         probs = total_count / (total_count + mu)
#         probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid numerical issues
#         return NegativeBinomial(total_count=total_count, probs=probs)

#     def expected_log_prob(self, target, function_dist, **kwargs):
#         mean = function_dist.mean.exp()
#         total_count = self.dispersion.float()
#         probs = total_count / (total_count + mean)
#         probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid numerical issues
#         dist = NegativeBinomial(total_count=total_count, probs=probs)
#         return dist.log_prob(target).sum(-1)

#     def log_marginal(self, observations, function_dist, **kwargs):
#         mean = function_dist.mean.exp()
#         total_count = self.dispersion.float()
#         probs = total_count / (total_count + mean)
#         probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid numerical issues
#         dist = NegativeBinomial(total_count=total_count, probs=probs)
#         return dist.log_prob(observations).sum(-1)

# Move model and likelihood to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
likelihood = StableNegativeBinomialLikelihood().to(device)
model = STVGPModel(inducing_points.to(device)).to(device)


print("Spatial lengthscale:", model.spatial_kernel.base_kernel.lengthscale.data)
print("Temporal lengthscale:", model.temporal_kernel.base_kernel.lengthscale.data)
print("Covariate lengthscale:", model.covariate_kernel.base_kernel.lengthscale.data)
print("Spatial outputscale:", model.spatial_kernel.outputscale.data)
print("Temporal outputscale:", model.temporal_kernel.outputscale.data)
print("Covariate outputscale:", model.covariate_kernel.outputscale.data)

# Quick diagnose for kernel matrix
with torch.no_grad():
    x_batch = train_x[:32].to(device)
    y_batch = train_y[:32].to(device)
    output = model(x_batch)
    print("Output mean:", output.mean)
    print("Output covar diag:", output.covariance_matrix.diag())
    print("Any NaN in output mean?", torch.isnan(output.mean).any().item())
    print("Any NaN in output covar?", torch.isnan(output.covariance_matrix).any().item())
    print("Any Inf in output covar?", torch.isinf(output.covariance_matrix).any().item())

# with torch.no_grad():
#     # Evaluate kernels for your inducing points
#     x_sp = inducing_points[:, :2].to(device)
#     x_tm = inducing_points[:, 2:3].to(device)
#     x_cov = inducing_points[:, 3:].to(device)

#     Ks2 = model.spatial_kernel(x_sp)
#     Kt2 = model.temporal_kernel(x_tm)
#     Kc2 = model.covariate_kernel(x_cov)
#     K_prod = Ks2.evaluate() * Kt2.evaluate() * Kc2.evaluate()
#     K_sum = Ks2.evaluate() + Kt2.evaluate() + Kc2.evaluate()
#     K_hybrid = K_prod + K_sum
#     print("K_prod min:", K_prod.min().item(), "max:", K_prod.max().item(), "any NaN?", torch.isnan(K_prod).any().item())
#     print("K_sum min:", K_sum.min().item(), "max:", K_sum.max().item(), "any NaN?", torch.isnan(K_sum).any().item())
#     print("K_hybrid min:", K_hybrid.min().item(), "max:", K_hybrid.max().item(), "any NaN?", torch.isnan(K_hybrid).any().item())
# with torch.no_grad():
#     x_sp = inducing_points[:, :2].to(device)
#     x_tm = inducing_points[:, 2:3].to(device)
#     x_cov = inducing_points[:, 3:].to(device)

#     K_sp = model.spatial_kernel(x_sp).evaluate()
#     K_tm = model.temporal_kernel(x_tm).evaluate()
#     K_cov = model.covariate_kernel(x_cov).evaluate()

#     K_total = K_sp + K_tm + K_cov

#     print("K_total min:", K_total.min().item())
#     print("K_total mean:", K_total.mean().item())
#     print("K_total diag min:", K_total.diag().min().item())
#     print("K_total is NaN:", torch.isnan(K_total).any().item())

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# print("Optimizer parameter groups and parameter names:")
# for i, group in enumerate(optimizer.param_groups):
#     print(f"\nGroup {i}:")
#     for p in group['params']:
#         # Loop through model and likelihood named_parameters to find name
#         found = False
#         for name, param in model.named_parameters():
#             if p is param:
#                 print(f"  MODEL PARAM: {name}, shape={p.shape}")
#                 found = True
#         for name, param in likelihood.named_parameters():
#             if p is param:
#                 print(f"  LIKELIHOOD PARAM: {name}, shape={p.shape}, value={p.data.item()}")
#                 found = True
#         if not found:
#             print(f"  UNKNOWN PARAM shape={p.shape}")

mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

#scaler2 = GradScaler()
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
        # Standard forward and backward (no AMP)
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}/{training_iterations}: Avg Loss = {total_loss:.3f}")
        print(f"Current dispersion: {likelihood.dispersion.item():.4f}")
        print("x_batch[:5]:", x_batch[:5])

print("Training complete.")

# for i in tqdm(range(training_iterations)):
#     total_loss = 0
#     for x_batch, y_batch in train_loader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         with autocast(), gpytorch.settings.max_cholesky_size(1000), gpytorch.settings.fast_computations(True):#, gpytorch.settings.cholesky_jitter(1e-3):
#             output = model(x_batch)
#             loss = -mll(output, y_batch)
#         scaler2.scale(loss).backward()
#         scaler2.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#         torch.nn.utils.clip_grad_norm_(likelihood.parameters(), max_norm=10.0)
#         scaler2.step(optimizer)
#         scaler2.update()
#         total_loss += loss.item()
#     if (i+1) % 10 == 0:
#         print(f"Iteration {i+1}/{training_iterations}: Avg Loss = {total_loss:.3f}")
#         print(f"Current dispersion: {likelihood.dispersion.item():.4f}")
#         print("Dispersion gradient:", likelihood.raw_log_dispersion.grad)
#         #print(f"Kernel lengthscale: {model.covariate_kernel.base_kernel.lengthscale.detach().cpu().numpy()}")
#         for n, p in model.named_parameters():
#             if p.grad is not None:
#                 print(f"{n} grad norm: {p.grad.norm().item()}")


# Save the model
print("Saving model...")
torch.save(model.state_dict(), 'stvgp_model.pth')
torch.save(likelihood.state_dict(), 'stvgp_likelihood.pth')
joblib.dump(scaler, 'scaler.pkl')

# Save inducing points
print("Saving inducing points...")
torch.save(inducing_points, 'inducing_points.pt')