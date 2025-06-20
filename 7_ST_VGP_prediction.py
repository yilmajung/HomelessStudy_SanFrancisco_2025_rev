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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved components
print("Loading scaler, model weights, and inducing points...")
scaler = joblib.load("scaler.pkl")
inducing_points = torch.load("inducing_points.pt").to(device)

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

# Load model and likelihood
model = STVGPModel(inducing_points).to(device)
likelihood = StableNegativeBinomialLikelihood().to(device)

model.load_state_dict(torch.load("stvgp_model.pth"))
likelihood.load_state_dict(torch.load("stvgp_likelihood.pth"))

model.eval()
likelihood.eval()

# Load and prepare test data
print("Loading test data...")
df = pd.read_csv("~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv")
df_test = df[df['ground_truth'].isna()].copy()
print(f"Test data shape: {df_test.shape}")

# Parse lat/lon and timestamp
df_test['latitude'] = df_test['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df_test['longitude'] = df_test['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df_test['latitude'] = df_test['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df_test['longitude'] = df_test['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
df_test['timestamp'] = (df_test['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Covariate order must match
test_x_np = df_test[['latitude', 'longitude', 'timestamp', 'max','min','precipitation',
                     'total_population','white_ratio','black_ratio','hh_median_income']].values

# Apply the same scaler used during training
test_x_scaled = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32)

# Predict with uncertainty
print("Running predictions...")
batch_size = 512
test_dataset = TensorDataset(test_x_scaled)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_means, all_stddevs = [], []

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for (x_batch,) in tqdm(test_loader):
        x_batch = x_batch.to(device)
        preds = likelihood(model(x_batch))
        mean_batch = preds.mean.cpu().numpy().reshape(-1)
        stddev_batch = preds.stddev.cpu().numpy().reshape(-1)
        all_means.append(mean_batch)
        all_stddevs.append(stddev_batch)

predicted_counts = np.concatenate(all_means)
predicted_std = np.concatenate(all_stddevs)

df_test['predicted_counts'] = predicted_counts
df_test['predicted_std'] = predicted_std

# Save predictions
df_test[['bboxid', 'timestamp', 'predicted_counts', 'predicted_std']].to_csv(
    '~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/predictions_st_vgp.csv', 
    index=False
)

print("Prediction saved.")