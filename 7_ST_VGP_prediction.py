import torch
import gpytorch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
import joblib
from tqdm import tqdm
from torch.distributions import NegativeBinomial

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


    def forward(self, x):
        spatial_x = x[:, :2]
        temporal_x = x[:, 2:3]
        covariate_x = x[:, 3:]
        mean_x = self.mean_module(covariate_x)
        covar_x = self.spatial_kernel(spatial_x) + self.temporal_kernel(temporal_x) + self.covariate_kernel(covariate_x) +\
                  self.spatial_kernel(spatial_x) * self.temporal_kernel(temporal_x) * self.covariate_kernel(covariate_x)
        covar_x = covar_x + torch.eye(covar_x.size(-1), device=x.device) * 1e-2

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Negative Binomial Likelihood
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
        probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid numerical issues
        return NegativeBinomial(total_count=total_count, probs=probs)

    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean.exp()
        total_count = self.dispersion
        probs = total_count / (total_count + mean)
        probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid numerical issues
        dist = NegativeBinomial(total_count=total_count, probs=probs)
        return dist.log_prob(target).sum(-1)

    def log_marginal(self, observations, function_dist, **kwargs):
        mean = function_dist.mean.exp()
        total_count = self.dispersion
        probs = total_count / (total_count + mean)
        probs = probs.clamp(min=1e-6, max=1-1e-6)  # Avoid numerical issues
        dist = NegativeBinomial(total_count=total_count, probs=probs)
        return dist.log_prob(observations).sum(-1)

# Load model and likelihood
model = STVGPModel(inducing_points).to(device)
likelihood = NegativeBinomialLikelihood().to(device)

model.load_state_dict(torch.load("stvgp_model.pth"))
likelihood.load_state_dict(torch.load("stvgp_likelihood.pth"))

model.eval()
likelihood.eval()

# Load and prepare test data
print("Loading test data...")
df = pd.read_csv("~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv")
df_test = df[df['ground_truth'].isna()].copy()

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
batch_size = 1024
test_dataset = TensorDataset(test_x_scaled)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_means, all_stddevs = [], []

with torch.no_grad(), gpytorch.settings.fast_pred_var(), autocast():
    for (x_batch,) in tqdm(test_loader):
        x_batch = x_batch.to(device)
        preds = likelihood(model(x_batch))
        mean_batch = preds.mean.cpu().numpy()
        stddev_batch = preds.stddev.cpu().numpy()
        all_means.append(mean_batch)
        all_stddevs.append(stddev_batch)

predicted_counts = np.concatenate(all_means)
predicted_std = np.concatenate(all_stddevs)

df_test['predicted_counts'] = predicted_counts
df_test['predicted_std'] = predicted_std

# Save predictions
df_test[['bboxid', 'predicted_counts', 'predicted_std']].to_csv(
    '~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/predictions_st_vgp.csv', 
    index=False
)

print("Prediction saved.")


#############
# Prediction
print("Starting prediction...")
model.eval()
likelihood.eval()

test_x_np = df_test[['latitude', 'longitude', 'timestamp', 'max','min','precipitation',
                     'total_population','white_ratio','black_ratio','hh_median_income']].values

test_x_scaled =  = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32)
test_dataset = TensorDataset(test_x_scaled)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

all_means = []
all_stddevs = []

with torch.no_grad(), gpytorch.settings.fast_pred_var(), autocast():
    for (x_batch,) in tqdm(test_loader):
        x_batch = x_batch.to(device)
        latent_f = model(x_batch)
        latent_mean = latent_f.mean
        print("latent_mean stats:", latent_mean.min().item(), latent_mean.max().item(), torch.isnan(latent_mean).any().item())

        # Safeguard exp before it's passed into the likelihood
        if torch.isnan(latent_mean).any():
            raise ValueError("NaN detected in latent function mean before applying exp")
        
        preds = likelihood(latent_f)
        mean_batch = preds.mean.cpu().numpy()
        stddev_batch = preds.stddev.cpu().numpy()

        all_means.append(mean_batch)
        all_stddevs.append(stddev_batch)

predicted_counts = np.concatenate(all_means)
predicted_std = np.concatenate(all_stddevs)

df_test['predicted_counts'] = predicted_counts
df_test['predicted_std'] = predicted_std

# Save predictions with uncertainty
df_test[['bboxid', 'timestamp', 'predicted_counts', 'predicted_std']].to_csv(
    '~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/predictions_st_vgp.csv', 
    index=False
)