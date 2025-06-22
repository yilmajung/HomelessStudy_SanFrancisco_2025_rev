import torch
import gpytorch
import numpy as np
import pandas as pd
import joblib
import re
from tqdm import tqdm

# Load the necessary files
print("Loading saved artifacts...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('scaler.pkl')
inducing_points = torch.load('inducing_points.pt', map_location=device)

# Define the model and likelihood classes exactly as in training
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
        self.const_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel())

    def forward(self, x):
        spatial_x = x[:, :2]
        temporal_x = x[:, 2:3]
        covariate_x = x[:, 3:]
        mean_x = self.mean_module(covariate_x)
        mean_x = mean_x.clamp(min=-10.0, max=10.0)
        Ks = self.spatial_kernel(spatial_x)
        Kt = self.temporal_kernel(temporal_x)
        Kc = self.covariate_kernel(covariate_x)
        Kconst = self.const_kernel(spatial_x)
        covar_x = Ks * Kt * Kc + Ks + Kt + Kc + Kconst
        covar_x = covar_x + torch.eye(covar_x.size(-1), device=x.device) * 1e-3
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class StableNegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, init_dispersion=1.0):
        super().__init__()
        raw_disp = torch.tensor(np.log(np.exp(init_dispersion) - 1), dtype=torch.float32)
        self.register_parameter(name="raw_log_dispersion", parameter=torch.nn.Parameter(raw_disp))

    @property
    def dispersion(self):
        return torch.nn.functional.softplus(self.raw_log_dispersion) + 1e-5

    def forward(self, function_samples, **kwargs):
        function_samples = function_samples.clamp(min=-10, max=10)
        mu = function_samples.exp().clamp(min=1e-3, max=1e3)
        r = self.dispersion
        logits = torch.log(mu + 1e-6) - torch.log(r + 1e-6)
        return torch.distributions.NegativeBinomial(total_count=r.expand_as(logits), logits=logits)
    
    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean.clamp(min=-10, max=10)
        mu = mean.exp().clamp(min=1e-3, max=1e3)
        r = self.dispersion
        logits = torch.log(mu + 1e-6) - torch.log(r + 1e-6)
        dist = torch.distributions.NegativeBinomial(total_count=r.expand_as(logits), logits=logits)
        return dist.log_prob(target)

# Instantiate and load trained parameters
model = STVGPModel(inducing_points.to(device)).to(device)
likelihood = StableNegativeBinomialLikelihood().to(device)
model.load_state_dict(torch.load('stvgp_model.pth', map_location=device))
likelihood.load_state_dict(torch.load('stvgp_likelihood.pth', map_location=device))

model.eval()
likelihood.eval()

# Load and preprocess the dataset (same as training)
print("Loading and prepping test data...")
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')
df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

df_test = df[df['ground_truth'].isna()]

# Prepare test features
spatial_coords = df_test[['latitude', 'longitude']].values
temporal_coords = df_test[['timestamp']].values
X_covariates = df_test[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

test_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
test_x = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32).to(device)

# Predict in batches (if test set is large)
print("Predicting...")
batch_size = 512
test_pred_means = []
test_pred_lowers = []
test_pred_uppers = []

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in tqdm(range(0, test_x.size(0), batch_size)):
        x_batch = test_x[i:i+batch_size]
        latent_dist = model(x_batch)
        pred_dist = likelihood(latent_dist)
        # Get predictive mean and quantiles
        mean_pred = pred_dist.mean.cpu().numpy()
        # 95% interval via sampling
        samples = pred_dist.sample((1000,))  # shape: [1000, batch_size]
        print(samples.shape)
        lower_pred = np.percentile(samples.cpu().numpy(), 2.5, axis=0)
        upper_pred = np.percentile(samples.cpu().numpy(), 97.5, axis=0)
        print(mean_pred.shape, lower_pred.shape, upper_pred.shape)
        test_pred_means.append(mean_pred)
        test_pred_lowers.append(lower_pred)
        test_pred_uppers.append(upper_pred)


# Concatenate batch predictions
print([arr.shape for arr in test_pred_means])
test_pred_mean = np.concatenate(test_pred_means)
test_pred_lower = np.concatenate(test_pred_lowers)
test_pred_upper = np.concatenate(test_pred_uppers)

# Attach results to test dataframe
df_test = df_test.reset_index(drop=True)
df_test['predicted_count_mean'] = test_pred_mean
df_test['predicted_count_lower'] = test_pred_lower
df_test['predicted_count_upper'] = test_pred_upper

# Save results
df_test.to_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/homeless_tent_test_predictions.csv', index=False)
print("Prediction complete. Results saved to 'homeless_tent_test_predictions.csv'.")
