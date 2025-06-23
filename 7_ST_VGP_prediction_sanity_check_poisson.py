import torch
import gpytorch
import numpy as np
import pandas as pd
import joblib
import re
from tqdm import tqdm
import torch.nn.functional as F

# Load the necessary files
print("Loading saved artifacts...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('scaler_poisson.pkl')
inducing_points = torch.load('inducing_points_poisson.pt', map_location=device)

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
        #self.mean_module = gpytorch.means.ZeroMean()  # Use ZeroMean for stability
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


class PoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()
        # No parameters for vanilla Poisson

    def forward(self, function_samples, **kwargs):
        # The function_samples should be on log-scale
        rate = function_samples.exp()
        return torch.distributions.Poisson(rate)
    
    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean
        rate = mean.exp()
        dist = torch.distributions.Poisson(rate)
        return dist.log_prob(target)

# Instantiate and load trained parameters
model = STVGPModel(inducing_points.to(device)).to(device)
likelihood = PoissonLikelihood().to(device)
model.load_state_dict(torch.load('stvgp_model_poisson.pth', map_location=device))

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

# Sanity Check with training data
print("Sanity check with training data...")
df_test = df.dropna(subset=['ground_truth']) # actually this is the training data

# Prepare test features
spatial_coords = df_test[['latitude', 'longitude']].values
temporal_coords = df_test[['timestamp']].values
X_covariates = df_test[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

test_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
test_x = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32).to(device)

# Predict in batches (if test set is large)
print("Predicting...")
batch_size = 512
num_lik_samples = 100
test_pred_means = []
test_pred_lowers = []
test_pred_uppers = []


with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(num_lik_samples):
    for i in tqdm(range(0, test_x.size(0), batch_size)):
        x_batch = test_x[i:i+batch_size]         
        latent_dist = model(x_batch)
        pred_dist = likelihood(latent_dist)
        
        mean_pred = pred_dist.mean.cpu().numpy() 
        samples = pred_dist.sample((num_lik_samples,))
        samples_np = samples.cpu().numpy().reshape(-1, samples.shape[-1])  # shape: [1000*10, batch_size]
        
        lower_pred = np.percentile(samples_np, 2.5, axis=0)
        upper_pred = np.percentile(samples_np, 97.5, axis=0)

        test_pred_means.append(mean_pred)
        test_pred_lowers.append(lower_pred)
        test_pred_uppers.append(upper_pred)

# Concatenate batch predictions
test_pred_means = np.concatenate(test_pred_means)
test_pred_lowers = np.concatenate(test_pred_lowers)
test_pred_uppers = np.concatenate(test_pred_uppers)

# Attach results to test dataframe
df_test = df_test.reset_index(drop=True)
df_test['predicted_count_mean'] = test_pred_means
df_test['predicted_count_lower'] = test_pred_lowers
df_test['predicted_count_upper'] = test_pred_uppers

# Save results
df_test.to_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/sanity_check_results_poisson.csv', index=False)
print("Sanity check complete. Results saved to 'sanity_check_results_poisson.csv'.")
