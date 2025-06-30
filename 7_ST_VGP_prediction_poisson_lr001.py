import torch
import gpytorch
import numpy as np
import pandas as pd
import joblib
import re
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Load the necessary files
print("Loading saved artifacts...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('scaler_pois_constmean.joblib')
constant_mean = joblib.load('constant_mean_pois.pkl')
inducing_points = torch.load('inducing_points_pois_constmean.pt', map_location=device)

# Define the model and likelihood classes exactly as in training
# Define ST-VGP model

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
        cov_dim = inducing_points.shape[1] - 3  # 2 spatial + 1 temporal
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=cov_dim))
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
        return torch.distributions.Poisson(rate)
    
    def expected_log_prob(self, target, function_dist, **kwargs):
        mean = function_dist.mean
        rate = mean.exp()
        dist = torch.distributions.Poisson(rate)
        return dist.log_prob(target)


# Load and preprocess the dataset (same as training)
print("Loading and prepping test data...")
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')

df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# # Sanity Check with training data
# print("Sanity check with training data...")
df_test = df.dropna(subset=['ground_truth']) # actually this is the training data
# Small subset for testing
# df_test = df[df['ground_truth'].isna()]
# df_test = df_test.sample(n=1000, random_state=42).reset_index(drop=True)


# Instantiate and load trained parameters
model = STVGPModel(inducing_points.to(device), constant_mean=constant_mean).to(device)
likelihood = PoissonLikelihood().to(device)

model.load_state_dict(torch.load('stvgp_pois_constmean.pth', map_location=device))
likelihood.load_state_dict(torch.load('likelihood_pois_constmean.pth', map_location=device))

model.eval()
likelihood.eval()

# Prepare test features
spatial_coords = df_test[['latitude', 'longitude']].values
temporal_coords = df_test[['timestamp']].values
X_covariates = df_test[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

test_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
test_x = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32).to(device)

# Predict in batches (if test set is large)
print("Predicting...")
batch_size = 512
num_lik_samples = 200

test_loader = DataLoader(TensorDataset(test_x), batch_size=batch_size, shuffle=False, drop_last=False)

pred_means = []
pred_lower95, pred_upper95, pred_lower90, pred_upper90 = [], [], [], []

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for x_batch, in tqdm(test_loader):
        latent_dist = model(x_batch)
        pred_dist = likelihood(latent_dist)
        
        samples = pred_dist.sample(torch.Size([num_lik_samples]))
        samples_np = samples.cpu().numpy()

        bs = samples_np.shape[-1]
        flat = samples_np.reshape(-1, bs)

        # debugging: print sample shapes
        print(f"flatten samples_np shape: {flat.shape}")

        batch_mean = flat.mean(axis=0)
        batch_median = np.median(flat, axis=0)
        batch_lower95 = np.percentile(flat, 2.5, axis=0)
        batch_upper95 = np.percentile(flat, 97.5, axis=0)
        batch_lower90 = np.percentile(flat, 5.0, axis=0)
        batch_upper90 = np.percentile(flat, 95.0, axis=0)

        # debugging: print batch shapes
        print(f"batch_mean shape: {batch_mean.shape}")

        pred_means.append(batch_mean)
        pred_lower95.append(batch_lower95)
        pred_upper95.append(batch_upper95)
        pred_lower90.append(batch_lower90)
        pred_upper90.append(batch_upper90)


print("Sample shapes in pred_means:")
for i in [0, -1]:
    arr = pred_means[i]
    print(f"  [{i}] type={type(arr)}  shape={getattr(arr, 'shape', None)}")

# Turn each list of arrays into one long 1D array
pred_means      = np.concatenate(pred_means)
pred_lower95    = np.concatenate(pred_lower95)
pred_upper95    = np.concatenate(pred_upper95)
pred_lower90    = np.concatenate(pred_lower90)
pred_upper90    = np.concatenate(pred_upper90)

# Sanity check
assert len(pred_means)     == len(df_test)
assert len(pred_lower95)   == len(df_test)
assert len(pred_upper95)   == len(df_test)
assert len(pred_lower90)   == len(df_test)
assert len(pred_upper90)   == len(df_test)

print('pred_means: ', pred_means[:10])
print('pred_lower95: ', pred_lower95[:10])
print('pred_upper95: ', pred_upper95[:10])

# print("total preds:", test_pred_mean.shape[0], "expected:", test_x.size(0))

# # Validate dimensions explicitly before assignment:
# assert len(test_pred_mean) == len(df_test), f"Mismatch: predictions ({len(test_pred_mean)}) vs test data ({len(df_test)})"

# Attach results to test dataframe
df_test = df_test.reset_index(drop=True)
df_test['predicted_count_mean'] = pred_means
df_test['predicted_count_lower'] = pred_lower95
df_test['predicted_count_upper'] = pred_upper95
df_test['predicted_count_lower_90'] = pred_lower90
df_test['predicted_count_upper_90'] = pred_upper90


# Save results
df_test.to_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/prediction_poisson_lr001_sanitycheck.csv', index=False)
print("Prediction complete. Results saved to 'prediction_poisson_lr001.csv'.")
