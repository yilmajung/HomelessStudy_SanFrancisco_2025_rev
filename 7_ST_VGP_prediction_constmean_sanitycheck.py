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
scaler = joblib.load('scaler_nb_constmean.joblib')
constant_mean = joblib.load('constant_mean_nb.pkl')
inducing_points = torch.load('inducing_points_constmean.pt', map_location=device)

# Define the model and likelihood classes exactly as in training
# Define ST-VGP model
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
        cov_dim = inducing_points.size(-1) - 3
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=cov_dim))
        self.const_kernel     = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ConstantKernel())

    def forward(self, x):
        s, t, c = x[:, :2], x[:, 2:3], x[:, 3:]
        # Constant mean uses covariates to determine batch shape
        mean_x = self.mean_module(c)
        mean_x = mean_x.clamp(min=-10.0, max=10.0)
        Ks = self.spatial_kernel(s)
        Kt = self.temporal_kernel(t)
        Kc = self.covariate_kernel(c)
        Kconst = self.const_kernel(s)

        covar = Ks * Kt * Kc + Ks + Kt + Kc + Kconst
        covar = covar + torch.eye(covar.size(-1), device=x.device) * 1e-3  # jitter
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class NegativeBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
     def __init__(self, init_dispersion=1.0):
         super().__init__()
         raw_dispersion = torch.tensor(
             np.log(np.exp(init_dispersion) - 1), 
             dtype=torch.float32
         )
         self.register_parameter(
             name="raw_log_dispersion", 
             parameter=torch.nn.Parameter(raw_dispersion)
         )

     @property
     def dispersion(self):
         return F.softplus(self.raw_log_dispersion) + 1e-5

     def forward(self, function_samples, **kwargs):
         # tighter clamp to prevent extreme μ
         log_mu = function_samples.clamp(min=-3, max=3)     # μ ≤ e³≈20
         mu     = log_mu.exp().clamp(min=1e-3, max=50)       # hard-cap at 50
         r      = self.dispersion
         probs  = r / (r + mu)
        
         # replace NaNs and clamp to (epsilon, 1-epsilon)
         probs = torch.nan_to_num(probs, nan=0.5, posinf=1-1e-6, neginf=1e-6)
         probs = probs.clamp(min=1e-6, max=1-1e-6)
         return torch.distributions.NegativeBinomial(
             total_count=r.expand_as(mu), probs=probs
             )

     def expected_log_prob(self, target, function_dist, **kwargs):
         log_mu = function_dist.mean.clamp(min=-3, max=3)
         mu     = log_mu.exp().clamp(min=1e-3, max=50)
         r      = self.dispersion
         probs  = r / (r + mu)
         probs = torch.nan_to_num(probs, nan=0.5, posinf=1-1e-6, neginf=1e-6)
         probs = probs.clamp(min=1e-6, max=1-1e-6)
         dist   = torch.distributions.NegativeBinomial(
             total_count=r.expand_as(probs), probs=probs
         )
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

# Sanity Check with training data
print("Sanity check with training data...")
df_test = df.dropna(subset=['ground_truth']) # actually this is the training data
# print("Sanity check with training data...")
# df_test = df.dropna(subset=['ground_truth']) # actually this is the training data
# df_test = df[df['ground_truth'].isna()]
# # Small subset for testing
# df_test = df_test.sample(n=1000, random_state=42).reset_index(drop=True)

# Instantiate and load trained parameters
model = STVGPModel(inducing_points.to(device), constant_mean).to(device)
likelihood = NegativeBinomialLikelihood().to(device)
model.load_state_dict(torch.load('stvgp_constmean.pth', map_location=device))
likelihood.load_state_dict(torch.load('likelihood_constmean.pth', map_location=device))

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
num_lik_samples = 300

test_loader = DataLoader(TensorDataset(test_x), batch_size=batch_size, shuffle=False, drop_last=False)
print("Test set size:", len(test_x))

N = test_x.size(0)
mean_all = np.empty(N, dtype=np.float32)
lower_95_all = np.empty(N, dtype=np.float32)
upper_95_all = np.empty(N, dtype=np.float32)
lower_90_all = np.empty(N, dtype=np.float32)
upper_90_all = np.empty(N, dtype=np.float32)

offset = 0
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i, (x_batch,) in tqdm(enumerate(test_loader)):
        x_batch = x_batch.to(device)
        B_i = x_batch.size(0)

        # 1) GP posterior → analytic mean
        f_dist = model(x_batch)
        mu     = f_dist.mean.clamp(-3,3).exp().clamp(1e-3,50)
        mean_all[offset:offset+B_i] = mu.cpu().numpy()

        # 2) Jointly sample S latent draws
        f_samps = f_dist.rsample(sample_shape=torch.Size([num_lik_samples]))  # (S, B)
        log_mu_samps = f_samps.clamp(-3,3)
        mu_samps     = log_mu_samps.exp().clamp(1e-3, 50)

        # 3) NB parameters
        r = likelihood.dispersion
        probs = r / (r + mu_samps)
        probs = probs.clamp(1e-6, 1-1e-6)

        # 4) Draw from torch.distributions.NegativeBinomial
        nb = torch.distributions.NegativeBinomial(
            total_count=r.expand_as(mu_samps),
            probs=probs
        )
        y_samps = nb.sample()  # (S, B)
        y_np    = y_samps.cpu().numpy()

        lower_95_all[offset:offset+B_i] = np.percentile(y_np, 2.5, axis=0)
        upper_95_all[offset:offset+B_i] = np.percentile(y_np,97.5, axis=0)
        lower_90_all[offset:offset+B_i] = np.percentile(y_np, 5.0, axis=0)
        upper_90_all[offset:offset+B_i] = np.percentile(y_np,95.0, axis=0)

        offset += B_i

assert offset == N

# offset = 0
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     for i, (x_batch,) in enumerate(test_loader):
#         x_batch = x_batch.to(device)
#         B_i = x_batch.size(0)
#         f_dist = model(x_batch)
#         mu = f_dist.mean.clamp(min=-3, max=3).exp().clamp(min=1e-3, max=50)  # μ ≤ e³≈20
#         mean_pred = mu.cpu().numpy()
#         with gpytorch.settings.num_likelihood_samples(300):
#             p_dist = likelihood(f_dist)
#             samples = p_dist.sample((num_lik_samples,)).cpu().numpy()
#         lower_95 = np.percentile(samples, 2.5, axis=0)
#         upper_95 = np.percentile(samples,97.5, axis=0)
#         lower_90 = np.percentile(samples, 5.0, axis=0)
#         upper_90 = np.percentile(samples,95.0, axis=0)
#         # debug check
#         print(f"Batch {i}: B_i={B_i}, mean.shape = {mean_pred.shape}, lower_95.shape = {lower_95.shape}")
#         test_pred_means[offset: offset+B_i] = mean_pred
#         test_pred_lowers[offset: offset+B_i] = lower_95
#         test_pred_uppers[offset: offset+B_i] = upper_95
#         test_pred_lowers_90[offset: offset+B_i] = lower_90
#         test_pred_uppers_90[offset: offset+B_i] = upper_90
#         offset += B_i

# # Final sanity check
# assert offset == N, f"Only wrote {offset} values but expected {N}"

# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(num_lik_samples):

#     for i in tqdm(range(0, test_x.size(0), batch_size)):
#         x_batch = test_x[i:i+batch_size]
#         latent = model(x_batch)
#         pred_dist = likelihood(latent)
        
#         # 1) analytic mean
#         mean_pred = pred_dist.mean.cpu().numpy()
#         #mean_pred_avg = mean_pred.mean(axis=0)        

#         # 2) empirical quantiles
#         samples = pred_dist.sample((num_lik_samples,))    # [S, B]
#         print("samples.shape:", samples.shape)  # should be [num_lik_samples, batch_size]
#         print("pred_dist.mean.shape:", pred_dist.mean.shape)  # should be [batch_size]

#         samples_np = samples.cpu().numpy()
#         lower_95 = np.percentile(samples_np, 2.5, axis=0)
#         upper_95 = np.percentile(samples_np,97.5, axis=0)
#         lower_90 = np.percentile(samples_np, 5.0, axis=0)
#         upper_90 = np.percentile(samples_np,95.0, axis=0)

#         test_pred_means.append(mean_pred)
#         test_pred_lowers.append(lower_95)
#         test_pred_uppers.append(upper_95)
#         test_pred_lowers_90.append(lower_90)
#         test_pred_uppers_90.append(upper_90)

# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(num_lik_samples):
#     for i in tqdm(range(0, test_x.size(0), batch_size)):
#         x_batch = test_x[i:i+batch_size]         
#         latent_dist = model(x_batch)
#         pred_dist = likelihood(latent_dist)
#         mean_pred = pred_dist.mean.mean(dim=0).cpu().numpy()
#         samples = pred_dist.sample((num_lik_samples,))
#         samples_np = samples.cpu().numpy().reshape(-1, samples.size(-1))  
        
#         lower_pred = np.percentile(samples_np, 2.5, axis=0)
#         upper_pred = np.percentile(samples_np, 97.5, axis=0)

#         lower_pred_90 = np.percentile(samples_np, 5.0, axis=0)
#         upper_pred_90 = np.percentile(samples_np, 95.0, axis=0)

#         # print(f"mean_pred shape: {mean_pred.shape}")
#         # print(f"lower_pred shape: {lower_pred.shape}")
#         # print(f"upper_pred shape: {upper_pred.shape}")

#         test_pred_means.append(mean_pred)
#         test_pred_lowers.append(lower_pred)
#         test_pred_uppers.append(upper_pred)
#         test_pred_lowers_90.append(lower_pred_90)
#         test_pred_uppers_90.append(upper_pred_90)

# Concatenate batch predictions
# test_pred_mean = np.concatenate(test_pred_means)
# test_pred_lower = np.concatenate(test_pred_lowers)
# test_pred_upper = np.concatenate(test_pred_uppers)
# test_pred_lower_90 = np.concatenate(test_pred_lowers_90)
# test_pred_upper_90 = np.concatenate(test_pred_uppers_90)

print('mean: ', mean_all[:20])
print('lower bound: ', lower_95_all[:20])
print('upper bound: ', upper_95_all[:20])

# print("total preds:", test_pred_mean.shape[0], "expected:", test_x.size(0))

# # Validate dimensions explicitly before assignment:
# assert len(test_pred_mean) == len(df_test), f"Mismatch: predictions ({len(test_pred_mean)}) vs test data ({len(df_test)})"

# # Attach results to test dataframe
# df_test = df_test.reset_index(drop=True)
# df_test['predicted_count_mean'] = mean_all
# df_test['predicted_count_lower'] = lower_95_all
# df_test['predicted_count_upper'] = upper_95_all
# df_test['predicted_count_lower_90'] = lower_90_all
# df_test['predicted_count_upper_90'] = upper_90_all


# # Save results
# df_test.to_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/prediction_nb_constmean.csv', index=False)
# print("Prediction complete. Results saved to 'prediction_nb_constmean.csv'.")
