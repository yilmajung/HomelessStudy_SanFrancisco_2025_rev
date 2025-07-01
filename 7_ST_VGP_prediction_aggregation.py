import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import torch.nn.functional as F
import re

# Load the necessary files
print("Loading saved artifacts...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('scaler_pois_constmean_ip700.joblib')
constant_mean = joblib.load('constant_mean_pois_ip700.pkl')
inducing_points = torch.load('inducing_points_pois_constmean_ip700.pt', map_location=device)

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
        rate = torch.nan_to_num(rate, nan=1e-6, posinf=1e6, neginf=1e-6)
        rate = rate.clamp(min=1e-6, max=1e6)  # Ensure rate is positive
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

df_test = df[df['ground_truth'].isna()]
# Sanity Check with training data
# print("Sanity check with training data...")
# df_test = df.dropna(subset=['ground_truth']) # actually this is the training data
# Small subset for testing
# df_test = df_test.sample(n=1000, random_state=42).reset_index(drop=True)

#=====================
df_training = df.dropna(subset=['ground_truth'])

# Extract coordinates and covariates
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_coords = df_training[['timestamp']].values
X_covariates = df_training[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']]
y_counts = df_training['ground_truth'].values

# Stack and scale
print("Preparing training tensors...")
t = np.hstack((spatial_coords, temporal_coords, X_covariates))
y = y_counts.astype(np.float32)
scaler = StandardScaler().fit(t)
x_scaled = scaler.transform(t).astype(np.float32)

# Prepare tensors
train_x = torch.tensor(x_scaled, dtype=torch.float32)
train_y = torch.tensor(y, dtype=torch.float32)
#=====================


# Instantiate and load trained parameters
model = STVGPModel(inducing_points.to(device), constant_mean=constant_mean).to(device)
likelihood = PoissonLikelihood().to(device)

model.load_state_dict(torch.load('stvgp_pois_constmean_ip700.pth', map_location=device))
likelihood.load_state_dict(torch.load('likelihood_pois_constmean_ip700.pth', map_location=device))

# debugging lines
for name, ker in [
    ("spatial",   model.spatial_kernel),
    ("temporal",  model.temporal_kernel),
    ("covariate", model.covariate_kernel),
]:
    ls = ker.base_kernel.lengthscale.detach().cpu().numpy()
    vs = ker.outputscale.detach().cpu().item()
    
    if ls.size == 1:
        print(f"{name} lengthscale = {ls.item():.4f}")
    else:
        print(f"{name} lengthscales = {ls.tolist()}")
    print(f"{name} variance    = {vs:.4f}")

saved = torch.load('stvgp_pois_constmean_ip700.pth', map_location=device)
missing, unexpected = model.load_state_dict(saved, strict=False)
print("MISSING:", missing)
print("UNEXPECTED:", unexpected)


model.eval()
likelihood.eval()

# Prepare test features
spatial_coords = df_test[['latitude', 'longitude']].values
temporal_coords = df_test[['timestamp']].values
X_covariates = df_test[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

test_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
test_x = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32).to(device)

# Predict in batches
print("Predicting...")
batch_size = 512
num_lik_samples = 500

test_loader = DataLoader(TensorDataset(test_x), batch_size=batch_size, shuffle=False, drop_last=False)

# z‐score for 95% CI
z95 = 1.96
z90 = 1.645

# Containers for per‐box summaries
pred_mean_Y      = []
pred_var_Y       = []
pred_rate_median = []
pred_rate_lower95    = []
pred_rate_upper95    = []
pred_rate_lower90 = []
pred_rate_upper90 = []

# Per‐box predictive mean & variance of Y, plus log‐normal rate CI
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for (x_batch,) in tqdm(test_loader, desc="Per‐box stats"):
        post = model(x_batch)          # MultivariateNormal over f
        m = post.mean               # shape (bsz,)
        v = post.variance           # shape (bsz,)
        #print(f"  m → min {m.min():.2f}, max {m.max():.2f},  v → min {v.min():.2f}, max {v.max():.2f}")
        m = m.clamp(min=-3.0, max=3.0)  
        v = v.clamp(max=4.0)
        s = v.sqrt()                # σ_f

        # E[Y] = E[e^f] = exp(m + ½ v)
        E_Y    = torch.exp(m + 0.5 * v)
        exp_v_minus1 = torch.expm1(v)
        Var_e_f = E_Y * E_Y * exp_v_minus1
        Var_Y   = E_Y + Var_e_f

        # log‐normal rate summaries
        rate_med  = torch.exp(m)
        rate_l95b = torch.exp(m - z95 * s)
        rate_u95b = torch.exp(m + z95 * s)
        rate_l90b = torch.exp(m - z90 * s)
        rate_u90b = torch.exp(m + z90 * s)

        # append (move to CPU numpy)
        pred_mean_Y.append(E_Y.cpu().numpy())
        pred_var_Y.append(Var_Y.cpu().numpy())
        pred_rate_median.append(rate_med.cpu().numpy())
        pred_rate_lower95.append(rate_l95b.cpu().numpy())
        pred_rate_upper95.append(rate_u95b.cpu().numpy())
        pred_rate_lower90.append(rate_l90b.cpu().numpy())
        pred_rate_upper90.append(rate_u90b.cpu().numpy())

# flatten and attach to df_test
df_test = df_test.reset_index(drop=True)
df_test['pred_mean_Y']      = np.concatenate(pred_mean_Y)
df_test['pred_var_Y']       = np.concatenate(pred_var_Y)
df_test['rate_median']      = np.concatenate(pred_rate_median)
df_test['rate_lower95']     = np.concatenate(pred_rate_lower95)
df_test['rate_upper95']     = np.concatenate(pred_rate_upper95)
df_test['rate_lower90']     = np.concatenate(pred_rate_lower90)
df_test['rate_upper90']     = np.concatenate(pred_rate_upper90)

pd.set_option('display.max_columns', None)
print(df_test.head())

# Monte Carlo aggregation to city‐daily totals
S         = 500
daily_out = []
n_days    = df_test['timestamp'].nunique()

for day, grp in tqdm(df_test.groupby('timestamp'),
                     total=n_days,
                     desc="Aggregating city totals"):
    # for each box in this day
    mu_Y    = grp['pred_mean_Y'].values    # E[Y_i], not needed for sampling
    var_Y   = grp['pred_var_Y'].values     # Var[Y_i], ditto
    med     = grp['rate_median'].values    # log‐normal medians
    l95     = grp['rate_lower95'].values
    u95     = grp['rate_upper95'].values
    l90     = grp['rate_lower90'].values
    u90     = grp['rate_upper90'].values
    nbox    = med.size

    # Reconstruct each box's lognormal rate parameters
    #   med = exp(mu_f)  →  mu_f = ln(med)
    #   l95 = exp(mu_f - z95·σ_f) → σ_f = (ln(med) - ln(l95)) / z95
    mu_f    = np.log(med)
    sigma_f = (np.log(med) - np.log(l95)) / z95

    # Sample latent f ∼ N(mu_f, sigma_f²)
    f_samps = np.random.normal(
        loc=mu_f[None, :],
        scale=sigma_f[None, :],
        size=(S, nbox)
    )
    # Convert to Poisson rates
    max_lambda = 1e3
    lam_samps = np.exp(f_samps)
    lam_safe = np.minimum(lam_samps, max_lambda)
    # Sample counts Y ∼ Poisson(lam)
    y_samps   = np.random.poisson(lam_safe)
    # Sum across boxes → city total per replicate
    city_samps = y_samps.sum(axis=1)       # shape (S,)

    # 2e) store daily summary
    daily_out.append({
        'timestamp':    day,
        'mean_total':   city_samps.mean(),
        'median_total': np.median(city_samps),
        'lower95':      np.percentile(city_samps, 2.5),
        'upper95':      np.percentile(city_samps, 97.5),
        'lower90':      np.percentile(city_samps, 5.0),
        'upper90':      np.percentile(city_samps, 95.0),
    })

# assemble final daily DataFrame
df_daily = (
    pd.DataFrame(daily_out)
      .sort_values('timestamp')
      .reset_index(drop=True)
)
print(df_daily.head())

# save results
df_daily.to_csv('st_vgp_pois_constmean_ip700_daily_totals.csv', index=False)
df_test.to_csv('st_vgp_pois_constmean_ip700_test_predictions.csv', index=False)
