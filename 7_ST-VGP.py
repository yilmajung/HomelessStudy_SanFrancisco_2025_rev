import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/df_cleaned_20250617.csv')

# Create latitude and longitude columns
import re
df['latitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
df['longitude'] = df['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
df['latitude'] = df['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
df['longitude'] = df['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))

# Convert timestamp (datetime) to number
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Separate training data
df_training = df.dropna(subset=['ground_truth'])
df_test = df[df['ground_truth'].isna()]

# Extract coordinates
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_coords = df_training[['timestamp']].values
X_covariates = df_training[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']]
y_counts = df_training['ground_truth'].values

# Inducing Points Strategy (Density-based)
bbox_counts = df.groupby('bboxid')['ground_truth'].mean().reset_index()
top_bbox_ids = bbox_counts.nlargest(638, 'ground_truth')['bboxid'].values # Top 638 bboxes which has larger than 1 on average

# Select inducing points coordinates from top boxes
inducing_df = df_training[df_training['bboxid'].isin(top_bbox_ids)].drop_duplicates(subset=['bboxid'])
Z_spatial = inducing_df[['longitude','latitude']].values
Z_temporal = inducing_df[['timestamp']].values
Z_covariates = inducing_df[['max','min','precipitation','total_population','white_ratio','black_ratio','hh_median_income']].values

# Define ST-VGP model
class STVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(STVGPModel, self).__init__(variational_strategy)

        # Kernel definition (separable spatial-temporal-covariate)
        self.spatial_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))

        self.mean_module = gpytorch.means.LinearMean(input_size=7)

    def forward(self, x):
        spatial_x = x[:, :2]
        temporal_x = x[:, 2:3]
        covariate_x = x[:, 3:]

        mean_x = self.mean_module(covariate_x)
        covar_x = self.spatial_kernel(spatial_x) * self.temporal_kernel(temporal_x) * self.covariate_kernel(covariate_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Prepare tensors for GPyTorch
train_x = torch.tensor(np.hstack((spatial_coords, temporal_coords, X_covariates)), dtype=torch.float32)
train_y = torch.tensor(y_counts, dtype=torch.float32)

inducing_points = torch.tensor(np.hstack((Z_spatial, np.full((Z_spatial.shape[0], 1), Z_temporal), Z_covariates)), dtype=torch.float32)

print(train_x.shape)
print(train_y.shape)

# Model Training (Negative Binomial Likelihood)
from torch.distributions import NegativeBinomial

class NegativeBinomialLikelihood(gpytorch.likelihoods.Likelihood):
    def __init__(self, dispersion=1.0):
        super().__init__()
        self.register_parameter(
            name="raw_dispersion",
            parameter=torch.nn.Parameter(torch.tensor(dispersion))
        )
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
    
likelihood = NegativeBinomialLikelihood(dispersion=1.0)

model = STVGPModel(inducing_points)

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # include model parameters
    {'params': likelihood.parameters()},  # include likelihood parameters
    ], lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

training_iterations = 500
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if (i+1) % 50 == 0:
        print(f"Iteration {i+1}/{training_iterations}: Loss = {loss.item():.3f}")
    optimizer.step()

# Prediction
model.eval()
likelihood.eval()

test_spatial_coords = df_test[['latitude', 'longitude']].values
test_temporal_coords = df_test[['timestamp']].values
test_X_covariates = df_test[['max', 'min', 'precipitation', 'total_population', 'white_ratio', 'black_ratio', 'hh_median_income']].values

test_x = torch.tensor(np.hstack((test_spatial_coords, test_temporal_coords, test_X_covariates)), dtype=torch.float32)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictive_dist = likelihood(model(test_x))
    predictions = predictive_dist.mean.numpy()

# Add predictions to the test DataFrame
df_test['predicted_counts'] = predictions

# Save predictions to CSV
df_test[['bboxid', 'predicted_counts']].to_csv('~/HomelessStudy_SanFrancisco_2025_rev_ISTServer/predictions_st_vgp.csv', index=False)