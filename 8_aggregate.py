# Build the full spatio‐temporal grid of size (B*T, D)
B = spatial_coords.shape[0]
T = len(unique_days)
# tile spatial & covariates, repeat times
X_sp = np.repeat(spatial_coords, T, axis=0)             # (B*T,2)
X_tm = np.tile(unique_days, B).reshape(-1,1)            # (B*T,1)
X_cv = np.repeat(covariates, T, axis=0)                 # (B*T,C)
X_np = np.hstack([X_sp, X_tm, X_cv]).astype(np.float32) # (B*T,3+C)
X_st = torch.tensor(scaler.transform(X_np), device=device)

# 2) Joint GP + NB sampling
N_SAMPLES = 300
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    f_dist = model(X_st)                                 # MVN over B*T dims
    # sample latent f ∼ MVN
    f_samps = f_dist.rsample(sample_shape=torch.Size([N_SAMPLES]))  # (S, B*T)
    # map to NB parameters
    log_mu = f_samps.clamp(-3,3)
    mu     = log_mu.exp().clamp(1e-3,50)
    r      = likelihood.dispersion
    probs  = r / (r + mu)
    nb     = torch.distributions.NegativeBinomial(total_count=r, probs=probs)
    y_samps = nb.sample()                               # (S, B*T)

# 3) Reshape and sum per day
y_np = y_samps.cpu().numpy().reshape(N_SAMPLES, B, T)   # (S, B, T)
city_samps = y_np.sum(axis=1)                          # (S, T)

# 4) Build DataFrame
rows = []
for ti, day in enumerate(unique_days):
    vals = city_samps[:, ti]
    rows.append({
      "day": day,
      "mean": np.mean(vals),
      "lo95": np.percentile(vals,2.5),
      "hi95": np.percentile(vals,97.5)
      "lo90": np.percentile(vals,5.0),
      "hi90": np.percentile(vals,95.0)
    })
df_city = pd.DataFrame(rows)
df_city["date"] = pd.to_datetime(df_city["day"], unit="s")
