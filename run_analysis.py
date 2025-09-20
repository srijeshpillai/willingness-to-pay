import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from sklearn.preprocessing import StandardScaler
# Optional, for professional-looking plots. If you don't have it, comment this line out.
import scienceplots 

# Use a professional plotting style
plt.style.use(['science', 'notebook', 'grid'])

print("--- Bayesian Conjoint Analysis: iPhone Feature Valuation (V6 - Final, Revenue Optimization) ---")

# ==============================================================================
# PART 1: GROUND TRUTH
# ==============================================================================
print("\n[Step 1/5] Defining the 'Ground Truth' Market...")
TRUE_WTP = {'storage_256gb': 100.0, 'storage_512gb': 250.0, 'camera_pro': 200.0, 'material_titanium': 80.0}
TRUE_WTP_SD = {'storage_256gb': 20.0, 'storage_512gb': 40.0, 'camera_pro': 50.0, 'material_titanium': 30.0}
print("Ground truth Willingness-to-Pay (WTP) defined.")

# ==============================================================================
# PART 2: SIMULATING THE SURVEY
# ==============================================================================
print("\n[Step 2/5] Simulating the Choice-Based Conjoint Survey...")
N_RESPONDENTS, N_QUESTIONS_PER_RESPONDENT = 300, 20
respondent_wtps = [{k: np.random.normal(v, TRUE_WTP_SD[k]) for k, v in TRUE_WTP.items()} for _ in range(N_RESPONDENTS)]
survey_data = []
feature_levels = {'storage': [128, 256, 512], 'camera': ['standard', 'pro'],
                  'material': ['aluminum', 'titanium'], 'price': [799, 899, 999, 1099, 1199]}
for resp_id in range(N_RESPONDENTS):
    for q_id in range(N_QUESTIONS_PER_RESPONDENT):
        profile_A = {feat: np.random.choice(levels) for feat, levels in feature_levels.items()}
        profile_B = {feat: np.random.choice(levels) for feat, levels in feature_levels.items()}
        while profile_A == profile_B:
            profile_B = {feat: np.random.choice(levels) for feat, levels in feature_levels.items()}
        def calculate_dollar_utility(profile, wtps):
            utility = -profile['price']
            if profile['storage'] == 256: utility += wtps['storage_256gb']
            if profile['storage'] == 512: utility += wtps['storage_512gb']
            if profile['camera'] == 'pro': utility += wtps['camera_pro']
            if profile['material'] == 'titanium': utility += wtps['material_titanium']
            return utility
        utility_A = calculate_dollar_utility(profile_A, respondent_wtps[resp_id])
        utility_B = calculate_dollar_utility(profile_B, respondent_wtps[resp_id])
        utility_diff = utility_A - utility_B
        prob_A = 1 / (1 + np.exp(-utility_diff))
        choice = 1 if np.random.rand() < prob_A else 0
        row = {'resp_id': resp_id, 'choice': choice}
        for p, d in [('A', profile_A), ('B', profile_B)]: row.update({f'{k}_{p}': v for k, v in d.items()})
        survey_data.append(row)
df = pd.DataFrame(survey_data)
print(f"Survey generated with {len(df)} choices.")

# --- Prepare data for the model ---
df['price_diff'] = df['price_A'] - df['price_B']
df['storage_256_diff'] = (df['storage_A'] == 256).astype(int) - (df['storage_B'] == 256).astype(int)
df['storage_512_diff'] = (df['storage_A'] == 512).astype(int) - (df['storage_B'] == 512).astype(int)
df['camera_pro_diff'] = (df['camera_A'] == 'pro').astype(int) - (df['camera_B'] == 'pro').astype(int)
df['material_titanium_diff'] = (df['material_A'] == 'titanium').astype(int) - (df['material_B'] == 'titanium').astype(int)

predictors = ['price_diff', 'storage_256_diff', 'storage_512_diff', 'camera_pro_diff', 'material_titanium_diff']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[predictors] = scaler.fit_transform(df[predictors])
print("Predictor variables have been standardized for model stability.")

# ==============================================================================
# PART 3: BUILDING THE UPGRADED HIERARCHICAL MODEL
# ==============================================================================
print("\n[Step 3/5] Building and Running the Upgraded Bayesian Model...")
respondent_idx, respondents = pd.factorize(df_scaled['resp_id'])
coords = {"respondent": respondents, "feature": predictors}

with pm.Model(coords=coords) as final_conjoint_model:
    X = df_scaled[predictors].values
    
    # Priors for population-level means
    mu_beta = pm.Normal('mu_beta', mu=0, sigma=2, dims="feature")
    
    # *** UPGRADE: NON-CENTERED PARAMETERIZATION FOR BETTER STABILITY ***
    # This is a best practice to reduce divergences in hierarchical models.
    # We model the *offset* from the mean, which is easier for the sampler.
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=1, dims="feature")
    offset = pm.Normal('offset', mu=0, sigma=1, dims=("respondent", "feature"))
    beta = pm.Deterministic('beta', mu_beta + offset * sigma_beta, dims=("respondent", "feature"))
    
    beta_per_obs = beta[respondent_idx, :]
    utility = pm.math.sum(beta_per_obs * X, axis=1)
    
    p_choice = pm.math.sigmoid(utility)
    y_obs = pm.Bernoulli('y_obs', p=p_choice, observed=df_scaled['choice'].values)

    trace = pm.sample(2000, tune=2000, cores=1, target_accept=0.95) # Slightly higher target_accept

print("Model fitting complete. Checking for divergences...")
divergences = trace.sample_stats.diverging.sum()
print(f"Number of divergences: {divergences.item()}")
if divergences > 0:
    print("Warning: Divergences detected. The model may have had issues.")
else:
    print("Success! No divergences detected. The model has converged successfully.")

# ==============================================================================
# PART 4: ANALYZING AND "UN-SCALING" THE RESULTS
# ==============================================================================
print("\n[Step 4/5] Un-scaling results and generating individual WTP plots...")

mu_betas = trace.posterior['mu_beta']
price_beta = mu_betas.sel(feature="price_diff")
price_std = scaler.scale_[0]

def calculate_wtp(feature_name):
    feature_beta = mu_betas.sel(feature=feature_name)
    feature_idx = predictors.index(feature_name)
    feature_std = scaler.scale_[feature_idx]
    return - (feature_beta / price_beta) * (price_std / feature_std)

wtp_storage_256 = calculate_wtp("storage_256_diff")
wtp_storage_512 = calculate_wtp("storage_512_diff")
wtp_camera_pro = calculate_wtp("camera_pro_diff")
wtp_material_titanium = calculate_wtp("material_titanium_diff")

# *** NEW: GENERATE FOUR SEPARATE PLOTS ***
def plot_and_save_wtp(data, title, true_value, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_posterior(data, ax=ax, hdi_prob=0.95, ref_val=true_value, ref_val_color='red')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Willingness-to-Pay ($)', fontsize=12)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to '{filename}'")
    plt.close(fig) # Close the figure to avoid displaying it in the console

plot_and_save_wtp(wtp_storage_256, 'WTP for 256GB vs 128GB Storage', TRUE_WTP['storage_256gb'], 'wtp_storage_256.png')
plot_and_save_wtp(wtp_storage_512, 'WTP for 512GB vs 128GB Storage', TRUE_WTP['storage_512gb'], 'wtp_storage_512.png')
plot_and_save_wtp(wtp_camera_pro, 'WTP for "Pro" vs. Standard Camera', TRUE_WTP['camera_pro'], 'wtp_camera_pro.png')
plot_and_save_wtp(wtp_material_titanium, 'WTP for Titanium vs. Aluminum', TRUE_WTP['material_titanium'], 'wtp_titanium.png')

# ==============================================================================
# PART 5: REVENUE OPTIMIZATION SIMULATION
# ==============================================================================
print("\n[Step 5/5] Running Revenue Optimization and generating policy plot...")

# Define the "Pro" bundle: Pro Camera + Titanium Frame
wtp_pro_bundle = wtp_camera_pro + wtp_material_titanium

# Define a baseline phone and a range of prices for the "Pro" model
BASE_PHONE_PRICE = 799
pro_price_points = np.linspace(999, 1299, 31) # e.g., $999, $1009, $1019, ...
expected_revenues = []

# For each potential price point...
for price in pro_price_points:
    # The utility of the Pro phone is the WTP for the bundle minus the price premium
    # This is an array of thousands of samples, representing our uncertainty
    utility_of_pro_vs_base = wtp_pro_bundle - (price - BASE_PHONE_PRICE)
    
    # Convert utility to purchase probability using the sigmoid function
    purchase_probability = 1 / (1 + np.exp(-utility_of_pro_vs_base / 100)) # Scaled for stability
    
    # Revenue is Price * Probability. This gives us a full distribution of possible revenues at this price.
    revenue_at_this_price = price * purchase_probability
    expected_revenues.append(revenue_at_this_price.values.flatten())

# Create the final revenue plot
plt.figure(figsize=(10, 6))
# Use violin plots to show the full distribution of revenue at each price point
plt.violinplot(dataset=expected_revenues, positions=pro_price_points, widths=15, showmeans=True)
# Find the price with the highest mean expected revenue
mean_revs = [np.mean(rev) for rev in expected_revenues]
optimal_price_idx = np.argmax(mean_revs)
optimal_price = pro_price_points[optimal_price_idx]
optimal_revenue = mean_revs[optimal_price_idx]
plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.0f}')
plt.title('Distribution of Expected Revenue vs. Price for "iPhone Pro" Bundle')
plt.xlabel('Price Point ($)')
plt.ylabel('Expected Revenue per Customer ($)')
plt.legend()
plt.tight_layout()
plt.savefig('revenue_optimization_plot.png', dpi=300)
print("Saved revenue optimization plot to 'revenue_optimization_plot.png'")

print("\n--- All Technical Implementations Complete ---")