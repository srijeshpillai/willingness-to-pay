"""
@file run_analysis.py
@brief End-to-end Bayesian Hierarchical Conjoint Analysis for feature valuation.

This script demonstrates a complete pipeline for estimating the Willingness-to-Pay
(WTP) for product features, as detailed in the paper "What is in a Price?".
The process includes:
1. Defining a "ground truth" market with known feature valuations.
2. Simulating a realistic choice-based conjoint survey to generate consumer data.
3. Building and fitting a Bayesian Hierarchical Logit Model using PyMC to
   recover individual and population-level preferences.
4. Analyzing the model's posterior distributions to calculate the WTP in dollars
   for each feature and validating the model's accuracy.
5. Running a revenue optimization simulation to identify the optimal price for a
   new product bundle based on the derived WTP.
"""

import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

# Optional: For professional-looking plots. Fails gracefully if not installed.
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'grid'])
except ImportError:
    print("Optional 'scienceplots' library not found. Using default plot style.")


# --- Configuration & Constants ---

# Ground truth average Willingness-to-Pay (WTP) for each feature upgrade
TRUE_WTP = {
    'storage_256gb': 100.0,
    'storage_512gb': 250.0,
    'camera_pro': 200.0,
    'material_titanium': 80.0
}

# Standard deviation to model heterogeneity in preferences across the market
TRUE_WTP_SD = {
    'storage_256gb': 20.0,
    'storage_512gb': 40.0,
    'camera_pro': 50.0,
    'material_titanium': 30.0
}

# Simulation parameters
N_RESPONDENTS = 300
N_QUESTIONS_PER_RESPONDENT = 20

# Model parameters
SAMPLING_DRAWS = 2000
SAMPLING_TUNE = 2000

# Directory for saving output plots
OUTPUTS_DIR = 'outputs'


# --- Data Simulation & Preparation ---

def simulate_conjoint_survey():
    """Simulates choice data from a heterogeneous market of consumers."""
    print(f"\n[1/5] Simulating choice-based conjoint survey for {N_RESPONDENTS} respondents...")
    
    # Create a population where each respondent has their own WTP, drawn from a distribution
    respondent_wtps = [
        {k: np.random.normal(v, TRUE_WTP_SD[k]) for k, v in TRUE_WTP.items()}
        for _ in range(N_RESPONDENTS)
    ]

    feature_levels = {
        'storage': [128, 256, 512],
        'camera': ['standard', 'pro'],
        'material': ['aluminum', 'titanium'],
        'price': [799, 899, 999, 1099, 1199]
    }

    survey_data = []
    for resp_id in range(N_RESPONDENTS):
        for _ in range(N_QUESTIONS_PER_RESPONDENT):
            profile_a = {feat: np.random.choice(lvls) for feat, lvls in feature_levels.items()}
            profile_b = {feat: np.random.choice(lvls) for feat, lvls in feature_levels.items()}
            if profile_a == profile_b:  # Ensure profiles are different
                continue
            
            # Calculate utility for each profile based on the respondent's personal WTP
            utility_a = _calculate_dollar_utility(profile_a, respondent_wtps[resp_id])
            utility_b = _calculate_dollar_utility(profile_b, respondent_wtps[resp_id])
            
            # Use the logit choice model to simulate a choice
            prob_a = 1 / (1 + np.exp(-(utility_a - utility_b)))
            choice = 1 if np.random.rand() < prob_a else 0

            row = {'resp_id': resp_id, 'choice': choice}
            for p_name, p_data in [('A', profile_a), ('B', profile_b)]:
                row.update({f'{k}_{p_name}': v for k, v in p_data.items()})
            survey_data.append(row)
            
    df = pd.DataFrame(survey_data)
    print(f"Survey generated with {len(df)} choices.")
    return df


def _calculate_dollar_utility(profile, wtps):
    """Helper function to calculate the utility of a product profile in dollars."""
    utility = -profile['price']
    if profile['storage'] == 256:
        utility += wtps['storage_256gb']
    if profile['storage'] == 512:
        utility += wtps['storage_512gb']
    if profile['camera'] == 'pro':
        utility += wtps['camera_pro']
    if profile['material'] == 'titanium':
        utility += wtps['material_titanium']
    return utility


def prepare_data_for_model(df):
    """Transforms raw survey data into a format suitable for the model."""
    print("\n[2/5] Preparing and standardizing data for the model...")
    # One-hot encode features and create difference variables
    df['price_diff'] = df['price_A'] - df['price_B']
    df['storage_256_diff'] = (df['storage_A'] == 256).astype(int) - (df['storage_B'] == 256).astype(int)
    df['storage_512_diff'] = (df['storage_A'] == 512).astype(int) - (df['storage_B'] == 512).astype(int)
    df['camera_pro_diff'] = (df['camera_A'] == 'pro').astype(int) - (df['camera_B'] == 'pro').astype(int)
    df['material_titanium_diff'] = (df['material_A'] == 'titanium').astype(int) - (df['material_B'] == 'titanium').astype(int)

    predictors = [
        'price_diff', 'storage_256_diff', 'storage_512_diff',
        'camera_pro_diff', 'material_titanium_diff'
    ]

    # Standardize predictors for model stability and efficiency
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[predictors] = scaler.fit_transform(df[predictors])
    print("Predictor variables standardized.")
    return df_scaled, scaler, predictors


# --- Bayesian Modeling ---

def build_and_run_model(df_scaled, predictors):
    """Builds and fits the Bayesian Hierarchical Logit model using PyMC."""
    print("\n[3/5] Building and running the Bayesian model...")
    respondent_idx, respondents = pd.factorize(df_scaled['resp_id'])
    coords = {"respondent": respondents, "feature": predictors}

    with pm.Model(coords=coords) as conjoint_model:
        X = pm.Data("X", df_scaled[predictors].values)
        
        # Hyperpriors for population-level parameters
        mu_beta = pm.Normal('mu_beta', mu=0, sigma=2, dims="feature")
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=1, dims="feature")
        
        # Non-centered parameterization for respondent-level coefficients
        # This is a best practice to improve sampler efficiency in hierarchical models.
        offset = pm.Normal('offset', mu=0, sigma=1, dims=("respondent", "feature"))
        beta = pm.Deterministic('beta', mu_beta + offset * sigma_beta, dims=("respondent", "feature"))
        
        # Calculate utility for each observation
        beta_per_obs = beta[respondent_idx]
        utility = pm.math.sum(beta_per_obs * X, axis=1)
        
        # Logit link function to get choice probability
        p_choice = pm.math.sigmoid(utility)
        
        # Likelihood (Bernoulli for binary choice)
        y_obs = pm.Bernoulli(
            'y_obs', p=p_choice, observed=df_scaled['choice'].values
        )

        # MCMC sampling
        trace = pm.sample(SAMPLING_DRAWS, tune=SAMPLING_TUNE, cores=1, target_accept=0.95)

    divergences = trace.sample_stats.diverging.sum().item()
    print(f"Model fitting complete. Number of divergences: {divergences}")
    if divergences > 0:
        print("Warning: Divergences detected. Model results may be unreliable.")
    else:
        print("Success! No divergences detected.")
        
    return trace


# --- Analysis & Visualization ---

def analyze_and_plot_wtp(trace, scaler, predictors):
    """Un-scales model results to calculate and plot WTP distributions."""
    print("\n[4/5] Analyzing posteriors and generating WTP plots...")
    
    feature_map = {
        "storage_256_diff": "storage_256gb",
        "storage_512_diff": "storage_512gb",
        "camera_pro_diff": "camera_pro",
        "material_titanium_diff": "material_titanium"
    }
    
    for feature_diff, feature_name in feature_map.items():
        wtp_dist = _calculate_wtp_distribution(trace, scaler, predictors, feature_diff)
        
        title = f'WTP for "{feature_name.replace("_", " ").title()}" Feature'
        filename = os.path.join(OUTPUTS_DIR, f'wtp_{feature_name}.png')
        _plot_posterior_and_save(
            wtp_dist, title, TRUE_WTP[feature_name], filename
        )


def _calculate_wtp_distribution(trace, scaler, predictors, feature_name):
    """Helper function to calculate the WTP distribution for a single feature."""
    mu_betas = trace.posterior['mu_beta']
    price_beta = mu_betas.sel(feature="price_diff")
    feature_beta = mu_betas.sel(feature=feature_name)
    
    price_idx = predictors.index("price_diff")
    feature_idx = predictors.index(feature_name)
    
    price_std = scaler.scale_[price_idx]
    feature_std = scaler.scale_[feature_idx]
    
    # WTP formula, corrected for standardization
    return - (feature_beta / price_beta) * (price_std / feature_std)


def _plot_posterior_and_save(data, title, true_value, filename):
    """Helper function to create and save a posterior plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_posterior(
        data, ax=ax, hdi_prob=0.95, ref_val=true_value, ref_val_color='red'
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Willingness-to-Pay ($)', fontsize=12)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"  - Saved plot to '{filename}'")
    plt.close(fig)


def run_revenue_optimization(trace, scaler, predictors):
    """Simulates expected revenue to find the optimal price for a product bundle."""
    print("\n[5/5] Running revenue optimization simulation...")
    
    # Define the "Pro" bundle as a combination of Pro Camera and Titanium Frame
    wtp_camera_pro = _calculate_wtp_distribution(trace, scaler, predictors, "camera_pro_diff")
    wtp_titanium = _calculate_wtp_distribution(trace, scaler, predictors, "material_titanium_diff")
    wtp_pro_bundle = wtp_camera_pro + wtp_titanium

    BASE_PHONE_PRICE = 799
    pro_price_points = np.linspace(900, 1300, 41)
    
    all_rev_samples = []
    for price in pro_price_points:
        # Calculate utility distribution of the Pro bundle vs. the base model
        utility_of_pro_vs_base = wtp_pro_bundle - (price - BASE_PHONE_PRICE)
        
        # Convert utility to purchase probability using the logit function
        # A scaling factor is used to prevent probabilities from becoming too extreme
        purchase_probability = 1 / (1 + np.exp(-utility_of_pro_vs_base / 100))
        
        # Calculate the distribution of revenue at this price point
        revenue_at_this_price = price * purchase_probability
        all_rev_samples.append(revenue_at_this_price.values.flatten())

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.violinplot(dataset=all_rev_samples, positions=pro_price_points, widths=15, showmeans=True)
    
    mean_revs = [np.mean(rev) for rev in all_rev_samples]
    optimal_price = pro_price_points[np.argmax(mean_revs)]
    
    ax.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.0f}')
    ax.set_title('Distribution of Expected Revenue vs. Price for "iPhone Pro" Bundle')
    ax.set_xlabel('Price Point ($)')
    ax.set_ylabel('Expected Revenue per Customer ($)')
    ax.legend()
    fig.tight_layout()
    
    filename = os.path.join(OUTPUTS_DIR, 'revenue_optimization.png')
    plt.savefig(filename, dpi=300)
    print(f"  - Saved revenue optimization plot to '{filename}'")
    plt.close(fig)


# --- Main Execution Block ---

def main():
    """Main function to run the entire analysis pipeline."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    df_raw = simulate_conjoint_survey()
    df_scaled, scaler, predictors = prepare_data_for_model(df_raw)
    trace = build_and_run_model(df_scaled, predictors)
    analyze_and_plot_wtp(trace, scaler, predictors)
    run_revenue_optimization(trace, scaler, predictors)
    
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
