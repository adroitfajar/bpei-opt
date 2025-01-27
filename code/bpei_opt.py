import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
from operator import itemgetter
from scipy.spatial.distance import euclidean


# Load Dataset
data = pd.read_csv("dataset.csv") 

# Define features (synthesis parameters) and labels (target properties)
features = data[['P_wt_percent', 'BPEI_Mn', 'AN_percent', 'EHA_percent', 'BA_percent']]
labels = data[['AC_mg-g', 'DT_C']]


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train Random Forest for AC_mg-g
ac_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                              param_grid=param_grid, cv=5, 
                              scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
ac_grid_search.fit(features, labels['AC_mg-g'])
ac_model = ac_grid_search.best_estimator_
print("Best parameters for AC_mg-g:", ac_grid_search.best_params_)

# Evaluate AC_mg-g model
ac_mae_scores = cross_val_score(ac_model, features, labels['AC_mg-g'], cv=5, scoring='neg_mean_absolute_error')
ac_rmse_scores = cross_val_score(ac_model, features, labels['AC_mg-g'], cv=5, scoring='neg_root_mean_squared_error')
ac_r2_scores = cross_val_score(ac_model, features, labels['AC_mg-g'], cv=5, scoring='r2')
# R² on the entire dataset
y_pred_ac = ac_model.predict(features)
r2_full_data_ac = r2_score(labels['AC_mg-g'], y_pred_ac)

# Display evaluation metrics
print("AC_mg-g Model Evaluation:")
print(f"MAE: {-np.mean(ac_mae_scores):.3f}")
print(f"RMSE: {-np.mean(ac_rmse_scores):.3f}")
print(f"R²: {np.mean(ac_r2_scores):.3f}")
print("R² on full dataset:", r2_full_data_ac)

# Visualization: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(labels['AC_mg-g'], y_pred_ac, alpha=0.7)
plt.plot([labels['AC_mg-g'].min(), labels['AC_mg-g'].max()], 
         [labels['AC_mg-g'].min(), labels['AC_mg-g'].max()], 'r--')
plt.title('Predicted vs Actual for AC_mg-g')
plt.xlabel('Actual AC_mg-g')
plt.ylabel('Predicted AC_mg-g')
plt.grid()
plt.show()

# Train Random Forest for DT_C
dt_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                              param_grid=param_grid, cv=5, 
                              scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
dt_grid_search.fit(features, labels['DT_C'])
dt_model = dt_grid_search.best_estimator_
print("Best parameters for DT_C:", dt_grid_search.best_params_)

# Evaluate DT_C model
dt_mae_scores = cross_val_score(dt_model, features, labels['DT_C'], cv=5, scoring='neg_mean_absolute_error')
dt_rmse_scores = cross_val_score(dt_model, features, labels['DT_C'], cv=5, scoring='neg_root_mean_squared_error')
dt_r2_scores = cross_val_score(dt_model, features, labels['DT_C'], cv=5, scoring='r2')
# R² on the entire dataset
y_pred_dt = dt_model.predict(features)
r2_full_data_dt = r2_score(labels['DT_C'], y_pred_dt)

# Display evaluation metrics
print("DT_C Model Evaluation:")
print(f"MAE: {-np.mean(dt_mae_scores):.3f}")
print(f"RMSE: {-np.mean(dt_rmse_scores):.3f}")
print(f"R²: {np.mean(dt_r2_scores):.3f}")
print("R² on full dataset:", r2_full_data_dt)

# Visualization: Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(labels['DT_C'], y_pred_dt, alpha=0.7)
plt.plot([labels['DT_C'].min(), labels['DT_C'].max()], 
         [labels['DT_C'].min(), labels['DT_C'].max()], 'r--')
plt.title('Predicted vs Actual for DT_C')
plt.xlabel('Actual DT_C')
plt.ylabel('Predicted DT_C')
plt.grid()
plt.show()

# Define objective function for Bayesian optimization
def objective_function(x):
    """
    Predicts adsorption capacity and desorption temperature for given synthesis parameters 
    using the two trained Random Forest models and combines them into a single weighted objective score.
    """
    # Convert parameters to DataFrame format for prediction
    params = pd.DataFrame(x, columns=['P_wt_percent', 'BPEI_Mn', 'AN_percent', 'EHA_percent', 'BA_percent'])
    
    # Predict using the separate models
    predicted_ac = ac_model.predict(params)  # Predictions from AC model
    predicted_dt = dt_model.predict(params)  # Predictions from DT model
    
    # Apply weights to DT and AC
    weight_dt = 0.55  # Weight for desorption temperature
    weight_ac = 0.45  # Weight for adsorption capacity
    
    # Combine into a single score (minimize weighted DT/AC)
    weighted_objective_score = ((weight_dt * predicted_dt) / (weight_ac * predicted_ac)).reshape(-1, 1)
    
    # Return the score for minimization in GPyOpt
    return weighted_objective_score

# Define Bounds for Synthesis Parameters
bounds = [
    {'name': 'P_wt_percent', 'type': 'discrete', 'domain': np.arange(0, 66, 1)},
    {'name': 'BPEI_Mn', 'type': 'discrete', 'domain': [600, 1800, 10000]},
    {'name': 'AN_percent', 'type': 'discrete', 'domain': np.arange(0, 101, 1)},
    {'name': 'EHA_percent', 'type': 'discrete', 'domain': np.arange(0, 101, 1)},
    {'name': 'BA_percent', 'type': 'discrete', 'domain': np.arange(0, 101, 1)}
]
# Define the constraint that the sum of AN_percent, EHA_percent, and BA_percent should be <= 100
constraints = [{'name': 'sum_constraint', 'constraint': 'x[:,2] + x[:,3] + x[:,4] - 100'}]

# Execute Bayesian optimization
print("Parameter Suggestion Starts")

# Step 1: Prepare Initial Conditions
initial_X = features.values  # Convert DataFrame to numpy array for GPyOpt
initial_Y = ((0.55 * labels['DT_C']) / (0.45 * labels['AC_mg-g'])).values.reshape(-1, 1)

# Step 2: Determine the Best (Lowest) Weighted Score
best_existing_score = initial_Y.min()
print(f"Best existing weighted score (minimized): {best_existing_score:.3f}")

# Step 3: Initialize Bayesian Optimization
np.random.seed(42)  # Set random seed for reproducibility
optimizer = BayesianOptimization(
    f=objective_function,
    domain=bounds,
    constraints=constraints,
    acquisition_type='MPI',  # Maximize Probability of Improvement
    acquisition_jitter=0.01,  # Focus on exploitation
    X=initial_X,
    Y=initial_Y
)

# Step 4: Run optimization for 100 iterations
optimizer.run_optimization(max_iter=100)

# Step 5: Enforce Diversity in Suggested Parameters
tested_params = optimizer.X
tested_scores = optimizer.Y

# Combine and sort parameters and scores by scores in ascending order (lower is better)
params_and_scores = sorted(zip(tested_params, tested_scores), key=itemgetter(1))

# Check for duplicates and enforce minimum distance
existing_params = features.values
filtered_params_and_scores = []

def is_far_enough(new_param_set, existing_sets, min_distance=5):
    """
    Ensure the new parameter set is sufficiently far from existing sets.
    """
    return all(euclidean(new_param_set, existing) >= min_distance for existing in existing_sets)

# Filter duplicates and near-duplicates with distance enforcement
for params, score in params_and_scores:
    if not any(np.allclose(params, row, atol=1e-3) for row in existing_params):  # Check duplicates
        if is_far_enough(params, [p for p, _ in filtered_params_and_scores], min_distance=5):
            filtered_params_and_scores.append((params, score))

# Step 6: Select Top 3 Parameter Sets
top_3_params_and_scores = filtered_params_and_scores[:3]

# Step 7: Display Results
print("\nTop suggested parameter sets for new experiments:")
for i, (params, score) in enumerate(top_3_params_and_scores):
    formatted_params = [int(param) for param in params]  # Convert to integers
    print(f"Set {i + 1}: Parameters = {formatted_params}, Weighted Score = {score[0]:.3f}")

print("Cycle completed")