# -*- coding: utf-8 -*-
"""
Created on Thu May 29 10:35:13 2025

@author: Utente
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import KFold 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz


# Loading and pre-processing data
dataset = pd.read_excel('AirQualityUCI.xlsx')

feature_names = dataset.drop(['CO(GT)','Date','Time'], axis=1).columns.tolist()
target_name = 'CO(GT)'

for col in dataset.select_dtypes(include=[np.number]).columns:
    dataset[col] = dataset[col].astype(float)
    mean_val = dataset.loc[dataset[col] != -200, col].mean()
    dataset.loc[dataset[col] == -200, col] = mean_val

x = dataset.drop(['CO(GT)','Date','Time'], axis=1).values
y = np.asarray(dataset['CO(GT)'])

# Initial split (maintaining a separate final test set)
n = len(x)
train_val_end = int(n * 0.85) # 85% for training+validation, 15% for final test

x_train_val = x[:train_val_end] # Data for training and validation (K-fold on these)
y_train_val = y[:train_val_end]

x_test_final = x[train_val_end:] # Final test set (for final evaluation of the chosen model)
y_test_final = y[train_val_end:]

# Standardization compared to the training+validation set
mean_x_train_val = x_train_val.mean(axis=0)
std_x_train_val = x_train_val.std(axis=0)
std_x_train_val[std_x_train_val == 0] = 1e-8 # Avoid division by zero

x_train_val_scaled = (x_train_val - mean_x_train_val) / std_x_train_val
x_test_final_scaled = (x_test_final - mean_x_train_val) / std_x_train_val # Apply the same standardization

mean_y_train_val = y_train_val.mean(axis=0)
std_y_train_val = y_train_val.std(axis=0)
if std_y_train_val == 0:
    std_y_train_val = 1e-8

y_train_val_scaled = (y_train_val - mean_y_train_val) / std_y_train_val
y_test_final_scaled = (y_test_final - mean_y_train_val) / std_y_train_val


# K-FOLD CROSS-VALIDATION FOR OPTIMAL DEPTH SELECTION 
kf = KFold(n_splits=10, shuffle=True, random_state=42) # Initialize K-Fold with 10 folds

best_avg_mse_cv = float('inf')
best_depth_cv = None

# Final_best_model will be trained after the loop with the best depth
final_best_model = None 

max_depth_range = range(2, 16) # Depth range to be tested

# Lists to save average results for charts
avg_mse_train_list = []
avg_mse_val_list = []
avg_r2_train_list = []
avg_r2_val_list = []
depth_list = []

print("K-fold Cross-Validation (CV) start for optimal depth selection...")

for depth in max_depth_range:
    mse_val_scores_for_depth = []
    mse_train_scores_for_depth = []
    r2_val_scores_for_depth = []
    r2_train_scores_for_depth = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(x_train_val_scaled)):
        x_train_fold, x_val_fold = x_train_val_scaled[train_index], x_train_val_scaled[val_index]
        y_train_fold, y_val_fold = y_train_val_scaled[train_index], y_train_val_scaled[val_index]

        # Create and add Scikit-learn's DecisionTreeRegressor template
        model = DecisionTreeRegressor(max_depth=depth, min_samples_split=10, random_state=42)
        model.fit(x_train_fold, y_train_fold)

        # Predictions and MSE/R² calculation on the fold validation set
        y_val_pred_fold = model.predict(x_val_fold)
        mse_val_scores_for_depth.append(mean_squared_error(y_val_fold, y_val_pred_fold))
        r2_val_scores_for_depth.append(r2_score(y_val_fold, y_val_pred_fold))

        # Predictions and MSE/R² calculation on the fold training set
        y_train_pred_fold = model.predict(x_train_fold)
        mse_train_scores_for_depth.append(mean_squared_error(y_train_fold, y_train_pred_fold))
        r2_train_scores_for_depth.append(r2_score(y_train_fold, y_train_pred_fold))

    # Calculate the averages of the MSE and R² for the current depth on all folds
    avg_mse_val_for_depth = np.mean(mse_val_scores_for_depth)
    avg_mse_train_for_depth = np.mean(mse_train_scores_for_depth)
    avg_r2_val_for_depth = np.mean(r2_val_scores_for_depth)
    avg_r2_train_for_depth = np.mean(r2_train_scores_for_depth)
    
    # Save results for charts
    depth_list.append(depth)
    avg_mse_train_list.append(avg_mse_train_for_depth)
    avg_mse_val_list.append(avg_mse_val_for_depth)
    avg_r2_train_list.append(avg_r2_train_for_depth)
    avg_r2_val_list.append(avg_r2_val_for_depth)
    
    print(f"\n  depth: {depth}")
    print(f"  Average Train MSE  (CV): {avg_mse_train_for_depth:.4f}")
    print(f"  Average Validation MSE (CV): {avg_mse_val_for_depth:.4f}")
    print(f"  Average Train R² (CV): {avg_r2_train_for_depth:.4f}")
    print(f"  Average Validation R² (CV): {avg_r2_val_for_depth:.4f}")

    # Update the best model if this depth has a better average validation MSE
    if avg_mse_val_for_depth < best_avg_mse_cv:
        best_avg_mse_cv = avg_mse_val_for_depth
        best_depth_cv = depth

print("Results Selection Optimal Depth")
print(f"Best depth selected with K-fold CV: {best_depth_cv}")
print(f"Average MSE Validation (better depth): {best_avg_mse_cv:.4f}")


# Final evaluation of the best model on the TEST SET and the COMPLETE TRAINING SET 

# Train the final model with the best depth 
# This is the final model we will use for evaluations on real training and test sets.
if best_depth_cv is not None:
    final_best_model = DecisionTreeRegressor(max_depth=best_depth_cv, min_samples_split=10, random_state=42)
    final_best_model.fit(x_train_val_scaled, y_train_val_scaled)
else:
    print("Warning: No optimal depth found. Using default depth (5).")
    final_best_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    final_best_model.fit(x_train_val_scaled, y_train_val_scaled)


# Evaluation on the complete training set (x_train_val_scaled)
y_train_val_pred_final = final_best_model.predict(x_train_val_scaled)
final_train_mse = mean_squared_error(y_train_val_scaled, y_train_val_pred_final)
final_train_r2 = r2_score(y_train_val_scaled, y_train_val_pred_final)

# Evaluation on the final test set (x_test_final_scaled)
y_test_pred_final = final_best_model.predict(x_test_final_scaled)
final_test_mse = mean_squared_error(y_test_final_scaled, y_test_pred_final)
final_test_r2 = r2_score(y_test_final_scaled, y_test_pred_final)

print("\n Final evaluation on the Training and Test Set")
print(f"\nTraining MSE: {final_train_mse:.4f}")
print(f"Training R²: {final_train_r2:.4f}")
print(f"Test MSE: {final_test_mse:.4f}")
print(f"Test R²: {final_test_r2:.4f}")

# For the Pearson correlation coefficient r on the Test Set 
y_test_pred_final_unscaled = (y_test_pred_final * std_y_train_val) + mean_y_train_val
r_final = np.corrcoef(y_test_final, y_test_pred_final_unscaled)[0, 1]
print(f"Pearson r on the Test Set (original data): {r_final:.4f}")

# MSE trend as a function of depth
plt.figure(figsize=(10, 6))
plt.plot(depth_list, avg_mse_train_list, label='Average Training MSE (CV)', marker='o')
plt.plot(depth_list, avg_mse_val_list, label='Average Validation MSE (CV)', marker='x')
plt.title('Trend of the Average Quadratic Error (MSE) vs. Depth of the Tree')
plt.xlabel('Maximum Tree Depth')
plt.ylabel('MSE')
plt.xticks(depth_list)
plt.legend()
plt.grid(True)
plt.show()

# Trend of R^2 as a function of depth
plt.figure(figsize=(10, 6))
plt.plot(depth_list, avg_r2_train_list, label='Average Training R² (CV)', marker='o')
plt.plot(depth_list, avg_r2_val_list, label='Average Validation R² (CV)', marker='x')
plt.title('Trend of the Determination Coefficient (R²) vs. Depth of the Tree')
plt.xlabel('Maximum Tree Depth')
plt.ylabel('R²')
plt.xticks(depth_list)
plt.legend()
plt.grid(True)
plt.show()

# Importance of the features of the final model
if final_best_model:
    feature_importances = final_best_model.feature_importances_
    
    # Create a DataFrame to facilitate sorting and viewing
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 7))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance of Feature')
    plt.ylabel('Feature')
    plt.title('Importance of Features in the Final Model (Decision Tree Regressor)')
    plt.gca().invert_yaxis() # The most important features above
    plt.show()
else:
    print("Unable to generate feature importance graph: final model has not been trained.")


# Tree View

if final_best_model:
    print(f"\nGenerating the tree graph with depth {best_depth_cv}...")
    plt.figure(figsize=(20, 10)) # Increase size for better readability
    plot_tree(final_best_model, 
              feature_names=feature_names, 
              filled=True, 
              rounded=True, 
              fontsize=8)
    plt.title(f"Optimal Decision Tree (Depth: {best_depth_cv})")
    plt.show()

    try:
        dot_data = export_graphviz(final_best_model,
                                   out_file=None, 
                                   feature_names=feature_names,
                                   filled=True, rounded=True,
                                   special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("decision_tree_optimal", format="png", view=True) # Save and open PNG file
        print(f"Tree exported as decision_tree_optimal.png with depth {best_depth_cv}")
    except Exception as e:
        print(f"Error exporting or viewing with Graphviz: {e}")
        print("Make sure you have Graphviz software installed on your system.")
else:
    print("The tree could not be displayed - the final model was not trained.")


# Predictions vs Real Values (Test Set)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_final, y_test_pred_final_unscaled, alpha=0.6) # Let’s use unscaled y_test_final and unscaled predictions
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], 'r--', lw=2) 
plt.title('Predictions vs Real Values (Test Set)')
plt.xlabel(f'real values of {target_name}')
plt.ylabel(f'predictions of {target_name}')
plt.grid(True)
plt.show()

# Residues vs Predictions (Test Set)
residuals = y_test_final - y_test_pred_final_unscaled
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred_final_unscaled, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residues vs Predictions (Test Set)')
plt.xlabel(f'predictions of {target_name}')
plt.ylabel('Residues (Real - Predicted)')
plt.grid(True)
plt.show()

print(f"Actual shaft depth: {final_best_model.get_depth()}")
print(f"Number of leaves: {final_best_model.get_n_leaves()}")