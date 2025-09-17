# 🌳 Air Quality Prediction with Decision Tree Regressor  

This project implements a **Decision Tree Regressor** using the [Air Quality UCI dataset](https://archive.ics.uci.edu/dataset/360/air+quality) to predict **carbon monoxide (CO) concentration**.  
The workflow includes **data preprocessing, cross-validation for hyperparameter tuning, model evaluation, feature importance analysis, and visualization**.  

---

## 📂 Project Overview  

The objective is to **predict CO(GT)** (true CO concentration measured in mg/m³) from environmental sensor data (e.g., temperature, humidity, hydrocarbons, NOx).  
A **Decision Tree Regressor** is trained with **K-Fold Cross Validation** to select the optimal depth and avoid overfitting.  

Key aspects include:  
* ✅ Handling missing values (`-200` → replaced with mean of valid values)  
* ✅ Standardization of input features and target  
* ✅ **10-Fold Cross Validation** for optimal depth selection  
* ✅ Final evaluation on a **hold-out test set** (15% of data)  
* ✅ **Graphical analysis**: MSE/R² trends, feature importances, tree visualization, residuals  
* ✅ Export of the trained decision tree with **Graphviz**  

---

## 🧩 Dataset  

- **Name**: Air Quality UCI Dataset  
- **Format**: `.xlsx`  
- **Target Variable**: `CO(GT)` (Carbon Monoxide concentration)  
- **Features**: Temperature, Humidity, NMHC, NOx, NO₂, and more  
- **Cleaning Rule**: Replace invalid values `-200` with column mean  

---

## ⚙️ Workflow  

### 1. 🔧 Data Preprocessing  
* Drop irrelevant columns (`Date`, `Time`)  
* Replace `-200` with feature-wise mean  
* Split dataset into:  
  - **85% Training + Validation** (for cross-validation)  
  - **15% Test** (for final evaluation)  
* Standardize both features and target  

---

### 2. 🔎 Model Training & Cross Validation  
* Algorithm: **DecisionTreeRegressor (scikit-learn)**  
* Hyperparameter tuned: **max_depth (2 → 15)**  
* Evaluation metrics:  
  - **MSE (Mean Squared Error)**  
  - **R² (Coefficient of Determination)**  
* Cross-validation: **10-fold**  

---

### 3. 📊 Evaluation & Results  

The model selected an **optimal depth of 7** with **104 leaves**.  

**Final results:**  

| Metric         | Training Set | Test Set |
|----------------|--------------|----------|
| MSE            | 0.1354       | 0.2391   |
| R²             | 0.8646       | 0.7556   |
| Pearson r      | –            | 0.8694   |

📌 Interpretation:  
- The model achieves **high explanatory power (R² ~0.86 on training, 0.76 on test)**.  
- A strong **Pearson correlation (0.87)** indicates reliable predictive performance.  
- The gap between training and test suggests **good generalization without severe overfitting**.  



---

### 4. 📈 Visualizations  
The following plots are automatically generated:  

1. **MSE vs Tree Depth**  
   Shows training & validation errors to analyze under/overfitting.  
<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/071fabe2-dc21-4b17-8b90-e6a72592ce7d" />


2. **R² vs Tree Depth**  
   Displays how explanatory power changes with depth.  
<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/09e51c44-57d3-497e-afaa-7ae85aa95e6b" />

3. **Feature Importance**  
   Horizontal bar chart ranking feature contribution.  
<img width="1366" height="655" alt="Figure_1" src="https://github.com/user-attachments/assets/59fc71c3-41e4-4f50-bbe3-f9660508337e" />

4. **Decision Tree Visualization**  
   Colored tree representation with splits, exported to `decision_tree_optimal.png`.  
<img width="931" height="490" alt="Figure_1" src="https://github.com/user-attachments/assets/94eaea14-bf98-4135-a65b-4cac785c2b4e" />

6. **Predictions vs Real Values**  
   Scatter plot showing model accuracy on test set.  
<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/7f576df5-2a21-4b44-b712-b746e94d2fc0" />

7. **Residuals vs Predictions**  
   Analyzes prediction errors for bias or variance issues.  
<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/aaa8f56d-6b0d-43d7-bdef-9f9ac0748ea7" />

---

