
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# Load the dataset
file_path = 'https://raw.githubusercontent.com/reyar/Statsomat/master/cacao.csv'
data = pd.read_csv(file_path)

# Generate descriptive statistics
descriptive_stats = data.describe()

# Number of columns in the grid
n_cols = 3

# Histograms for each variable
fig, axes = plt.subplots(nrows=(len(data.columns) + n_cols - 1) // n_cols, ncols=n_cols, figsize=(15, 15))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    sns.histplot(data[col], kde=True, color='gray', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Boxplots for each variable
fig, axes = plt.subplots(nrows=(len(data.columns) + n_cols - 1) // n_cols, ncols=n_cols, figsize=(15, 15))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    sns.boxplot(data=data[col], color='gray', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# ECDF plots for each variable
fig, axes = plt.subplots(nrows=(len(data.columns) + n_cols - 1) // n_cols, ncols=n_cols, figsize=(15, 15))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    sns.ecdfplot(data[col], color='black', ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# OLS Regression model
y = data['stem_diameter']
X = data.drop(columns=['stem_diameter'])
X = sm.add_constant(X)
model = sm.OLS(y, X)
ols_model = model.fit()

# Calculate studentized residuals
influence = ols_model.get_influence()
studentized_residuals = influence.resid_studentized_internal

# Plot studentized residuals vs index
plt.figure(figsize=(10, 6))
plt.plot(studentized_residuals, 'o', label='Studentized Residuals')
plt.axhline(y=3, color='red', linestyle='--', label='Threshold (+3)')
plt.axhline(y=-3, color='red', linestyle='--', label='Threshold (-3)')
for i, residual in enumerate(studentized_residuals):
    if abs(residual) > 3:
        plt.annotate(i, (i, residual), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Studentized Residuals vs Index')
plt.xlabel('Index')
plt.ylabel('Studentized Residual')
plt.legend()
plt.show()

# Calculate leverage and Cook's Distance
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

# Plot leverage vs index
plt.figure(figsize=(10, 6))
plt.stem(range(len(X)), leverage, linefmt='black', markerfmt='o', basefmt=" ", use_line_collection=True)
plt.axhline(y=2*X.shape[1]/len(X), color='red', linestyle='--', label=f'Threshold (2p/n = {2*X.shape[1]/len(X):.2f})')
for i, lev in enumerate(leverage):
    if lev > 2*X.shape[1]/len(X):
        plt.annotate(i, (i, lev), textcoords="offset points", xytext=(0,10), ha='center')
        plt.stem([i], [lev], linefmt='red', markerfmt='o', basefmt=" ", use_line_collection=True)
plt.title('Leverage vs Index')
plt.xlabel('Index')
plt.ylabel('Leverage')
plt.legend()
plt.show()

# Plot Cook's Distance vs index
plt.figure(figsize=(10, 6))
plt.stem(range(len(X)), cooks_d, linefmt='black', markerfmt='o', basefmt=" ", use_line_collection=True)
plt.axhline(y=4/len(X), color='red', linestyle='--', label=f'Threshold (4/n = {4/len(X):.4f})')
for i, cd in enumerate(cooks_d):
    if cd > 4/len(X):
        plt.annotate(i, (i, cd), textcoords="offset points", xytext=(0,10), ha='center')
        plt.stem([i], [cd], linefmt='red', markerfmt='o', basefmt=" ", use_line_collection=True)
plt.title("Cook's Distance vs Index")
plt.xlabel('Index')
plt.ylabel("Cook's Distance")
plt.legend()
plt.show()

# Perform the Rainbow Test
rainbow_stat, rainbow_p_value = linear_rainbow(ols_model)

# Plot residuals vs fitted values
predictions = ols_model.fittedvalues
residuals = ols_model.resid
plt.figure(figsize=(10, 6))
sns.residplot(x=predictions, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.scatter(predictions, residuals, color='black', alpha=0.5)
plt.axhline(0, color='blue', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Perform the Breusch-Pagan Test
standardized_residuals = ols_model.get_influence().resid_studentized_internal
sqrt_standardized_residuals = np.sqrt(np.abs(standardized_residuals))
bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
bp_statistic, bp_p_value = bp_test

# Plot the Scale-Location plot
plt.figure(figsize=(10, 6))
sns.regplot(x=predictions, y=sqrt_standardized_residuals, lowess=True, line_kws={'color': 'red'})
plt.scatter(predictions, sqrt_standardized_residuals, color='black', alpha=0.5)
plt.axhline(0, color='blue', linestyle='--')
plt.title('Scale-Location Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Sqrt |Standardized Residuals|')
plt.show()

# Perform the Durbin-Watson test
dw_statistic = durbin_watson(ols_model.resid)

# Plot the studentized residuals over time (index)
plt.figure(figsize=(10, 6))
plt.plot(studentized_residuals, color='black')
plt.title('Studentized Residuals Over Time')
plt.xlabel('Observation Index')
plt.ylabel('Studentized Residuals')
plt.axhline(0, color='blue', linestyle='--')
plt.show()

# Perform the Shapiro-Wilk test
shapiro_statistic, shapiro_p_value = stats.shapiro(standardized_residuals)

# Generate the QQ plot for standardized residuals
plt.figure(figsize=(10, 6))
stats.probplot(standardized_residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Standardized Residuals')
plt.show()

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Predictors')
plt.show()
