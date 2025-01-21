import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
import itertools

# Load the data set as a DataFrame 
vehicles_df = pd.read_csv(r'C:\Users\brttk\Documents\sprint_4_project\notebooks\vehicles_us.csv')

# Convert date_posted' to datetime data type.
vehicles_df['date_posted'] = pd.to_datetime(vehicles_df['date_posted'])


# In our preliminary EDA for this project we verified that there were only 1.0 and NaN values in the 
# is_4wd column. The most reasonable interpretation of this is that 1.0 means yes and NaN means no 
# Therefore, we are going to replace the 1.0 values with 'yes' and the NaN values with 'no' 
# Replace NaN values with 'no' and 1.0 values with 'yes' 
vehicles_df['is_4wd'] = vehicles_df['is_4wd'].replace({1.0: 'yes', float('NaN'): 'no'})


# Defining a list of non-standard placeholders we want to check for. 
non_standard_values = ['', 'na', 'null', 'missing']

# Create DataFrame that contains only object datatypes.  
categorical_df = vehicles_df.select_dtypes(include=[object])

for col in categorical_df:
    categorical_df[col] = categorical_df[col].str.strip().str.lower()
    # Calculating the number of non-standard null values in 'paint_color' and assigning that value to msk_null_paint_color
    mask_null_paint_color = categorical_df[col].isin(non_standard_values).sum()
    # we verified that this code worked as it was supposed to in our EDA. 


# Create a manufacturer column by running the 'model' column through the lambda x: x.split()[0] function with .apply()
vehicles_df['manufacturer'] = vehicles_df['model'].apply(lambda x: x.split()[0])


# Step 2: Identify numeric and categorical columns
numeric_columns = vehicles_df.select_dtypes(include='number').columns.tolist()
categorical_columns = vehicles_df.select_dtypes(exclude='number').columns.tolist()

# Step 3: Display the user interface (Variable selection)
st.header("Correlation and Statistical Analysis")
selected_numeric_var = st.selectbox("Select a numeric variable", numeric_columns)
selected_cat_var = st.selectbox("Select a categorical variable", categorical_columns)

# Checkbox to show correlation
show_corr = st.checkbox("Show correlation", value=True)
show_significance = st.checkbox("Show statistical significance", value=True)
show_effect_size = st.checkbox("Show effect size", value=True)

# Step 4: Correlation Calculation
# Numeric-to-numeric correlation (Pearson's)
def calc_numeric_corr(x, y):
    valid_data = pd.concat([x, y], axis=1).dropna()
    if len(valid_data) > 0:
        return pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
    else:
        return np.nan, np.nan

# Categorical-to-categorical correlation (Cramer's V)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p_value, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# Numeric-to-categorical correlation (ANOVA for F-statistic)
def calc_numeric_cat_corr(numeric, categorical):
    grouped_data = [numeric[categorical == category] for category in categorical.unique()]
    f_statistic, p_value = f_oneway(*grouped_data)
    return f_statistic, p_value

# Effect size calculation (Cohen's d for numeric)
def cohens_d(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    std_x, std_y = x.std(), y.std()
    return (mean_x - mean_y) / np.sqrt((std_x**2 + std_y**2) / 2)

# Step 5: Display correlation, significance, and effect size based on selection
if show_corr:
    if isinstance(vehicles_df[selected_numeric_var], pd.Categorical):
        correlation = cramers_v(vehicles_df[selected_numeric_var], vehicles_df[selected_cat_var])
    else:
        correlation, p_value = calc_numeric_corr(vehicles_df[selected_numeric_var], vehicles_df[selected_numeric_var])
    st.write(f"Correlation: {correlation}")
    
if show_significance:
    st.write(f"P-value: {p_value}")
    if p_value < 0.001:
        st.write("The correlation is statistically significant.")
    else:
        st.write("The correlation is not statistically significant.")
    
if show_effect_size:
    if isinstance(vehicles_df[selected_numeric_var], pd.Categorical):
        effect_size = mutual_info_score(vehicles_df[selected_numeric_var], vehicles_df[selected_cat_var])
    else:
        effect_size = cohens_d(vehicles_df[selected_numeric_var], vehicles_df[selected_numeric_var])
    st.write(f"Effect size: {effect_size}")

# Step 6: Visualization
if show_corr:
    if isinstance(vehicles_df[selected_numeric_var], pd.Categorical):  # Numeric-to-Categorical
        # Box plot for Numeric-to-Categorical correlation
        fig = px.box(vehicles_df, x=selected_cat_var, y=selected_numeric_var,
                     title=f"Box Plot: {selected_numeric_var} vs {selected_cat_var}")
        st.plotly_chart(fig)
    elif isinstance(vehicles_df[selected_cat_var], pd.Categorical):  # Categorical-to-Categorical
        # Histogram or Bar plot for Categorical-to-Categorical correlation
        fig = px.histogram(vehicles_df, x=selected_numeric_var, color=selected_cat_var, 
                            title=f"Histogram: {selected_numeric_var} by {selected_cat_var}")
        st.plotly_chart(fig)
    else:  # Numeric-to-Numeric (already default scatter plot)
        fig = px.scatter(vehicles_df, x=selected_numeric_var, y=selected_cat_var,
                         title=f"Scatter Plot: {selected_numeric_var} vs {selected_cat_var}")
        st.plotly_chart(fig)

# Step 7: Display the correlation matrix for numeric variables
if st.checkbox("Show correlation matrix for numeric variables"):
    corr_matrix = vehicles_df[numeric_columns].corr()
    st.write(corr_matrix)
    fig_corr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, 
                         title="Correlation Matrix for Numeric Variables")
    st.plotly_chart(fig_corr)

# More functionality can be added here for more complex analysis