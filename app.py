import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency
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
st.header("Used Car Traits: Statistical Associations")
all_columns = vehicles_df.columns.tolist()  # List of all columns (both numeric and categorical)


# User selects the two variables to compare
selected_var_1 = st.selectbox("Select the first variable", all_columns)
selected_var_2 = st.selectbox("Select the second variable", all_columns)

# Checkbox to show correlation
show_corr = st.checkbox("Show correlation", value=False)
show_significance = st.checkbox("Show statistical significance", value=False)
show_effect_size = st.checkbox("Show effect size", value=False)

# # Step 4: Correlation Calculation
# Numeric-to-numeric correlation (Pearson's)
def calc_numeric_corr(x, y):
    valid_data = pd.concat([x, y], axis=1).dropna()
    # Check if both variables are numeric
    if len(valid_data) > 0 and pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        return pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
    else:
        return np.nan, np.nan

# Categorical-to-categorical correlation (Cramer's V)
def cramers_v(x, y):
    # Drop NaN values for each column separately before performing the chi-squared test
    valid_data = pd.concat([x, y], axis=1).dropna(subset=[x.name, y.name])
    
    if len(valid_data) > 0:
        confusion_matrix = pd.crosstab(valid_data[x.name], valid_data[y.name])
        chi2, p_value, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        return p_value, np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
    else:
        return np.nan, np.nan
    

# Numeric-to-categorical correlation (ANOVA for F-statistic)
def calc_numeric_cat_corr(numeric, categorical):
    # Drop NaN values for numeric and categorical columns separately
    valid_data = pd.concat([numeric, categorical], axis=1).dropna(subset=[numeric.name, categorical.name])
    
    if len(valid_data) > 0:
        grouped_data = [valid_data[numeric.name][valid_data[categorical.name] == category] for category in categorical.unique()]
        f_statistic, p_value = f_oneway(*grouped_data)
        return f_statistic, p_value
    else:
        return np.nan, np.nan

# Effect size calculation (Cohen's d for numeric)
def calc_omega_squared(numeric, categorical):
    if not pd.api.types.is_numeric_dtype(numeric):
        raise ValueError(f"The numeric variable {numeric.name} is not numeric. Please check the data type.")
    if not pd.api.types.is_object_dtype(categorical):
        raise ValueError(f"The categorical variable {categorical.name} is not an object (string) type. Please check the data type.")
    
    grouped_data = [numeric[categorical == category] for category in categorical.unique()]
    # Calculate the sums of squares (SS)
    # SS_between: variance explained by the group (between-groups)
    grand_mean = np.mean(numeric)  # The grand mean of the entire numeric dataset
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in grouped_data)

    # SS_within: within-group variance (residual variance)
    ss_within = sum(np.sum((group - np.mean(group))**2) for group in grouped_data)

    # SS_total: total variance in the data
    ss_total = ss_between + ss_within

    # Degrees of freedom (df)
    df_between = len(grouped_data) - 1  # k - 1, where k is the number of groups
    df_within = sum(len(group) for group in grouped_data) - len(grouped_data)  # N - k

    # Calculate omega-squared
    omega_squared = (ss_between - (df_between * ss_within / sum(len(group) for group in grouped_data))) / ss_total
    return omega_squared
# Step 5: Display correlation, significance, and effect size based on selection
if show_corr:
    # Check if both selected variables are categorical
    if (vehicles_df[selected_var_1].dtype == 'object') and (vehicles_df[selected_var_2].dtype == 'object'):
        # Run Cramér's V for categorical-to-categorical correlation
        association_measure, p_value = cramers_v(vehicles_df[selected_var_1], vehicles_df[selected_var_2])
        st.write(f"Association Measure (Cramer's V): {association_measure}")
    # Check if one is categorical and the other is numeric
    elif (vehicles_df[selected_var_1].dtype == 'object') and (pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])) or \
         (vehicles_df[selected_var_2].dtype == 'object') and (pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        # Run ANOVA (F-statistic) for numeric-to-categorical correlation
        if vehicles_df[selected_var_1].dtype == 'object':
            numeric_var = vehicles_df[selected_var_2]
            categorical_var = vehicles_df[selected_var_1]
        else:
            numeric_var = vehicles_df[selected_var_1]
            categorical_var = vehicles_df[selected_var_2]
        
        f_statistic, p_value = calc_numeric_cat_corr(numeric_var, categorical_var)
        st.write(f"F-statistic (ANOVA): {f_statistic}")
    
    # If both are numeric, run Pearson's correlation
    else:
        # Run Pearson's r for numeric-to-numeric correlation
        correlation, p_value = calc_numeric_corr(vehicles_df[selected_var_1], vehicles_df[selected_var_2])
    # Display the correlation coefficient here
    
        st.write(f"Correlation (Pearson's r): {correlation}")
    
if show_significance:
    st.write(f"P-value: {p_value}")
    if p_value < 0.001:
        st.write("The correlation is statistically significant.")
    else:
        st.write("The correlation is not statistically significant.")
    
if show_effect_size:
    if (vehicles_df[selected_var_1].dtype == 'object') and (vehicles_df[selected_var_2].dtype == 'object'):
        st.text("For a Cramer's V analysis the association measure is the effect size.\n Cramer's V Effect Size Guide:\n0.0 - 0.1: Weak association\n0.1 - 0.3: Small association\n0.3 - 0.5: Medium association\n0.5 - 1.0: Strong association")
    elif (vehicles_df[selected_var_1].dtype == 'object') and (pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])):
         omega_squared = calc_omega_squared(vehicles_df[selected_var_2], vehicles_df[selected_var_1])
         st.text(f"Effect size (omega-squared): {omega_squared}\n\nOmega-Squared Effect Size Guide:\n0.01: Very small effect size\n0.01 - 0.06: Small effect size\n0.06 - 0.14: Medium effect size\n0.14 - 0.25: Large effect size\n> 0.25: Very large effect size")
    elif (vehicles_df[selected_var_2].dtype == 'object') and (pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        omega_squared = calc_omega_squared(vehicles_df[selected_var_1], vehicles_df[selected_var_2])
        st.text(f"Effect size (omega-squared): {omega_squared}\n\nOmega-Squared Effect Size Guide:\n0.01: Very small effect size\n0.01 - 0.06: Small effect size\n0.06 - 0.14: Medium effect size\n0.14 - 0.25: Large effect size\n> 0.25: Very large effect size")

    else: 
        st.text("For a Pearson's r analysis the correlation coefficient is the effect size.\n\nPearson's r Effect Size Guide (the values below are magnitudes which means they apply to both positive and negative correlations):\n0 - 0.1: Very small effect size\n0.1 - 0.3 Small effect size\n0.3 - 0.5: Medium effect size\n0.5 - 1.0: Large effect size") 

# Step 6: Visualization
if show_corr:
    if (vehicles_df[selected_var_1].dtype == 'object') and (pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])):
        fig = px.box(vehicles_df, x=selected_var_1, y=selected_var_2, 
                     title=f"Box Plot: {selected_var_1} vs {selected_var_2}") 
        st.plotly_chart(fig) 
    elif (vehicles_df[selected_var_2].dtype == 'object') and (pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        fig = px.box(vehicles_df, x=selected_var_2, y=selected_var_1, 
                     title=f"Box Plot: {selected_var_2} vs {selected_var_1}") 
        st.plotly_chart(fig)
    elif (vehicles_df[selected_var_1].dtype == 'object') and (vehicles_df[selected_var_2].dtype == 'object'):  # Categorical-to-Categorical 
        # Histogram or Bar plot for Categorical-to-Categorical correlation
        fig = px.histogram(vehicles_df, x=selected_var_1, color=selected_var_2,  
                            title=f"Histogram: {selected_var_1} by {selected_var_2}") 
        st.plotly_chart(fig)
    else:  # Numeric-to-Numeric (already default scatter plot)
        fig = px.scatter(vehicles_df, x=selected_var_1, y=selected_var_2, 
                         title=f"Scatter Plot: {selected_var_1} vs {selected_var_2}") 
        st.plotly_chart(fig)

# Step 7: Display the correlation matrix for numeric variables
if st.checkbox("Show correlation matrix for numeric variables"):
    corr_matrix = vehicles_df[numeric_columns].corr() 
    st.write(corr_matrix)
    fig_corr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, 
                         title="Correlation Matrix for Numeric Variables")
    st.plotly_chart(fig_corr)

# More functionality can be added here for more complex analysis