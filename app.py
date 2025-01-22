"""Used Car Traits: Statistical Associations, by Brett Kunkel
The script below creates an app utilyzing the streamlit that allows
users to select any used car variable they want and compare its relationship
to any othre used car variable. Providing the user this information will help
inform their search for a new car, or, if the user works at a used car
dealership, will provide them with informatino they can utilyze to increase
revenue."""


# import the relevant modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency

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

# Iterate over all the columns in categorical_df and making them uniform
# (no capital letters, no begining or ends spaces.)
for col in categorical_df:
    categorical_df[col] = categorical_df[col].str.strip().str.lower()
    # we verified that the code above worked as it was supposed to in our EDA.


# Create a manufacturer column by running the 'model' column through the
# lambda x: x.split()[0] function with .apply()
vehicles_df['manufacturer'] = vehicles_df['model'].apply(lambda x: x.split()[0])

# We're going to want the 'cylinders' column to be an object data type to start
# So, let's make that conversion:
vehicles_df['cylinders'] = vehicles_df['cylinders'].astype('object')

# Identify numeric and categorical columns
numeric_columns = vehicles_df.select_dtypes(include='number').columns.tolist()
categorical_columns = vehicles_df.select_dtypes(include='object').columns.tolist()

# Display the name of the app
st.header("Used Car Traits: Statistical Associations")
# Create a list of all columns (both numeric and categorical)
# These will be the variable selection options for the user.
all_columns = vehicles_df.columns.tolist()


# User selects the two variables to compare
selected_var_1 = st.selectbox("Select the first variable", all_columns)
selected_var_2 = st.selectbox("Select the second variable", all_columns)

# Create correlation, significance, and effect size boxes for the user to check
run_anal = st.checkbox("Run analysis", value=False)
show_significance = st.checkbox("Show statistical significance", value=False)
show_effect_size = st.checkbox("Show effect size", value=False)


def calc_numeric_corr(x, y):
    """We are defining this function to calculate the Pearson's r correlation
    between two numeric variables. Later we will call this function following the
    conditional if statement that the user choose two numeric variables."""
    # concatinate the input variables (i.e. columns) and drop the NaN values from them.
    # As with the EDA, dropping the NaN values this way will prevent us from having to
    # unnecessarily drop values just because another variable happened to have an
    # NaN value in that same row. Therefore, we will have more data points to
    # analyze and more robust results.
    valid_data = pd.concat([x, y], axis=1).dropna()
    # Check if both variables are numeric
    if (len(valid_data) > 0) and (pd.api.types.is_numeric_dtype(x)) and (
        pd.api.types.is_numeric_dtype(y)):
        # Run Pearson'r analysis on all of the rows in the two columns
        # we have selected
        return pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])

    return None


def cramers_v(x, y):
    """We are defining the below function to define a Cramer's V calculation.
    Thus function will be activated when the user selects two categorical
    variables to compare. Therefore, we will call the function following
    the the conditional statement that stipulates, if two categorical variables are
    selected"""
    # Drop NaN values for each column separately before performing the chi-squared test
    valid_data = pd.concat([x, y], axis=1).dropna(subset=[x.name, y.name])
    # Create confusion matrix for chi squared analysis if there are enough values left
    # over after we drop the NaN values.
    if len(valid_data) > 0:
        # Create the confusion matrix
        confusion_matrix = pd.crosstab(valid_data[x.name], valid_data[y.name])
        n = confusion_matrix.sum().sum()
        # Run chi squared analysis on the confusion matrix.
        return chi2_contingency(confusion_matrix), n, confusion_matrix
    return None



def calc_numeric_cat_corr(numeric, categorical):
    """We are defining the function below to run an ANOVA analysis
    when the user selects a numeric variable and a categorical variable to
    compare. Later, we will call this function following a conditional statement
    about the user's selection of a numeric variable and a categorical variable"""
    # Print an error statement if the wrong data types are passed to the function.
    if not pd.api.types.is_numeric_dtype(numeric):
        raise ValueError(f"The numeric variable {numeric.name} is not numeric. "
                         "Please check the data type.")

    if not pd.api.types.is_object_dtype(categorical):
        raise ValueError(f"The categorical variable {categorical.name} is not "
                         "an object (string) type. Please check the data type.")

    # Drop NaN values for numeric and categorical columns separately
    valid_data = pd.concat([numeric, categorical], axis=1).dropna(
        subset=[numeric.name, categorical.name])
    # Create conditional statement to create an ANOVA table if not all of the
    # rows were wiped out when we dropped the NaN values.
    if len(valid_data) > 0:
        # Group the numeric data with their corresponding categorical
        # variable values (i.e. the categorical variable value for their row/car)
        grouped_data = [valid_data[numeric.name][valid_data[categorical.name]
            == category] for category in categorical.unique()]
        # Compare the means of the numeric distributions for each
        # categorical variable. (i.e. calculate the f-statistic and
        # the p_value for the ANOVA)
        return f_oneway(*grouped_data)
        # Return the f-statistic and the p_value
    return None


def calc_spearman_rank_correlation(numeric, categorical):
    """We are giong to define a spearman rank function because cylinders is an ordinal
    categorical variable, which means its relationship to numeric variables
    should be analyzed with a Spearman's r correlation. Later will we will call this
    function following the conditional statement related to the user's choice of the
    'cylinders' variable along with a numeric variable."""
    # Concatinate the two columns selected by the user, drop
    # the NaN values, and assign the result to the valid_data variable.
    valid_data = pd.concat([numeric, categorical], axis=1).dropna(
        subset=[numeric.name, categorical.name])
    # Create a conditional statement that assigns the output of the
    # spearmanr method to the variables spearman_r and
    # p_value if there are rows left in the DataFrame after we
    # dropped the NaN values.
    if len(valid_data) > 0:
        return spearmanr(
            valid_data.iloc[:, 0], valid_data.iloc[:, 1])
    return None

def calc_omega_squared(numeric, categorical):
    """We are defining the function to calculate the omega squared statistic
    for our ANOVA analyses. Omega squared can be thought of as the effect size
    for an ANOVA. Since our data set is so large, statistical significance may
    not entail a meaningful effect. Therefore, we want to take a look at the
    effect sizes for the relationships we find to determine if those relationships
    are meaningful."""
    # Print an error statement if the wrong data types are passed to the function.
    if not pd.api.types.is_numeric_dtype(numeric):
        raise ValueError(
            f"The numeric variable {numeric.name} is not numeric. "
            "Please check the data type.")
    if not pd.api.types.is_object_dtype(categorical):
        raise ValueError(
            f"The categorical variable {categorical.name} is not an "
            "object (string) type. Please check the data type.")
    # Group the numeric data points with their corresponding categorical
    # variable values (i.e. the categorical variable values for their row/car)
    grouped_data = [numeric[
        categorical == category] for category in categorical.unique()]
    # Calculate the sums of squares (SS)
    # Find the Grand mean (i.e. the mean of the entire distribution for the given numeric variable)
    grand_mean = np.mean(numeric)
    # Calculate the between groups sum of squares (i.e. the between-group variance)
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in grouped_data)

    # Calculate the within group sum of squares
    # (i.e. within-group variance or residual variance) and assign it to the ss_within variable
    ss_within = sum(np.sum((
        group - np.mean(group))**2) for group in grouped_data)

    # Calculate the total sum of squares (i.e. total variance in the data)
    ss_total = ss_between + ss_within

    # Calculate the between groups degrees of freedom
    # (df) and assign it to the variable df_between
    df_between = len(grouped_data) - 1  # k - 1, where k is the number of groups

    # Calculate the omega-squared value
    omega_2 = (ss_between - (df_between * ss_within / sum(len(
        group) for group in grouped_data))) / ss_total
    # Return the omega_Squared value
    return omega_2
# If the run analysis box is checked display the relevant
# association statistic

# First let's initialize some variables we are going to use in our conditionals:
categorical_var = None # pylint: disable=C0103
numeric_var = None # pylint: disable=C0103

# The below conditinoal will be met if the user clicks "Run analysis"
if run_anal:
    # Create a conditional statement which dictates
    # that if both variables are categorical
    # association_measure and p_value equal the values
    # returned by the cramers_v() function
    if (vehicles_df[selected_var_1].dtype == 'object') and (
        vehicles_df[selected_var_2].dtype == 'object'):
        # Run Cramér's V for categorical-to-categorical correlation
        (chi2, p_value, _ , _) , total, matrix = cramers_v(
            vehicles_df[selected_var_1], vehicles_df[selected_var_2])
        association_measure = np.sqrt(chi2 / (total * (min(matrix.shape) - 1)))
        # Display the Cramer's V statistic
        st.write(f"Association Measure (Cramer's V): {association_measure}")
    # Create a conditional statement which dictates that if one
    # variable is numeric and the other is categorical
    # the program will assign the numeric variable to
    # numeric_var and it will assign the categorical
    # variable to categorical_var and then it will assign f_statistic
    # and p_value the values returned by the calc_numeric_cat_corr() function.
    elif (vehicles_df[selected_var_1].dtype == 'object') and (
        pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])) or \
         (vehicles_df[selected_var_2].dtype == 'object') and (
            pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        # Assign the numeric variable to numeric_Var and the categorical variable
        # to categorical_var
        if vehicles_df[selected_var_1].dtype == 'object':
            numeric_var = vehicles_df[selected_var_2]
            categorical_var = vehicles_df[selected_var_1]
        elif vehicles_df[selected_var_2].dtype == 'object':
            numeric_var = vehicles_df[selected_var_1]
            categorical_var = vehicles_df[selected_var_2]
        # Since 'cylinders is an ordinal categorical vairable the proepr analysis
        # for its relationship to numeric variables is a Spearman's
        # r correlation analysis. Create a conditional statement
        # that runs a Spearman's r correlation if the user
        # selections 'cylinders' with a numeric variable.
        if (categorical_var.name == 'cylinders') and (
            pd.api.types.is_numeric_dtype(numeric_var)):
            # Call the function to calculate Spearman's
            # correlation for 'cylinders'
            spearman_r, p_value = calc_spearman_rank_correlation(
                numeric_var, categorical_var)
            st.write(f"Spearman Rank: {spearman_r}")
        else:
            # Assign the output of calc_numeric_Cat_Corr()
            # functin to f_statistic and p_value
            f_statistic, p_value = calc_numeric_cat_corr(
                numeric_var, categorical_var)
            # Display the f-statistic
            st.write(f"F-statistic (ANOVA): {f_statistic}")

    # Leftover conditional dictates that we run Pearson's correlation because both
    # variabels are numeric.
    else:
        # Assign output of calc_numeric_corr() function to correlation and p_value
        correlation, p_value = calc_numeric_corr(
            vehicles_df[selected_var_1], vehicles_df[selected_var_2])

        # Display the correlation coefficient here
        st.write(f"Correlation (Pearson's r): {correlation}")

# If the user checks Show significance display message regarding significance
if show_significance:
    st.write(f"P-value: {p_value}")
    if p_value < 0.001:
        st.text('The relationship is statistically significant. '
            'However, for a data set this large a result can be statistically '
            'significant without there being a meaningful effect. If you want '
            'to confirm that the effect was meaningful check the '
            '"Show effect size" box above.')
    else:
        st.write("The correlation is not statistically significant.")

# If the user checks Show effect size display the effect size.
if show_effect_size:
    # The effect size for Cramer's V is just the Craer's V statistic
    if (vehicles_df[selected_var_1].dtype == 'object') and (
        vehicles_df[selected_var_2].dtype == 'object'):
        st.text("For a Cramer's V analysis the association measure is "
                "the effect size.\n Cramer's V Effect Size Guide:\n0.0 - 0.1: Weak "
                "association\n0.1 - 0.3: Small association\n0.3 - 0.5: "
                "Medium association\n0.5 - 1.0: Strong association")
    # The effect size for Spearman's r is just the correlation coefficient
    if (vehicles_df[selected_var_1].name == 'cylinders') and (
        pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])):
        st.text("For a Spearman's r analysis the effect size is the "
                "correlation coefficient.\n\nSpearman's r effect Size "
                "Guide:\n0.0 - 0.1: Weak to no relationship\n0.1 - 0.3: "
                "Weak relationship\n0.3 - 0.5: Moderate relationship\n0.5 - 0.7: "
                "Strong relationship\n0.7 - 0.9: Very strong relationship")
    if (vehicles_df[selected_var_2].name == 'cylinders') and (
        pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        st.text("For a Spearman's r analysis the effect size is the correlation "
                "coefficient.\n\nSpearman's r effect Size Guide:\n0.0 - 0.1: Weak "
                "to no relationship\n0.1 - 0.3: Weak relationship\n0.3 - 0.5: "
                "Moderate relationship\n0.5 - 0.7: Strong relationship\n0.7 - "
                "0.9: Very strong relationship")
    # Display effect size for the ANOVA test
    elif (vehicles_df[selected_var_1].dtype == 'object') and (
        vehicles_df[selected_var_1].name != 'cylinders') and (
            pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])):
        omega_squared = calc_omega_squared(
             vehicles_df[selected_var_2], vehicles_df[selected_var_1])
        st.text(
             f"Effect size (omega-squared): {omega_squared}"
             "\n\nOmega-Squared Effect Size Guide:\n0.01: "
             "Very small effect size\n0.01 - 0.06: Small effect "
             "size\n0.06 - 0.14: Medium effect size\n0.14 - 0.25: "
             "Large effect size\n> 0.25: Very large effect size")
    elif (vehicles_df[selected_var_2].dtype == 'object') and (
        vehicles_df[selected_var_2].name != 'cylinders') and (
            pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        omega_squared = calc_omega_squared(
            vehicles_df[selected_var_1], vehicles_df[selected_var_2])
        st.text("Effect size (omega-squared): "
                f"{omega_squared}\n\nOmega-Squared Effect Size "
                "Guide:\n0.01: Very small effect size\n0.01 - 0.06: "
                "Small effect size\n0.06 - 0.14: Medium effect size\n0.14 "
                "- 0.25: Large effect size\n> 0.25: Very large effect size")
    elif (vehicles_df[selected_var_1].dtype != 'object') and (
        vehicles_df[selected_var_2].dtype != 'object'):
        # The effect size for a pearson's r analysis is just
        # the pearson's r correlation coefficient.
        st.text("For a Pearson's r analysis the "
                "correlation coefficient is the effect size. "
                "\n\nPearson's r Effect Size Guide (the values below "
                "are magnitudes which means they apply to both positive "
                "and negative correlations):\n0 - 0.1: Very small effect "
                "size\n0.1 - 0.3 Small effect size\n0.3 - 0.5: Medium effect "
                "size\n0.5 - 1.0: Large effect size")

# If the user clicks 'Run analysis' present a scatter plot if
# both variables are numeric present a histogram if both
# variables are categorical, and present a series of box plots
# if one variable is numeric and the other is categorical.
if run_anal:
    # If one variable is numeric and the other is categorical
    # persent a series of box plots
    if (vehicles_df[selected_var_1].dtype == 'object') and (
        pd.api.types.is_numeric_dtype(vehicles_df[selected_var_2])):
        fig = px.box(vehicles_df, x=selected_var_1, y=selected_var_2,
                     title=f"Box Plot: {selected_var_1} vs {selected_var_2}")
        st.plotly_chart(fig)
    elif (vehicles_df[selected_var_2].dtype == 'object') and (
        pd.api.types.is_numeric_dtype(vehicles_df[selected_var_1])):
        fig = px.box(vehicles_df, x=selected_var_2, y=selected_var_1,
                     title=f"Box Plot: {selected_var_2} vs {selected_var_1}")
        st.plotly_chart(fig)
    # If both variables are objects display a histogram.
    elif (vehicles_df[selected_var_1].dtype == 'object') and (
        vehicles_df[selected_var_2].dtype == 'object'):
        # Histogram or Bar plot for Categorical-to-Categorical correlation
        fig = px.histogram(
            vehicles_df, x=selected_var_1, color=selected_var_2,
                            title=f"Histogram: {selected_var_1} by {selected_var_2}")
        st.plotly_chart(fig)
    # If both variables are numeric present a scatter plot.
    else:
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
