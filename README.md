# Used Car Traits: Statistical Associations
## Overview

This project provides a Streamlit-based app that allows users to explore the relationships between various used car attributes. Users can select two variables from the dataset, and the app will run the appropriate statistical tests (Pearson's correlation, Spearman's rank correlation, ANOVA, or Cramér's V) based on the types of variables selected. The app provides users with the statistical association measures, p-values, effect sizes, and visualizations to help them make informed decisions when searching for a car or analyzing used car market trends.

### Interactive Analysis
Users can choose two variables (numeric or categorical) and see how they are related through different statistical measures.
The app supports the following tests: Pearson's r (for correlation between two numeric variables),
Spearman's rank correlation (specifically used when one of the variables is the ordinal categorical variable 'cylinders'),
ANOVA (to compare the means of a numeric variable across different categories in a categorical variable), and
Cramér's V (to measure the association between two categorical variables).

The app calculates effect sizes for the statistical tests to give a sense of the strength of the relationships.
Based on the user’s variable selection, different visualizations are generated:
1. scatter plots for numeric-to-numeric relationships.
2. box plots for numeric-to-categorical relationships.
3. histograms for categorical-to-categorical relationships.
4. Correlation matrix and heat map for the numeric variables.

Users can view a correlation matrix for numeric variables to quickly assess the relationships between multiple numeric columns.
The app analyzes a dataset of used cars (contained in vehicles_us.csv). The dataset contains the following information about used cars:

- price
- model year
- model
- condition
- cylinders
- fuel
- odometer
- transmission
- type
- paint color
- if 4 wheel drive (yes or no)
- date posted
- days listed
- season posted
- manufacturer


### Data Preparation and Preliminary Data Analysis
Some data preprocessing was done to handle missing or non-standard values. For example, the is_4wd column was converted to yes or no based on 1.0 and NaN values, and string columns were standardized to lowercase with spaces trimmed. Additionally, for the app, the cylinders column was converted to an object datatype for analysis.


An initial exploratory data analysis (EDA) was conducted to better understand the dataset. This was done without interactivity, providing a brief overview of the data's characteristics and relationships between its variables.

The EDA included:
- Summary statistics of numeric and categorical columns.
- Visualizations of the distributions of each variable.
- Identification of missing values and non-standard placeholders.
- Preliminary analysis of the relationships between variables, including correlations and potential outliers.

While the EDA provides a detailed, static overview of the data, the app allows for interactive exploration of statistical associations between any two variables.

### App Instillation
To run the app locally, follow these steps:

1. Clone this repository: git clone https://github.com/KunkelBrett/sprint_4_project
2. Move into the project directory by running "cd sprint_4_project" in your command like interface.
3. Create and activate a virtual environment (optional but recommended) by running "pip install pipenv" in your command line interface.
4. Run "pipenv shell" in your command line interface.
5. Install the required dependencies by running "pip install -r requirements.txt" inyour command line interface.
6. Run the Streamlit app by entering "streamlit run app.py" in your command line interface.

### Requirements
- python 3.x
- streamlit
- pandas
- numPy
- plotly.express
- scipy.stats

Each of the above is contained in the requirements.txt file. However, If you want to run the EDA as well you will need to add:
- seaborn
- statsmodels.api
- statsmodels.formula.api

### App Demo
Try the app for yourself here: https://sprint-4-project-17.onrender.com/
Start by choosing two variables from the dataset. Then, click the "Run analysis" checkbox to run the statistical test, and the app will display the correlation coefficient (or test statistic), along with a plot. The type of plot that is displayed will be determined by the vairable type you selected. If you want to check wheter or not results were statistically significant click "Show statistical significance" and the app will display the p-value along with a statement about whether or not the results were statistically siginificant and how to interpret them. If you want to see the effect size click "Show effect size."

For example, if you choose:
Variable 1: price (numeric)
Variable 2: cylinders (ordinal categorical)
the app will run a Spearman's rank correlation to assess the relationship between price and cylinders and display the corresponding correlation coefficient, p-value, and effect size. It will also display a box plot to visualize the relationship between these variables.