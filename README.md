# Used Car Traits: Statistical Associations
Overview:
This project provides a Streamlit-based app that allows users to explore the relationships between various used car attributes. Users can select two variables from the dataset, and the app will run the appropriate statistical tests (Pearson's correlation, Spearman's rank correlation, ANOVA, or Cramér's V) based on the types of variables selected. The app provides users with the statistical association measures, p-values, effect sizes, and visualizations to help them make informed decisions when searching for a car or analyzing used car market trends.


Interactive Analysis: Users can choose two variables (numeric or categorical) and see how they are related through different statistical measures.
The app supports the following tests:
Pearson's r: for correlation between two numeric variables,
Spearman's rank correlation (specifically used when one of the variables is the ordinal categorical variable 'cylinders'),
ANOVA (to compare the means of a numeric variable across different categories in a categorical variable), and
Cramér's V (to measure the association between two categorical variables).

The app calculates effect sizes for the statistical tests to give a sense of the strength of the relationships.
Based on the user’s variable selection, different visualizations are generated:
scatter plots for numeric-to-numeric relationships,
box plots for numeric-to-categorical relationships, and
histograms for categorical-to-categorical relationships.

Users can view a correlation matrix for numeric variables to quickly assess the relationships between multiple numeric columns.
The app analyzes a dataset of used cars (contained in vehicles_us.csv). The dataset contains various features of used cars, including but not limited to:

Manufacturer
Model
Price
Year
Mileage
Engine size
Number of cylinders (categorical: ordinal variable)
4WD status (yes/no)


Some data preprocessing has been done to handle missing or non-standard values. For example, the is_4wd column was converted to yes or no based on 1.0 and NaN values, and string columns were standardized to lowercase with spaces trimmed. Additionally, the cylinders column was converted to an object datatype for analysis.


An initial exploratory data analysis (EDA) was conducted to better understand the dataset. This was done without interactivity, providing a brief overview of the data's characteristics and relationships between its variables.

The EDA included:

Summary statistics of numeric and categorical columns.
Visualizations of the distributions of key variables (e.g., price, mileage, etc.).
Identification of missing values and non-standard placeholders.
Preliminary analysis of the relationships between variables, including correlations and potential outliers.
While the EDA provides a detailed, static overview of the data, the app allows for interactive exploration of statistical associations between any two variables.

Installation
To run the app locally, follow these steps:

Clone this repository:

git clone https://github.com/KunkelBrett/sprint_4_project
Navigate to the project directory:


after cloning the repository move into the project directory
using
"cd sprint_4_project"

Create and activate a virtual environment (optional but recommended)
using
"pip install pipenv"
Then run pipenv shell
Install the required dependencies with
"pip install -r requirements.txt"
Run the Streamlit app -
"streamlit run app.py"
Visit http://localhost:8501 in your web browser to access the app.

Requirements
python 3.x
streamlit
pandas
numPy
plotly.express
scipy.stats

and if you want to run the EDA as well you will need to add:

seaborn
statsmodels.api
statsmodels.formula.api


To try the app choose two variables from the dataset.
Click the "Run analysis" checkbox to run the statistical test, and the app will display the correlation coefficient (or test statistic), along with a plot. IF you want to check wheter or not results were statistically significant click "Show statistical significance" and the app will display the p-value along with a statement about whether or not the results were statistically siginificant and how to interpret them. If you want to see the effect size click "Show effect size."
Visualizations including scatter plots, box plots, or histograms will be displayed based on the variable types you selected.

For example, if you choose:

Variable 1: price (numeric)
Variable 2: cylinders (ordinal categorical)
The app will run a Spearman's rank correlation to assess the relationship between price and cylinders and display the corresponding correlation coefficient, p-value, and effect size. It will also display a box plot to visualize the relationship between these variables.