# What is Linear Regression

It's a stats model which looks at the linear relationship between a dependent variable and one or more independent variables. The model estimates the slope and intercept of the line of best fit representing the relationship between variables.

# Goal of the analysis

The goal for this analysis is to create a model that estimates the sales revenue of a retail store. We have identified UnitsSold as our independent variable and Sales Revenue as our dependent.

### Approach to the Analysis

   1. **Loading the data into SAS Studio**
   2. **Cleaning the data.**
   3. **Data Visualization and Analysis**
   4. **Prediction by Linear Regression**

# Importing the libraries
```python
 #first we import our libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

## Interpretation
- **Pandas**: a python library used for working with datasets.
- **Numpy**: a python library used for working with large multi-dimensional arrays along with a collection of math functions.
- **Matplotlib**: a python library used for creating static, animated and interactive visualizations in Python.
- **Seaborn**: a python library used for generating more aesthetically pleasing visualizations compared to matplotlib.
- **Scipy**: a python library used for used for scientific computing in python.


# Cleaning the data
```python
# Load data into pandas DataFrame
kaggle = pd.read_csv('mock_kaggle.csv')

# Rename columns of the DataFrame
kaggle.rename(columns={'data': 'Date', 'venda': 'UnitsSold', 'estoque': 'Stock', 'preco': 'Price'}, inplace=True)

# Create new column to calculate revenue
kaggle['Revenue'] = kaggle['UnitsSold'] * kaggle['Price']

# Convert dates in a column into months
kaggle['Month'] = pd.to_datetime(kaggle['Date']).dt.month_name()

# Identify which dates in a column correspond to which weekdays
kaggle['Day_of_Week'] = pd.to_datetime(kaggle['Date']).dt.day_name()

# Assign seasons to the corresponding months
kaggle['Month_Num'] = pd.to_datetime(kaggle['Date']).dt.month
kaggle['Season'] = np.where(kaggle['Month_Num'].isin([12, 1, 2]), 'Winter',
                             np.where(kaggle['Month_Num'].isin([3, 4, 5]), 'Spring',
                                      np.where(kaggle['Month_Num'].isin([6, 7, 8]), 'Summer', 'Fall')))

# Print the database
kaggle
```
# Simple Statistical Analysis

After cleaning the data, we can then perform some simple statistics. We could calculate the average revenue by Week, Month, and Season.

```python
# Calculate avg rev by weekday, month, and season
avg_rev_weekday = kaggle.groupby('Day of Week')['Revenue'].mean()reset_index()
avg_rev_month = kaggle.groupby('Month')['Revenue'].mean().reset_index()
avg_rev_season = kaggle.groupby('Season')['Revenue'].mean().reset_index()
```

