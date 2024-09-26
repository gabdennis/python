
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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


#Print the database
kaggle

# Calculate the Average Revenue by Weekday
avg_revenue_weekday = kaggle.groupby('Day_of_Week')['Revenue'].mean().reset_index()

# Plot a histogram based on the Average Revenue by Weekday
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_revenue_weekday, x='Day_of_Week', y='Revenue')
plt.title('Average Revenue by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Average Revenue')
plt.xticks(rotation=45)
plt.show()

# Calculate the Average Revenue by Month
avg_revenue_month = kaggle.groupby('Month')['Revenue'].mean().reset_index()

# Plot a histogram based on the Average Revenue by Month
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_revenue_month, x='Month', y='Revenue')
plt.title('Average Revenue by Month')
plt.xlabel('Month')
plt.ylabel('Average Revenue')
plt.xticks(rotation=45)
plt.show()

# Calculate the Average Revenue by Season
avg_revenue_season = kaggle.groupby('Season')['Revenue'].mean().reset_index()

# Plot a histogram based on the Average Revenue by Season
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_revenue_season, x='Season', y='Revenue')
plt.title('Average Revenue by Season')
plt.xlabel('Season')
plt.ylabel('Average Revenue')
plt.show()

sns.pairplot(kaggle, kind='scatter', plot_kws = {'alpha':0.04})

sns.lmplot(x='UnitsSold',
           y='Revenue', 
           data=kaggle,
           scatter_kws={'alpha': 0.3})



from sklearn.model_selection import train_test_split

x = kaggle[['UnitsSold','Price','Stock']]
y = kaggle['Revenue']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

#training the model
from sklearn.linear_model import LinearRegression

ln = LinearRegression()

ln.fit(x_train, y_train)

ln.coef_

cdf = pd.DataFrame(ln.coef_, x.columns, columns=['Coef'])
cdf

#predictions
predictions = ln.predict(x_test)
predictions

sns.scatterplot(x=predictions,y=y_test)
plt.xlabel('Predictions')
plt.title('Evaluation of LM Model')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

print('Mean Absolute Error: ', mean_absolute_error(y_test, predictions))
print('Mean Squared Error: ', mean_squared_error(y_test, predictions))
print('RMSE: ',math.sqrt(mean_squared_error(y_test, predictions)))

#residuals
residuals = y_test - predictions

sns.displot(residuals, bins=20, kde=True)

import pylab
import scipy.stats as stats

stats.probplot(residuals, dist='norm', plot=pylab)
pylab.show()

# Pearson Correlation
correlation = kaggle[['Revenue', 'UnitsSold']].corr().iloc[0, 1]
print(f"Pearson Correlation Between Revenue and Sales: {correlation}")

# Predicting Revenue based on 5000 units sold
units_sold = 5000
# Assuming the regression coefficients are known (Intercept and slope)
intercept = 0  # Replace with actual intercept from regression output
slope = 1.547899  # Replace with actual slope from regression output
new_revenue = intercept + slope * units_sold
print(f"Predicted Revenue for {units_sold} units sold: {new_revenue}")