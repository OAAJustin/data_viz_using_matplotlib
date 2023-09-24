## Import libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

## Line plots display numerical values on one axis, and categorical values on the other. 
## More commonly used to keep track of changes over time, but can be used in the same way as bar charts. 

dataframe = pd.read_csv('gdp_csv.csv')
print(dataframe)

## Filter the dataset to display the GDP of the EU and the US, year over year.

dataframe_eu = dataframe.loc[dataframe['Country Name'] == 'European Union']
dataframe_na = dataframe.loc[dataframe['Country Name'] == 'North America']
dataframe_sa = dataframe.loc[dataframe['Country Name'] == 'South Asia']
dataframe_ea = dataframe.loc[dataframe['Country Name'] == 'East Asia & Pacific']

print(dataframe_eu.head())
print(dataframe_na.head())
print(dataframe_sa.head())
print(dataframe_ea.head())

## Note that 'Country Name' could be misleading as these do not reference Countries, but Continents or Regions. 

fig, ax = plt.subplots()
ax.plot(dataframe_eu['Year'], dataframe_eu['Value'], label = 'European Union GDP per Year')
ax.plot(dataframe_na['Year'], dataframe_na['Value'], label = 'North America GDP per Year')
ax.plot(dataframe_sa['Year'], dataframe_sa['Value'], label = 'South Asia GDP per Year')
ax.plot(dataframe_ea['Year'], dataframe_ea['Value'], label = 'East Asia & Pacific GDP per Year')

ax.legend()
plt.show()

## PLOTTING A LINE PLOT LOGARITHMICALLY

x = np.linspace(0, 5, 10)
y = np.exp(x)

plt.plot(x,y)
plt.show()

## Change the Y-Axis to logarithmic

x = np.linspace(0, 5, 10)
y = np.exp(x)

plt.yscale('log')
plt.plot(x,y)
plt.show()

## Use GDP dataset to visualize the plot on a logartihmic scale

fig, ax = plt.subplots()
ax.plot(dataframe_eu['Year'], dataframe_eu['Value'], label = 'European Union GDP per Year')
ax.plot(dataframe_na['Year'], dataframe_na['Value'], label = 'North America GDP per Year')
ax.plot(dataframe_sa['Year'], dataframe_sa['Value'], label = 'South Asia GDP per Year')
ax.plot(dataframe_ea['Year'], dataframe_ea['Value'], label = 'East Asia & Pacific GDP per Year')

plt.yscale('log')
ax.legend()
plt.show()

## CUSTOMIZING LINE PLOTS IN MATPLOTLIB

## Pass in aruments into the plot() function such as linewidth, linestyle or color

fig, ax = plt.subplots()
ax.plot(dataframe_eu['Year'], dataframe_eu['Value'], label = 'European Union GDP per Year', linestyle = 'dotted', color = 'k', linewidth = 1)
ax.plot(dataframe_na['Year'], dataframe_na['Value'], label = 'North America GDP per Year', linestyle = 'solid', color = 'g', linewidth = 1)
ax.plot(dataframe_sa['Year'], dataframe_sa['Value'], label = 'South Asia GDP per Year', linestyle = 'dashed', color = 'r', linewidth = 1.5)
ax.plot(dataframe_ea['Year'], dataframe_ea['Value'], label = 'East Asia & Pacific GDP per Year', linestyle = 'dashdot', color = 'b', linewidth = 2)

plt.yscale('log')
ax.legend()
plt.show()

## https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html#sphx-glr-gallery-lines-bars-and-markers-linestyles-py

## PLOT MULTIPLE LINE PLOTS WITH DIFFERENT SCALES

## Plot an exponentially increasing sequence of numbers, and plot it next to another line on the same Axes, linearly
## This example is what it would look like without logarithmic plotting for the exponential sequence

linear_sequence = np.linspace(0, 10, 10)
exponential_sequence = np.exp(linear_sequence)

fig, ax = plt.subplots()

ax.plot(linear_sequence)
ax.plot(exponential_sequence)
plt.show()

## This example is when the logarithm is applied to the Y Axis

linear_sequence = [1, 2, 3, 4, 5, 6, 7, 10, 15, 20]
exponential_sequence = np.exp(np.linspace(0, 10, 10))

fig, ax = plt.subplots()

ax.plot(linear_sequence, color = 'red')
ax.tick_params(axis = 'y', labelcolor = 'red')

ax2 = ax.twinx()

ax2.plot(exponential_sequence, color = 'green')
ax2.set_yscale('log')
ax2.tick_params(axis = 'y', labelcolor = 'green')

plt.show()

## BAR PLOT

## Bar plots display numerical quantities on one axis and categorical variables on the other, letting you see how many occurrences there are for the different categories. 
## Bar charts can be used for visualizing a time series, as well as just cateogrical data. 

## PLOTTING A BAR PLOT
## Pass the bar() function on the PyPlot or Axes instance and pass the categorical and numerical variables we'd like to visualize

x = ['A', 'B', 'C']
y = [1, 5, 3]

plt.bar(x, y)
plt.show()

## https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021
## Use a dataset for a world happiness report to visualize

df_world_happiness = pd.read_csv('world-happiness-report-2021.csv')

fig, ax = plt.subplots()
ax.bar(df_world_happiness['Regional indicator'], df_world_happiness['Freedom to make life choices'], label = 'Freedom score')
ax.legend()

plt.xticks(rotation = 10, wrap = True)
plt.show()

## Describe the information under the 'Freedom to make life choices' column

print(df_world_happiness['Freedom to make life choices'].describe())

## Data could be displayed behind the bars presented forward as they are the highest of value, be sure to identify these potential events when processing the data
## Add median and mean lines to the plot

df_world_happiness = pd.read_csv('world-happiness-report-2021.csv')
mean = df_world_happiness['Freedom to make life choices'].mean()
median = df_world_happiness['Freedom to make life choices'].median()

fig, ax = plt.subplots()
ax.bar(df_world_happiness['Regional indicator'], df_world_happiness['Freedom to make life choices'], label = 'Freedom score')
ax.axhline(mean, 0, 1, color = 'red', label = 'Mean Value for Dataset')
ax.axhline(median, 0, 1, color = 'green', label = 'Median Value for Dataset')
ax.legend()

plt.xticks(rotation = 10, wrap = True)
plt.show()

## Group the dataset by the regional indicators, and calculate the mean for each group and store it in the variable in a new dataframe named df_means

df_means = df_world_happiness.groupby('Regional indicator').mean()
print(df_means)
print(df_means['Freedom to make life choices'])

## Plot the aggregated data for each region

df_means = df_world_happiness.groupby('Regional indicator').mean()
mean = df_world_happiness['Freedom to make life choices'].mean()
median = df_world_happiness['Freedom to make life choices'].median()

fig, ax = plt.subplots()
ax.bar(df_means.index, df_means['Freedom to make life choices'], label = 'Freedom Score')

ax.axhline(mean, 0, 1, color = 'red', label = 'Mean')
ax.axhline(median, 0, 1, color = 'green', label = 'Median')

ax.legend()
plt.xticks(rotation = 10, clip_on = True)

plt.show()

## As aggregate is just an average of the values, one value could be a 0 and one value could be a 1 making a 0.5 value aggregate. Error bars could fix this issue

## BAR PLOT WITH ERROR BARS IN MATPLOTLIB

## There are two estimates you typically use for error bars:
##      o Standard Deviation - https://en.wikipedia.org/wiki/Standard_deviation
##      o Standard Error - https://en.wikipedia.org/wiki/Standard_error

df_world_happiness_stds = df_world_happiness.groupby('Regional indicator').std()
print(df_world_happiness_stds)
print(df_world_happiness_stds['Freedom to make life choices'])

## Visualize error bars with the bar() function and pass the categorical values and numerical values along with the yerr argument as we are plotting on the y axis
## if we were plotting horizontally or on the x axis then we would use the xerr argument

df_world_happiness_mean = df_world_happiness.groupby('Regional indicator').mean()
df_world_happiness_stds = df_world_happiness.groupby('Regional indicator').std()

mean = df_world_happiness['Freedom to make life choices'].mean()
median = df_world_happiness['Freedom to make life choices'].median()

fig, ax = plt.subplots()

ax.bar(df_world_happiness_mean.index, df_world_happiness_mean['Freedom to make life choices'], 
       label = 'Freedom score', yerr = df_world_happiness_stds['Freedom to make life choices'])

ax.axhline(mean, 0, 1, color = 'red', label = 'Mean')
ax.axhline(median, 0, 1, color = 'green', label = 'Median')

ax.legend()
plt.xticks(rotation = 10, wrap = True)
plt.show()

## CHANGING BAR PLOT COLORS
## Color palette in RGB and Hexadecimal - https://www.rapidtables.com/web/color/RGB_Color.html

fig, ax = plt.subplots()

colors = ['#DC143C', '#FFA07A', '#FFD700', '#008000', '#4682B4', '#4B0082', '#FFB6C1', '#F5F5DC', '#D2691E', '#DCDCDC']
ax.bar(df_world_happiness_mean.index, df_world_happiness_mean['Freedom to make life choices'], 
       label = 'Freedom score', yerr = df_world_happiness_stds['Freedom to make life choices'], color = colors)

ax.axhline(mean, 0, 1, color = 'red', label = 'Mean')
ax.axhline(median, 0, 1, color = 'green', label = 'Median')

ax.legend()
plt.xticks(rotation = 10, wrap = True)
plt.show()

## PLOTTING HORIZONTAL BAR PLOTS

## Plot bars horizontally by calling the barh() function. It accepts the same arguments, though, we'll have to change our yerr to xerr, and our axhline() to axvline()

fig, ax = plt.subplots()

colors = ['#DC143C', '#FFA07A', '#FFD700', '#008000', '#4682B4', '#4B0082', '#FFB6C1', '#F5F5DC', '#D2691E', '#DCDCDC']
ax.barh(df_world_happiness_mean.index, df_world_happiness_mean['Freedom to make life choices'], 
       label = 'Freedom score', xerr = df_world_happiness_stds['Freedom to make life choices'], color = colors)

ax.axvline(mean, 0, 1, color = 'red', label = 'Mean')
ax.axvline(median, 0, 1, color = 'green', label = 'Median')

ax.legend()
plt.xticks(rotation = 10, wrap = True)
plt.show()

## SORTING BAR ORDER

df_means = df_world_happiness.groupby('Regional indicator').mean()
df_means.sort_values('Freedom to make life choices', inplace= True)

df_world_happiness_stds = df_world_happiness.groupby('Regional indicator').std()

df_world_happiness_stds.rename(columns = {'Freedom to make life choices': 'Freedom std'}, inplace= True)
df_means.rename(columns = {'Freedom to make life choices': 'Freedom mean'}, inplace= True)

df_merged = pd.merge(df_world_happiness_stds['Freedom std'], df_means['Freedom mean'], right_index = True, left_index = True)

print(df_merged)

## 

df = pd.read_csv('world-happiness-report-2021.csv')

df_means = df.groupby('Regional indicator').mean()
df_stds = df.groupby('Regional indicator').std()

df_stds.rename(columns = {'Freedom to make life choices': 'Freedom std'}, inplace= True)
df_means.rename(columns = {'Freedom to make life choices': 'Freedom mean'}, inplace= True)

df_merged = pd.merge(df_stds['Freedom std'], df_means['Freedom mean'], right_index= True, left_index= True)
df_merged.sort_values('Freedom mean', inplace= True)

mean = df['Freedom to make life choices'].mean()
median = df['Freedom to make life choices'].median()

fig, ax = plt.subplots()

colors = ['#DC143C', '#FFA07A', '#FFD700', '#008000', '#4682B4', '#4B0082', '#FFB6C1', '#F5F5DC', '#D2691E', '#DCDCDC']
ax.barh(df_merged.index, df_merged['Freedom mean'], label = 'Freedom score', xerr = df_merged['Freedom std'], color = colors)

ax.axvline(mean, 0, 1, color = 'red', label = 'mean')
ax.axvline(median, 0, 1, color = 'green', label = 'median')

ax.legend()
plt.xticks(rotation = 10, wrap = True)
plt.show()

## PIE CHART 

## Pie charts represent data broken down into categories / labels. Used to visualize proportional data such as percentages

## PLOTTING A PIE CHART

## Plot a pie chart by calling the pie() function onf the PyPlot or Axes instance. The only mandatory argument is the data we'd like to plot

x = [15, 25, 25, 30, 5]

fig, ax = plt.subplots()
ax.pie(x)
plt.show()

## Add additional data

x = [15, 25, 25, 30, 5]
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']

fig, ax = plt.subplots()
ax.pie(x, labels = labels)
plt.show()

## CUSTOMIZING PIE CHARTS

## CHANGING PIE CHART COLORS

## Supply an array of colors to the colors argument while plotting

x = [15, 25, 25, 30, 5]
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
colors = ['#FF0000', '#008000', '#0000FF', '#FFFF00', '#FF00FF']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors)
ax.set_title('Survey Responses')
plt.show()

## SHOWING PERCENTAGES ON SLICES

## Add numerical percentages to each slice by calling the autopct argunment. 

x = [15, 25, 25, 30, 5]
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
colors = ['#FF0000', '#008000', '#0000FF', '#FFFF00', '#FF00FF']

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct = '%.0f%%')
ax.set_title('Survey Responses')
plt.show()

## EXPLODE / HIGHLIGHT WEDGES

x = [15, 25, 25, 30, 5]
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
colors = ['#FF0000', '#008000', '#0000FF', '#FFFF00', '#FF00FF']
explode = [0, 0, 0, 0, 0.2]

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct = '%.0f%%', explode = explode)
ax.set_title('Survey Responses')
plt.show()

## ADDING A SHADOW 

## Add a shadow to Matplotlib pie chart by setting the shadow argument to True

x = [15, 25, 25, 30, 5]
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
colors = ['#FF0000', '#008000', '#0000FF', '#FFFF00', '#FF00FF']
explode = [0, 0, 0, 0, 0.2]

fig, ax = plt.subplots()
ax.pie(x, labels = labels, colors = colors, autopct = '%.0f%%', explode = explode, shadow = True)
ax.set_title('Survey Responses')
plt.show()

## ROTATING PIE CHART

x = [15, 25, 25, 30, 5]
labels = ['Very Likely', 'Likely', 'Unsure', 'Unlikely', 'Very Unlikely']
colors = ['#FF0000', '#008000', '#0000FF', '#FFFF00', '#FF00FF']
explode = [0, 0, 0, 0, 0.2]

fig, ax = plt.subplots()
ax.pie(x, labels = labels, 
       colors = colors, 
       autopct = '%.0f%%', 
       explode = explode, 
       shadow = True,
       startangle = 180)

ax.set_title('Survey Responses')
plt.show()

## VISUALIZE THE PERCEPTIONS OF CORRUPTION FROM WORLD HAPPINESS REPORT DATASET

df = pd.read_csv('world-happiness-report-2021.csv')
df_mean = df.groupby("Regional indicator").mean()
print(df_mean['Perceptions of corruption'])

## FORM A PIE CHART

df = pd.read_csv('world-happiness-report-2021.csv')
df_sum = df.groupby("Regional indicator").sum()

fig, ax = plt.subplots()

colors = ['#FF0000', '#008000', '#0000FF', '#FFFF00', '#FF00FF']
ax.pie(df_sum['Perceptions of corruption'], 
       labels = df_sum.index, 
       colors = colors, 
       autopct = '%.0f%%')
plt.show()

## SCATTER PLOT

## Scatter Plots, one of the most important and commonly used plot type, visualizes the relationship between two numerical features. 
## These variables can be dependent, or independent of each other. 
## Scatter Plots are really useful for their ability to visualize relationships and correlation between multiple variables, denoted by markers. 
## Other notable plots that visualize correlation are Correlation Heatmaps and Correlation Matrices
## A variation of Scatter Plots, called Bubble Plots can include the size of a third variable into account by manipulating the size of the markers used to 
## represent the relationship between two other variables. 
    
## DATASET: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

## Plot the living area above ground and the sale price on the x and y arguments using the scatter() function

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(x = df['Gr Liv Area'], y = df['SalePrice'])
plt.xlabel('Living Area Above Ground')
plt.ylabel('House Price')

plt.show()

## Plot the relationships between the area above ground level and overall quality against the sale price

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(2, figsize = (10, 6))
ax[0].scatter(x = df['Gr Liv Area'], y = df['SalePrice'])
ax[0].set_xlabel('Living Area Above Ground')
ax[0].set_ylabel('House Price')

ax[1].scatter(x = df['Overall Qual'], y = df['SalePrice'])
ax[1].set_xlabel('Overall Quality')
ax[1].set_ylabel('House Price')

plt.show()

## The second scatter plot is called a Strip Plot. The markers can be customized to create better visuals by accessing the marker argument 
## inside of the scatter function and placing a string of the marker you would like to use

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(2, figsize = (10, 6))
ax[0].scatter(x = df['Gr Liv Area'], y = df['SalePrice'])
ax[0].set_xlabel('Living Area Above Ground')
ax[0].set_ylabel('House Price')

ax[1].scatter(x = df['Overall Qual'], y = df['SalePrice'], marker = '_')
ax[1].set_xlabel('Overall Quality')
ax[1].set_ylabel('House Price')

plt.show()

## CUSTOMIZING SCATTER PLOT IN MATPLOTLIB

## Color and alpha are optional arguments in the scatter function for customizing. 

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(figsize = (10, 6))

ax.scatter(x = df['Gr Liv Area'], y = df['SalePrice'],
           color = 'blue', 
           edgecolors = 'white', 
           linewidth = 0.1,
           alpha = 0.7)

plt.show()

## Customization can be for styling purposes but also for informative purposes. Inject more information through color-coding a certain class/feature of each house in the dataset.
## Define the colors in a dictionary to map the class to their specified color

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(figsize = (10, 6))

colors = {'Normal': '#FF0000',
          'Abnorml': '#008000',
          'Family': '#0000FF',
          'Partial': '#800080',
          'Alloca': '#FFFF00',
          'AdjLand': '#FF00FF'}

ax.scatter(x = df['Overall Qual'], y = df['SalePrice'],
           marker = '_',
           color = df['Sale Condition'].apply(lambda x: colors[x]))
           
ax.set_xlabel('Overall Quality')
ax.set_ylabel('House Price')

plt.show()

## CHANGE MARKER SIZE IN MATPLOTLIB SCATTER PLOT

## DATASET: https://www.kaggle.com/datasets/unsdsn/world-happiness

## Use THE 2019 World Happiness Dataset to visualize the GDP per Capita and Generosity scores

df = pd.read_csv('C:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(x = df['GDP per capita'], y = df['Generosity'])
plt.xlabel('GDP Per Capita')
plt.ylabel('Generosity Score')

plt.show()

## Bubble Plots are called Bubble Plots because they resemble a swarm of bubbles. 
## They're typically each differently sized and transparent to a degree. 
## Increase the size of each marker, based on the third variable - the perceived Happiness of the inhabitants of that country by accessing the s argument in scatter()

df = pd.read_csv('C:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(x = df['GDP per capita'], y = df['Generosity'], s = df['Score']*25)
plt.xlabel('GDP Per Capita')
plt.ylabel('Generosity Score')

plt.show()

## Update the block by creating a new list, based on the values of the Happiness Score but introduce an exponent to the power of 2, for a non-linear growth pattern
## where really small values will get smaller, while higher values will get bigger

df = pd.read_csv('C:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

size = df['Score'].to_numpy()
s = [3*s**2 for s in size]

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(x = df['GDP per capita'], y = df['Generosity'], s = s, alpha = 0.5)
plt.xlabel('GDP Per Capita')
plt.ylabel('Generosity Score')

plt.show()

## Raise to the power of 3

df = pd.read_csv('C:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

size = df['Score'].to_numpy()
s = [3*s**3 for s in size]

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(x = df['GDP per capita'], y = df['Generosity'], s = s, alpha = 0.5)
plt.xlabel('GDP Per Capita')
plt.ylabel('Generosity Score')

plt.show()

## EXPLORING RELATIONSHIPS WITH SCATTER PLOTS

## Visualize if the Lot Area has increased over the years, the trend of Total Basement Area over the years and change the marker size of the Living Area
## against SalePrice in the scatter plot

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(4, figsize = (10,6))

size = df['Overall Qual'].to_numpy()
s = [s**1.2 for s in size]

ax[0].scatter(x = df['Gr Liv Area'], y = df['SalePrice'], s = s)
ax[0].set_xlabel('Living Area Above Ground')
ax[0].set_ylabel('House Price')

colors = {'Normal': '#FF0000',
          'Abnorml': '#008000',
          'Family': '#0000FF',
          'Partial': '#800080',
          'Alloca': '#FFFF00',
          'AdjLand': '#FF00FF'}

ax[1].scatter(x = df['Overall Qual'], y = df['SalePrice'],
              marker = '_',
              color = df['Sale Condition'].apply(lambda x: colors[x]))

ax[1].set_xlabel('Overall Quality')
ax[1].set_ylabel('House Price')

ax[2].scatter(x = df['Year Built'], y = df['Lot Area'], alpha = 0.6, s = 25)
ax[2].set_xlabel('Year Built')
ax[2].set_ylabel('Lot Area')

ax[3].scatter(x = df['Year Built'], y = df['Total Bsmt SF'], alpha = 0.6, s = 25)
ax[3].set_xlabel('Year Built')
ax[3].set_ylabel('Total Basement Area')

plt.show()

## Visualize the total Lot Area, the Total Basement Area and the Garage Area over the years 

df = pd.read_csv('AmesHousing.csv')

fig, ax = plt.subplots(figsize = (10, 6))

ax.scatter(x = df['Year Built'], y = df['Lot Area'],
           alpha = 0.6, s = 25, color = 'red', label = 'Lot Area')

ax.scatter(x = df['Year Built'], y = df['Total Bsmt SF'],
           alpha = 0.6, s = 25, color = 'blue', label = 'Total Basement Area')

ax.scatter(x = df['Year Built'], y = df['Garage Area'],
           alpha = 0.6, s = 25, color = 'green', label = 'Garage Area')

ax.set_xlabel('Year Built')
ax.legend()

plt.show()

## The outliers in this example are skewing the results. Limiting the quantiles of a DataFrame is a solution, removing some of the more offending outliers.
## Alternatively, calculating the z-score (Standard Score) of the columns, and drop them based on the result. 
## Z-Score is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured
## Z-score can be caluclated through the stats module of scipy, a Python library dedicated for science, mathematics, and engineering, mainly used for scientific
## and technical computing. 
## Z = \frac{x - \mu}{\sigma} - mu is the mean, sigma for standard deviation where x is the value we're calculating
## In most cases, the Z-score will be between -3 and 3, which means that outliers will most likely be above the 3 mark. Outliers are subjective.

df = pd.read_csv('AmesHousing.csv')

df = df[['Year Built', 'Lot Area', 'Total Bsmt SF', 'Garage Area']].copy()

print(df.shape)

df = df[df.apply(lambda x : np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

print(df.shape)

## Plot the truncated DataFrame on two Axes, keeping the more similar features like Garage Areas and Basement Areas on one Axes, while plotting the Lot Area on the other

df = pd.read_csv('AmesHousing.csv')
df = df[['Year Built', 'Lot Area', 'Total Bsmt SF', 'Garage Area']].copy()
df = df[df.apply(lambda x : np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

fig, ax = plt.subplots(ncols = 2)

ax[0].scatter(x = df['Year Built'], y = df['Lot Area'],
              alpha = 0.6, s = 25, label = 'Lot Area')

ax[1].scatter(x = df['Year Built'], y = df['Total Bsmt SF'],
              alpha = 0.6, s = 25, label = 'Total Basement Area')

ax[1].scatter(x = df['Year Built'], y = df['Garage Area'], 
              alpha = 0.6, s = 25, label = 'Garage Area')

ax[0].set_xlabel('Year Built')
ax[1].set_xlabel('Year Built')
ax[1].legend()

plt.show()

## HISTOGRAM PLOT

## Histogram plots are a great way to visualize distributions of data. Each bar represents input data placed into ranges, oftentimes called bins or buckets. 
## A Histogram displays the shape and spread of numerical data

## PLOTTING A HISTOGRAM PLOT

## DATASET: https://www.kaggle.com/datasets/shivamb/netflix-shows

df = pd.read_csv('netflix_titles.csv')

print(df)
print(df.columns)

## Visualize the distribution of the release_year
## To plot a Histogram, use the hist() function on either the PyPlot instance or an Axes instance.

df = pd.read_csv('netflix_titles.csv')
plt.hist(df['release_year'])

plt.show()

## CHANGING A HISTOGRAM'S BIN SIZE

df = pd.read_csv('netflix_titles.csv')
data = df['release_year']

plt.hist(data, bins = np.arange(min(data), max(data), 1))

plt.show()

## PLOTTING HISTOGRAM WITH DENSITY DATA

x_1 = np.random.randint(0, 10, 10)
x_2 = np.random.randint(0, 10, 100)

plt.hist(x_1, alpha = 0.5)
plt.hist(x_2, alpha = 0.5)

plt.show()

## Plt the data with a count of 100 to 10000

x_1 = np.random.randint(0, 10, 100)
x_2 = np.random.randint(0, 10, 10000)

plt.hist(x_1, alpha = 0.5)
plt.hist(x_2, alpha = 0.5)

plt.show()

## Density looks at the relative data instead of absolute data and even smaller datasets can be compared to larger ones. 
## Plot density by setting the density flag to True in the hist() function

x_1 = np.random.randint(0, 10, 10)
x_2 = np.random.randint(0, 10, 10000)

plt.hist(x_1, alpha = 0.5, density = True)
plt.hist(x_2, alpha = 0.5, density = True)

plt.show()

## Plot the distribution of release_year of the Netflix dataset

df = pd.read_csv('netflix_titles.csv')
data = df['release_year']
bins = np.arange(min(data), max(data), 1)

plt.hist(data, bins = bins, density = True)
plt.ylabel('Density')
plt.xlabel('Year')

plt.show()

## HISTOGRAM PLOT WITH KDE

## Kernerl Density Estimation (KDE) attempts to estimate the density function of a variable. 
## As Matplotlib does not contain a KDE line flag, Pandas does which uses Matplotlib as its foundation so we can visualize a plot with the KDE using pandas .plot method

df = pd.read_csv('netflix_titles.csv')

fig, ax = plt.subplots()

df['release_year'].plot.kde(ax = ax)

df['release_year'].plot.hist(ax = ax, bins = 25, density = True)

ax.set_ylabel('Density')
ax.set_xlabel('Year')

plt.show()

## Use the Axes instance to plot the histogram

df = pd.read_csv('netflix_titles.csv')

fig, ax = plt.subplots()

ax.hist(df['release_year'], bins = 25, density = True)
df['release_year'].plot.kde(ax = ax)

ax.set_ylabel('Density')
ax.set_xlabel('Year')

plt.show()

## CUSTOMIZING HISTOGRAM PLOTS

df = pd.read_csv('netflix_titles.csv')
data = df['release_year']
bins = np.arange(min(data), max(data), 1)

plt.hist(data, bins = bins,
         histtype = 'step', 
         alpha = 0.5, 
         align = 'right', 
         orientation = 'horizontal', 
         log = True)

plt.show()

## Several arguments defined:
## bins - Number of bins in the plot
## density - Whether PyPlot uses count or density to populate the plot
## histtype - The type of Histogram Plot (default is bar, through other values such as step or stepfilled are available)
## alpha - The alpha / transparency of the lines and bars
## align - To which side of the bins are the bars aligned, default is mid
## orientation - Horizontal / Vertical orientation, default is vertical
## log - Whether the plot should be put on a logarthmic scale or not

## BOX PLOT

## Box plots are used to visualize summary statistics of a dataset, displaying attributes of the distribution like the data's range and distribution
## Box Plots are used to visualize summaries of the data through quartiles
## A quartile is just a type of quantile, which represents one fourth of the data. A wellknown quantile is the percentile which represents on hundredth of the data,
## and a decile which represents one tenth of the data.

## Bars typically have whiskers - lines extending from the box that let us know what variance we can expect beyond Q3 and below Q1. 
## Outliers are detected using the IQR Method, namely - if they're located 1.5 IQR below Q1 or 1.5 IQR above Q3:
## IQR = Q3 - Q1
## Lower outliers < Q1 - 1.5*IQR
## Upper outliers < Q3 + 1.5*IQR
## 1.5 - The 68 - 95 - 99.7 rule postulates that any data that fits within a normal distribution will be within three standard deviations from the mean.
## 68% of the data will be present in the first standard deviation, 95% within the first and second and 99.7% within the first, second and third standard deviation of the mean.
## Safe to say that these 0.3% are most likely outliers. 
## When calculating the 'maximum' and 'minimum' these 0.3% are ignored

## 68 - 95 - 99.7 rule : https://en.wikipedia.org/wiki/68%e2%80%9395%e2%80%9399.7_rule

## DATASET - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

df = pd.read_csv('winequality-red.csv')
print(df)
print(df.isnull().values.any())

## PLOTTING A BOX PLOT
## To plot a Box Plot, call the boxplot() function and pass in the data

df = pd.read_csv('winequality-red.csv')

fixed_acidity = df['fixed acidity']
free_sulfur_dioxide = df['free sulfur dioxide']
total_sulfur_dioxide = df['total sulfur dioxide']
alcohol = df['alcohol']

fig, ax = plt.subplots()
ax.boxplot(fixed_acidity)
plt.show()

## Save the returned object of the plotting methods which contain various data on the plot to use the returned values to draw lines and add annotations 

df = pd.read_csv('winequality-red.csv')

fixed_acidity = df['fixed acidity']
free_sulfur_dioxide = df['free sulfur dioxide']
total_sulfur_dioxide = df['total sulfur dioxide']
alcohol = df['alcohol']

fig, ax = plt.subplots()
box = ax.boxplot(fixed_acidity)

median = box['medians'][0].get_ydata()[0]

upper_whisker = box['whiskers'][1].get_ydata()[1]
lower_whisker = box['whiskers'][0].get_ydata()[1]

new_min = box['boxes'][0].get_ydata()[1]
new_max = box['boxes'][0].get_ydata()[2]

min = fixed_acidity.min()
max = fixed_acidity.max()

ax.axhline(lower_whisker, 0, 1, label = 'Lower whisker, "minimum"', color = 'red')
ax.axhline(upper_whisker, 0, 1, label = 'Upper whisker, "Maximum"', color = 'blue')

ax.axhline(new_min, 0, 1, label = 'Q1, "minimum"', color = 'green')
ax.axhline(median, 0, 1, label = 'Q2, "median"', color = 'Yellow')
ax.axhline(new_max, 0, 1, label = 'Q3, "maximum"', color = 'orange')

ax.annotate('Actual maximum', 
            xy = (1, max),
            xytext = (0.6, max-0.5),
            arrowprops = dict(arrowstyle = '<->', connectionstyle = 'arc3, rad = -0.15'))

ax.annotate('Actual manimum', 
            xy = (1, min),
            xytext = (0.6, min-0.5),
            arrowprops = dict(arrowstyle = '<->', connectionstyle = 'arc3, rad = -0.15'))

ax.legend()
plt.show()

## Plot multiple features on one figure by providing a list. This can be done on either the plt instance, the fig object or the ax object

dataframe = pd.read_csv('winequality-red.csv')

fixed_acidity = dataframe['fixed acidity']
free_sulfur_dioxide = dataframe['free sulfur dioxide']
total_sulfur_dioxide = dataframe['total sulfur dioxide']
alcohol = dataframe['alcohol']

columns = [fixed_acidity, free_sulfur_dioxide, total_sulfur_dioxide, alcohol]

fig, ax = plt.subplots()
ax.boxplot(columns)
plt.show()

## Plot these in different Axes instances

dataframe = pd.read_csv('winequality-red.csv')

fig,ax = plt.subplots(nrows = 1, ncols = 4)

fixed_acidity = dataframe['fixed acidity']
free_sulfur_dioxide = dataframe['free sulfur dioxide']
total_sulfur_dioxide = dataframe['total sulfur dioxide']
alcohol = dataframe['alcohol']

ax[0].boxplot(fixed_acidity)
ax[0].set_title('Fixed Acidity')

ax[1].boxplot(free_sulfur_dioxide)
ax[1].set_title('Free Sulfur Dioxide')

ax[2].boxplot(total_sulfur_dioxide)
ax[2].set_title('Total Sulfur Dioxide')

ax[3].boxplot(alcohol)
ax[3].set_title('Alcohol')

plt.show()

## CUSTOMIZING BOX PLOTS

## We can customize the plot and add labels to the X-axis by using xticks() function. Pass in the number of labels we want to add and then the labels for each column.

fig, ax = plt.subplots()
ax.boxplot(columns)
plt.xticks([1, 2, 3, 4],
           ['Fixed acidity', 'Free sulfur dioxide', 'Total sulfur dioxide', 'Alcohol'], rotation = 10)
plt.show()

## Provide the vert argument 0 = False, 1 = True to control whether or not the plot is rendered vertically and it is set to 1 by default

fig, ax = plt.subplots()
ax.boxplot(fixed_acidity, vert = 0)
plt.show()

## notch=True attribute creates the notch format to the boxplot, path_artist=True fills the boxplot with colors

fig, ax = plt.subplots()
columns = [free_sulfur_dioxide, total_sulfur_dioxide]
ax.boxplot(columns, notch = True, patch_artist = True)
plt.xticks([1, 2], ['Free sulfur dioxide', 'Total sulfur dioxide'])
plt.show()

## The meanline argument can be used to render the mean on the box. This would conflict with notch, so should be avoided calling together.
## The showmeans parameter will be used as well and if possible, the mean will be visualized as a line that runs all the way across the box.
## If not possible, the mean will be shown as points:

fig, ax = plt.subplots()
columns = [free_sulfur_dioxide, total_sulfur_dioxide]
ax.boxplot(columns, patch_artist = True, meanline = True, showmeans = True)
plt.xticks([1, 2], ['Free sulfur dioxide', 'Total sulfur dioxide'])
plt.show()

## Color the Box Plots with the set_facecolor() function by using the zip() function containing the 'boxes' element of the box object with the colors and then set face color

columns = [fixed_acidity, free_sulfur_dioxide, total_sulfur_dioxide, alcohol]
fig, ax = plt.subplots()
box = ax.boxplot(columns, notch = True, patch_artist = True)
plt.xticks([1, 2, 3, 4], 
           ['Fixed acidity', 'Free sulfur dioxide', 'Total sulfur dioxide', 'Alcohol'], 
           rotation = 10)

colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    
plt.show()

## EXPLORING RELATIONSHIPS WITH SCATTER PLOTS, HISTOGRAMS AND BOX PLOTS

df = pd.read_csv('winequality-red.csv')

fixed_acidity = df['fixed acidity']
free_sulfur_dioxide = df['free sulfur dioxide']
total_sulfur_dioxide = df['total sulfur dioxide']
alcohol = df['alcohol']
quality = df['quality']

fig, ax = plt.subplots(nrows = 4)

fig.supxlabel('Quality')

ax[0].scatter(x = quality, y = fixed_acidity)
ax[0].set_ylabel('Fixed Acidity')

ax[1].scatter(x = quality, y = free_sulfur_dioxide)
ax[1].set_ylabel('Free Sulfer Dioxide')

ax[2].scatter(x = quality, y = total_sulfur_dioxide)
ax[2].set_ylabel('Total Sulfur Dioxide')

ax[3].scatter(x = quality, y = alcohol)
ax[3].set_ylabel('Alcohol')

plt.show()

## Plot the distribution of the alcohol content to the quality in a histogram

df = pd.read_csv('winequality-red.csv')

alcohol = df['alcohol']
fig, ax = plt.subplots()
ax.hist(alcohol)

plt.show()

## Plot mulitple types of plot types on one plot -- Not the ideal way to approach this task. A joint task would offer a better advantage to optimally gauge the correlations

df = pd.read_csv('winequality-red.csv')

alcohol = df['alcohol']
quality = df['quality']

fig, ax = plt.subplots(3, 2)

ax[0][0].hist(alcohol)
ax[1][0].boxplot(alcohol, vert = 0)
ax[2][0].scatter(x = alcohol, y = quality)
ax[2][0].set_xlabel('Alcohol')

ax[1][1].boxplot(quality)
ax[2][1].hist(quality, orientation = 'horizontal')
ax[2][1].set_xlabel('Quality')

plt.show()

## SCATTER PLOT WITH MARGINAL DISTRIBUTIONS (JOINT PLOT)

## Joint Plots are used to explore relationships between bivariate data, as well as their distributions at the same time
## Joint Plots are just Scatter Plots with accompanying Distribution Plots (Histograms, Box Plots, Violin Plots) on both axes of the plot, to explore the distribution
## of the variables that constitute the Scatter Plot itself. The JointPlot name was actually coined by the Seaborn team. Following popularity other libraries such as
## Bokeh and Pandas introduced them

## Joint Plot tasks are much more suited for libraries such as Seaborn, which has a built-in joinplot() function. With Matplotlib, we'll construct a JointPlot manually,
## using GridSpec and multiple Axes objects. GridSpec is used for advanced customization of the subplot grid on a Figure.

## DATASET : https://www.kaggle.com/datasets/uciml/iris

df = pd.read_csv('Iris.csv')
print(df.head())

## PLOTTING A JOINT PLOT WITH SINGLE-CLASS HISTOGRAMS

## Create Figure and Axes objects

df = pd.read_csv('Iris.csv')

fig = plt.figure()
gs = GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

plt.show()

## Update the script to plot the SepalLengthCm and SepalWidthCm features through a Scatter Plot, on the ax_+scatter axes, and the distributions of each on 
## ax_hist_y and ax_hist_x axes

df = pd.read_csv('Iris.csv')

fig = plt.figure()
gs = GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(df['SepalLengthCm'], df['SepalWidthCm'])

ax_hist_x.hist(df['SepalLengthCm'])
ax_hist_y.hist(df['SepalWidthCm'], orientation= 'horizontal')

plt.show()

## PLOTTING A JOIN PLOT WITH MULTIPLE-CLASS HISTOGRAMS

## Dissect the DataFrame by the flower Species

df = pd.read_csv('Iris.csv')

setosa = df[df['Species'] == 'Iris-setosa']
virginica = df[df['Species'] == 'Iris-virginica']
versicolor = df[df['Species'] == 'Iris-versicolor']
species = df['Species']
colors = {
        'Iris-setosa': 'blue',
        'Iris-virginica': 'red',
        'Iris-versicolor': 'green'
          }

fig = plt.figure()
gs = GridSpec(4,4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c = species.map(colors))

ax_hist_y.hist(versicolor['SepalLengthCm'], color = 'red', alpha = 0.4)
ax_hist_y.hist(virginica['SepalLengthCm'], color = 'green', alpha = 0.4)
ax_hist_y.hist(setosa['SepalLengthCm'], color = 'blue', alpha = 0.4)

ax_hist_x.hist(versicolor['SepalWidthCm'],
               orientation= 'horizontal', color = 'red', alpha = 0.4)

ax_hist_x.hist(virginica['SepalWidthCm'],
               orientation = 'horizontal', color = 'green', alpha = 0.4)

ax_hist_x.hist(setosa['SepalWidthCm'],
               orientation = 'horizontal', color = 'blue', alpha = 0.4)

plt.show()

## The fill can be taken away to eliminate the overlapping of colors in the histogram by using the step argument

df = pd.read_csv('Iris.csv')

setosa = df[df['Species'] == 'Iris-setosa']
virginica = df[df['Species'] == 'Iris-virginica']
versicolor = df[df['Species'] == 'Iris-versicolor']
species = df['Species']
colors = {
        'Iris-setosa': 'blue',
        'Iris-virginica': 'red',
        'Iris-versicolor': 'green'
          }

fig = plt.figure()
gs = GridSpec(4,4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c = species.map(colors))

ax_hist_y.hist(versicolor['SepalLengthCm'], histtype = 'step', color = 'red', alpha = 0.4)
ax_hist_y.hist(virginica['SepalLengthCm'], histtype = 'step', color = 'green', alpha = 0.4)
ax_hist_y.hist(setosa['SepalLengthCm'], histtype = 'step', color = 'blue', alpha = 0.4)

ax_hist_x.hist(versicolor['SepalWidthCm'],
               orientation= 'horizontal', histtype = 'step', color = 'red', alpha = 0.4)

ax_hist_x.hist(virginica['SepalWidthCm'],
               orientation = 'horizontal', histtype = 'step', color = 'green', alpha = 0.4)

ax_hist_x.hist(setosa['SepalWidthCm'],
               orientation = 'horizontal', histtype = 'step', color = 'blue', alpha = 0.4)

plt.show()

## JOINT PLOTS WITH BOX PLOTS

## Rewrite Red Wine Quality pack of plot types using GridSpec

df = pd.read_csv('winequality-red.csv')

alcohol = df['alcohol']
quality = df['quality']

fig = plt.figure()
gs = GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_box_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(x = alcohol, y = quality)
ax_box_y.boxplot(quality)
ax_hist_x.hist(alcohol)

fig.supxlabel('Alcohol')
fig.supylabel('Quality')

plt.show()

## VIOLIN PLOTS

## Violin plots show the same summary staistics as box plots, but they also include Kernel Density Estimations that represent the shape/distribution of the data

## DATASET : https://www.kaggle.com/datasets/tklimonova/gapminder-datacamp-2007

dataframe = pd.read_csv('gapminder_full.csv', error_bad_lines = False ) # error bad lines depricated
print(dataframe)
print(dataframe.isnull().values.any())

## Plot a Violin Plot by calling the violinplot() function on either the Axes instance, or the PyPlot instance itself.

dataframe = pd.read_csv('gapminder_full.csv', error_bad_lines = False)

population = dataframe.population
life_exp = dataframe.life_exp
gdp_cap = dataframe.gdp_cap

fig, ax = plt.subplots()

ax.violinplot([population, life_exp, gdp_cap])

ax.set_title('Violin Plot')
plt.show()

## Plot on different axes and filter the data to make the dataset easier to compare 

dataframe = pd.read_csv('gapminder_full.csv', error_bad_lines = False)

dataframe = dataframe.sort_values(by = ['population'], ascending = False)
plt.bar(dataframe.country, dataframe.population)
plt.xticks(rotation = 90)
plt.show()

## Drop the top several countries with a significantly higher population count to normailze the data. 

dataframe = dataframe.groupby('country').last()
dataframe = dataframe.sort_values(by = ['population'], ascending = False)
dataframe = dataframe.iloc[10:]
print(dataframe)

## Plot newly compiled dataframe with Violin Plots with median lines

population = dataframe.population
life_exp = dataframe.life_exp
gdp_cap = dataframe.gdp_cap

fig, (ax1, ax2, ax3) = plt.subplots(nrows =1, ncols = 3)

ax1.violinplot(dataframe.population, showmedians = True)
ax1.set_title('Population')

ax2.violinplot(life_exp, showmedians = True)
ax2.set_title('Life Expectancy')

ax3.violinplot(gdp_cap, showmedians = True)
ax3.set_title('GDP Per Cap')

plt.show()

## Plot a Box Plot next to the Violin Plot and annotate the constituent elements

dataframe = pd.read_csv('gapminder_full.csv', error_bad_lines = False)

dataframe = dataframe.groupby('country').last()
dataframe = dataframe.sort_values(by = ['population'], ascending = False)
dataframe = dataframe.iloc[10:]

population = dataframe.population

fig, ax = plt.subplots(nrows = 1, ncols = 2)

box = ax[0].boxplot(population)
violin = ax[1].violinplot(population, showmedians = True)

median = box['medians'][0].get_ydata()[0]

upper_whisker = box['whiskers'][1].get_ydata()[1]
lower_whisker = box['whiskers'][0].get_ydata()[1]

new_min = box['boxes'][0].get_ydata()[1]
new_max = box['boxes'][0].get_ydata()[2]

min = population.min()
max = population.max()

ax[0].axhline(lower_whisker, 0, 1, label = 'Lower whisker, "minimum"', color = 'red')
ax[0].axhline(upper_whisker, 0, 1, label = 'Upper whisker, "minimum"', color = 'blue')

ax[0].axhline(new_min, 0, 1, label = 'Q1, "minimum"', color = 'green')
ax[0].axhline(median, 0, 1, label = 'Q2, "median"', color = 'yellow')
ax[0].axhline(new_max, 0, 1, label = 'Q3, "maximum"', color = 'orange')

ax[1].axhline(lower_whisker, 0, 1, label = 'Lower whisker, "minimum"', color = 'red')
ax[1].axhline(upper_whisker, 0, 1, label = 'Upper whisker, "minimum"', color = 'blue')

ax[1].axhline(new_min, 0, 1, label = 'Q1, "minimum"', color = 'green')
ax[1].axhline(median, 0, 1, label = 'Q2, "median"', color = 'yellow')
ax[1].axhline(new_max, 0, 1, label = 'Q3, "maximum"', color = 'orange')

ax[0].annotate('Acutal maximum', 
               xy = (1, max),
               xytext = (0.6, max-0.5),
               arrowprops = dict(arrowstyle = '<->', connectionstyle = 'arc3, rad = -0.15'))
ax[0].annotate('Acutal minimum', 
               xy = (1, min),
               xytext = (0.6, min-0.5),
               arrowprops = dict(arrowstyle = '<->', connectionstyle = 'arc3, rad = -0.15'))

plt.show()

## KDE Line, vertically, using Pandas plotting module cannot be done. Plotting horizontally can be done. 

dataframe = pd.read_csv('gapminder_full.csv', error_bad_lines = False)

dataframe = dataframe.groupby('country').last()
dataframe = dataframe.sort_values(by = ['population'], ascending = False)
dataframe = dataframe.iloc[10:]

population = dataframe.population
life_exp = dataframe.life_exp
gdp_cap = dataframe.gdp_cap

fig, ax = plt.subplots(nrows = 3, ncols = 1)

population.hist(ax = ax[0], density = True, alpha = 0.1)
population.plot.kde(ax = ax[0])

ax[1].boxplot(population, vert = False)
ax[2].violinplot(population, showmedians = True, vert = False)

for axes in ax:
    axes.set_xlim(0, population.max())
    
plt.show()

## CUSTOMIZING VIOLIN PLOTS

## ADDING X AND Y TICKS

fig, ax = plt.subplots()
ax.violinplot(gdp_cap, showmedians = True)
ax.set_title('Violin Plot')
ax.set_xticks([1])
ax.set_xticklabels(['Country GDP'])
plt.show()

## SHOWING DATASET MEANS AND MEDIANS IN VIOLIN PLOTS

## By default, unline Box Plots - Violin Plots don't show a median line. This and means can be turned on by passing the showmeans and showmedians parameters

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3)
ax1.violinplot(population, showmedians = True, showmeans = True, vert = False)
ax1.set_title('Population')

ax2.violinplot(life_exp, showmedians = True, showmeans = True, vert = False)
ax2.set_title('Life Expectancy')

ax3.violinplot(gdp_cap, showmedians = True, showmeans = True, vert = False)
ax3.set_title('GDP Per Cap')
plt.show()

## CUSTOMIZING KERNEL DENSITY ESTIMATION FOR VIOLIN PLOTS

## The points parameter can alter how many data points the model considers when creaing the KDE - 100 points by default

fig, ax = plt.subplots()
ax.violinplot(gdp_cap, showmedians = True, points = 10)
ax.set_title('Violin Plot')
ax.set_xticks([1])
ax.set_xticklabels(['Country GDP'])
plt.show()

## Plot a 10 point, 100 point and 500 point Violin Plot

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3)
ax1.violinplot(gdp_cap, showmedians = True, points = 10)
ax1.set_title('GDP Per Cap, 10p')

ax2.violinplot(gdp_cap, showmedians = True, points = 100)
ax2.set_title('GDP Per Cap, 100p')

ax3.violinplot(gdp_cap, showmedians = True, points = 500)
ax3.set_title('GDP Per Cap, 500p')
plt.show()

## SCATTER PLOTS WITH VIOLIN PLOTS

## Tweak Joint Plot example to use a Violin Plot on both margins instead

df = pd.read_csv('winequality-red.csv')

alcohol = df['alcohol']
quality = df['quality']

fig = plt.figure()
gs = GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_violin_y = fig.add_subplot(gs[0, 0:3])
ax_violin_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(x = alcohol, y = quality)
ax_violin_x.violinplot(quality)
ax_violin_y.violinplot(alcohol, vert = 0)

fig.supxlabel('Alcohol')
fig.supylabel('Quality')

plt.show()

## STACK PLOTS

## Stack Plots are used to plot linear data, in a vertical order, stacking each linear plot on another. 

## DATASET : https://ourworldindata.org/grapher/cumulative-covid-vaccinations

dataframe = pd.read_csv('cumulative-covid-vaccinations.csv')
print(dataframe)

## Plot a Stack Plot

x = [1, 2, 3, 4, 5]
y1 = [5, 6, 4, 5, 7]
y2 = [1, 6, 4, 5, 6]
y3 = [1, 1, 2, 3, 2]

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3)
plt.show()

## Use a dictionary to avoid utilizing multiple lists

x = [1, 2, 3, 4, 5]

y_values = {
    'y1' : [5, 6, 4, 5, 7],
    'y2' : [1, 6, 4, 5, 6],
    'y3' : [1, 1, 2, 3, 2]
}

fig, ax = plt.subplots()
ax.stackplot(x, y_values.values())
plt.show()

## Add labels and a legend to the plot.

x = [1, 2, 3, 4, 5]

y_values = {
    'y1' : [5, 6, 4, 5, 7],
    'y2' : [1, 6, 4, 5, 6],
    'y3' : [1, 1, 2, 3, 2]
}

fig, ax = plt.subplots()
ax.stackplot(x, y_values.values(), labels = y_values.keys())
ax.legend(loc = 'upper left')
plt.show()

## Cumulative Covid Vaccinations Stack Plot
## Process the data

dataframe = pd.read_csv('cumulative-covid-vaccinations.csv')
indices = dataframe[(dataframe['Entity'] == 'World')
                    | (dataframe['Entity'] == 'European Union')
                    | (dataframe['Entity'] == 'High income')].index
dataframe.drop(indices, inplace = True)

countries_vaccinations = dataframe.groupby('Entity')['total_vaccinations'].apply(list)
print(countries_vaccinations)

## Convert the series into a dictionary

cv_dict = countries_vaccinations.to_dict()
df = pd.Series(cv_dict)
print(cv_dict)

## Create a new dictionary and fill with 0's for the days missing values by finding the key with the most values and inserting 0's for missing values

### max_key, max_value = max(cv_dict.items(), key = lambda x: len(set(x[1])))

max_key = 'Canada'
max_value = 90

cv_dict_full = {}
for k, v in cv_dict.items():
    if len(v) < len(max_value):
        trailing_zeros = [0]*(len(max_value)-len(v))
        cv_dict_full[k] = trailing_zeros + v
    else:
        cv_dict_full[k] = v
        
print(cv_dict_full)

print(max_key, len(max_value))

dates = np.arange(0, len(max_value))

fig, ax = plt.subplots()
ax.stackplot(dates, cv_dict_full.values(), labels = cv_dict_full.keys())
ax.legend(loc = 'upper left', ncol = 4)
ax.set_title('Cumulative Covid Vaccinations')
ax.set_xlabel('Day')
ax.set_ylabel('Number of People')

plt.show()

## HEATMAPS

## Heatmpaps color-code variables, based on the value of other variables
## The color-coding of variables is usually ascribed to an intensity variable, which changes the color of the cell it belongs to
## colormaps of Heatmaps are typically from cold colors like blue to warm colors like red, giving the intuitive feel of the intensity of the variable

## Correlation Heatmaps are commonly used in the Exploratory Data Analysis (which includes Data Visualization) step of Data Science

df = pd.read_csv('AmesHousing.csv')

print(df)
print(df.columns)

## PLOTTING A HEATMAP

## corr() function is a way to calculate the correlation of features in a DataFrame using Pandas

saleprice_corr = df.corr()[['SalePrice']]
print(saleprice_corr)

## The corr() function calculates the correlation between all numerical features and target feature selected. It ignores all null-fields and calculates
## the Pearson Correlation Coefficient
## The Pearson Correlation Coefficient measures the linear association between two variables - and depending on the value returned, from -1 to 1 it can be interpreted as:

## +1       Complete Positive Correlation
## +0.8     Strong Positive Correlation
## +0.6     Moderately Positive Correlation
## 0        No Correlation Whatsoever
## -0.6     Moderately Negative Correlation
## -0.8     Strong Negative Correlation
## -1       Complete Negative Correlation

## Other than the default pearson value, we can specify the method as kendall or spearman - signifying the Kendall Rank Correlation Coefficient and Spearman's Rank Correlation
## Coefficient respectively

## Pearson calculates the correlation as a linear relationship between two vriables, Spearman valculates the monotonic relation between a pair of variables, 
## while Kendall calculates the ordinal association of two variables
## The corr() function returns a DataFrame

df = pd.read_csv('AmesHousing.csv')

pearson_corr = df.corr(method = 'pearson')
print(pearson_corr)

## Limit the correlation calculation to a single variable against all other variables as well, or any subset

pearson_corr = df.corr(method = 'pearson')[['SalePrice']]

## Limit the correlation calculation to a select set of variables.

pearson_corr = df.corr(method = 'pearson')[['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area']]

## As long as we have a 2D array, we can plot a Heatmap. The Heatmaps are plotted via the imshow() function which is used for plotting images.
## The scalar image is rendered as a pseudocolor image. The plot can be tweaked by forcing aspect to be 'equal', the interpolation to 'nearest' and origin to 'upper'
## to achieve the look of a Heatmap

## matshow(), short for Matrix Show, is a wrapper for the imshow() function, and with these areguments set to the right values it would produce a Heatmap/Correlation Matrix

df = pd.read_csv('AmesHousing.csv')
pearson_corr = df.corr(method = 'pearson')[['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area']]

fig, ax = plt.subplots()
ax.matshow(pearson_corr)

plt.yticks(range(0, len(pearson_corr.index)), pearson_corr.index, fontsize = 6)

plt.xticks(range(0, len(pearson_corr.columns)), pearson_corr.columns, fontsize = 6, rotation = 90)

plt.show()

 ## Sort the resulting Dataframe, change the aspect and add numerical values
 
df = pd.read_csv('AmesHousing.csv')
 
pearson_corr = df.corr(method = 'pearson')[['SalePrice']].sort_values(by = 'SalePrice', ascending = False)
 
fig, ax = plt.subplots()
 
ax.matshow(pearson_corr, aspect = 0.05)
 
plt.yticks(range(0, len(pearson_corr.index)), pearson_corr.index, fontsize = 6)
plt.xticks(range(0, len(pearson_corr.columns)), pearson_corr.columns, fontsize = 6, rotation = 90)
 
for y in range(pearson_corr.shape[0]):
    plt.text(0, y, '%.1f' % pearson_corr.iloc[y],
            horizontalalignment = 'center', 
            verticalalignment = 'center',
            fontsize = 6)

plt.show()

## Widen the scope of the correlation to all features

df = pd.read_csv('AmesHousing.csv')

pearson_corr = df.corr(method = 'pearson')

fig, ax = plt.subplots()
ax.matshow(pearson_corr)

plt.yticks(range(0, len(pearson_corr.index)),
           pearson_corr.index, fontsize = 6)
plt.xticks(range(0, len(pearson_corr.columns)),
           pearson_corr.columns, fontsize = 6, rotation = 90)

for y in range(pearson_corr.shape[0]):
    for x in range(pearson_corr.shape[1]):
        plt.text(x, y, '%.1f' % pearson_corr.iloc[x, y],
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 fontsize = 6)
        
plt.show()

## RIDGE PLOTS (JOY PLOTS)

## Ridge Plots, also known as Joy Plots, are vertical multi-Axes plot type that combines several, usually distribution plots
## The Axes on Ridge Plots overlap, and form something that looks like a 3D image of ridges, even though they're really 2D. 
## A 3D variation of Ridge Plots does exist.

## PLOTTING A RIDGE PLOT

df = pd.read_csv('clean.csv')
print(df)

groups = df.groupby(['line'])

print(groups.get_group(1))

## Plot a Line Plot for each group. The groups is of type DataFrameGroupBy, and each group in it is a tuple of a Group Number (int) and Group Content (DataFrame).
## Reference the DataFrame for that group when looping through the groups

df = pd.read_csv('clean.csv')
groups = df.groupby(['line'])
num_of_groups = len(groups.groups)

fig, ax = plt.subplots(num_of_groups, 1, figsize = (6, 8))

i = 0
for group in groups:
    ax[i].plot(group[1]['x'], group[1]['y'])
    i += 1

plt.show()

## Add style to the plot

df = pd.read_csv('clean.csv')
groups = df.groupby(['line'])
num_of_groups = len(groups.groups)

fig, ax = plt.subplots(num_of_groups, 1, figsize = (6, 8), facecolor = 'black')
fig.subplots_adjust(hspace = 0.7, left = 0.25, bottom = 0.25, right = 0.75, top = 0.75)

i = 0
for group in groups:
    ax[i].plot(group[1]['x'], group[1]['y'])
    i += 1

plt.show()

## Closer to the original wave

df = pd.read_csv('clean.csv')
groups = df.groupby(['line'])
num_of_groups = len(groups.groups)

fig, ax = plt.subplots(num_of_groups, 1, figsize = (6, 8), facecolor = 'black')
fig.subplots_adjust(hspace = 0.7, left = 0.4, bottom = 0.4, right = 0.6, top = 0.6)

i = 0
for group in groups:
    ax[i].plot(group[1]['x'], group[1]['y'], color = 'white', linewidth = 0.7)
    ax[i].axis('off')
    i += 1

plt.show()

## SPECTROGRAM PLOTS

## Spectograms are visual representations of the spectrum of frequencies of a signal in time-series
## Commonly applied to audio signals, and are sometimes known as sonographs or voicegrams
## Hidden Hierarchical Markov Models
## https://en.wikipedia.org/wiki/Hierarchical_hidden_Markov_model

## Spectrograms also have a use in decoding steganography - the art of hiding messages and data within ther data or physical obsjects. 
## Plot a Spectrogram in Matplotlib through the specgram() function of the Axes or plt instances. It accepts a 1D array of data. 
## NFFT - the number of data points used in each block for the Fast Fourier Transform (FFT)
## Fs - is the sampling frequency, used to calculate Fourier frequencies
## Window - is the window function used in the FFT
## Scale - is the Y-Scale onm which we're plotting, and can be linear, dB(10*log10) or default. The default depends on the mode parameter.
## Mode - is the type of spectrum to use. The default is psd (power-spectral density), though other spectrums such as magnitude (magnitude spectrum),
## angle (phase spectrum without unwrapping) and phase (phase spectrum with unwrapping) exist

## MUSIC PIECE FOR DATASET : https://github.com/StackAbuse/venetian-snares-look-data/blob/main/venetian-snares-look.txt

df = pd.read_csv('venetian-snares-look.txt', sep = ' ', header = None)

data1, data2 = np.hsplit(df, 2)

fig, ax = plt.subplots(3, 1)

ax[0].plot(df)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Intensity')

ax[1].specgram(data1)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Frequency')

ax[2].specgram(data2, NFFT = 1024, Fs = rate, noverlap = 512, scale = 'default')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Frequency')
ax[2].set_yscale('symlog')
ax[2].set_ylim(100, 10000)

plt.show()

## MATPLOTLIB - 3D SUPPORT

## Use the Axes3D object instead of the Axes object when plotting 3D plots.
## Pass the projection = '3d' as an arguemnt into the add_subplot() function - which changes the returned Axes instance into an Axes3D

## Note that the subplots() function foesn't accept the projection argument. Create a Figure instance via the figure() call and then add an Axes3D to it via the add_subplot()
## function and the projection argument. This is done because we can combine multiple Axes3D and Axes instances on a single Figure - the Figure itself is projection-agnostic

## Import Axes3D from the mpl_toolkits.mplot3d module. 

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

plt.show()

## The Axes3D is interactive and can be panned and rotated in all three dimensions - the spines will adapt to the perspective you're looking from automatically.
## You can rotate two different Axes3D instances independent of eachother 

fig = plt.figure()
ax = fig.add_subplot(121, projection = '3d')
ax2 = fig.add_subplot(122, projection = '3d')

plt.show()

## 3D SCATTER PLOTS AND BUBBLE PLOTS

df = pd.read_csv('AmesHousing.csv')

fig = plt.figure()
ax = fig.add_subplot(131, projection = '3d')
ax2 = fig.add_subplot(132, projection = '3d')
ax3 = fig.add_subplot(133, projection = '3d')

sale_price = df['SalePrice']
gr_liv_area = df['Gr Liv Area']
overall_qual = df['Overall Qual']
lot_area = df['Lot Area']
total_bsmt_sf = df['Total Bsmt SF']
year_built = df['Year Built']

ax.scatter(sale_price, gr_liv_area, overall_qual)
ax.set_xlabel('Sale Price')
ax.set_ylabel('Living Area Above Ground Level')
ax.set_zlabel('Overall Quality')

ax2.scatter(year_built, lot_area, overall_qual)
ax2.set_xlabel('Year Built')
ax2.set_ylabel('Lot Area')
ax2.set_zlabel('Overall Quality')

ax3.scatter(year_built, lot_area, total_bsmt_sf)
ax3.set_xlabel('Year Built')
ax3.set_ylabel('Lot Area')
ax3.set_zlabel('Total Basement Area')

plt.show()

## Bubble Plot with World Happiness dataset

df = pd.read_csv('C:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

fig = plt.figure()
ax3d = fig.add_subplot(111, projection = '3d')
gdp = df['GDP per capita']
generosity = df['Generosity']
social_support = df['Social support']

size = df['Score'].to_numpy()
s = [3*s**2 for s in size]

ax3d.scatter(gdp, generosity, social_support, s = s, alpha = 0.5)
ax3d.set_xlabel('GDP Per Capital')
ax3d.set_ylabel('Generosity')
ax3d.set_zlabel('Social Support')

plt.show()

## COMBINING 2D AND 3D PLOTS
## Figures are projection-agnostic which means we can have multiple Axes3D and Axes instances on a single Figure

df = pd.read_csv('C:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

fig = plt.figure()

gdp = df['GDP per capita']
generosity = df['Generosity']
social_support = df['Social support']
correlations = df.corr()

size = df['Score'].to_numpy()
s = [3*s**2 for s in size]

ax3d = fig.add_subplot(121, projection = '3d')
ax3d.scatter(gdp, generosity, social_support, s = s, alpha = 0.5)
ax3d.set_xlabel('GDP Per Capital')
ax3d.set_ylabel('Generosity')
ax3d.set_zlabel('Social Support')

ax = fig.add_subplot(122)
ax.matshow(correlations)

plt.yticks(range(0, len(correlations.index)),
           correlations.index, fontsize = 8)
plt.xticks(range(0, len(correlations.columns)),
           correlations.columns, fontsize = 8, rotation = 90)

for y in range(correlations.shape[0]):
    for x in range(correlations.shape[1]):
        plt.text(x, y, '%.1f' % correlations.iloc[x, y],
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 fontsize = 8
                 )

plt.tight_layout()

plt.show()

## 3D SURFACE PLOTS AND WIREFRAME PLOTS

## Imagine the heatmap as a pliable surface, like a table cloth that takes shape over a clay tablet with certain terrain beneath it. 
## Bend and reshape a Heatmap based on another feature; this plot is called a Surface Plot
## Plot a Surface Plot in Matplotlib by calling the plot_surface() function and provide 3 features (x, y, z) and a color map for them. 
## A variant of the Surface Plot is a Wareframe plot which conveys the exact same data, b ut consists of a wire-grid without filled/colored rectangles
## Wireframe Plot can be called with the plot_wireframe() function providing the 3 features. 
## plot_surface() requires 2D arrays. Utilize a meshgrid that is created from two 1D arrays, and is itself a 2D array - a grid made from the two arrays. 
## Create a meshgrid with the meshgrid() function

fig, ax =  plt.subplots()

nums = np.arange(0, 10, 1)
dummy = [0]*10

x = nums
y = nums
ax.scatter(x, dummy)
ax.scatter(dummy, y)

plt.show()

## Use meshgrid() to visualize a new plot 

nums = np.arange(0, 10, 1)

x = nums
y = nums

print('x before meshgrid: ','\n', x)
print('y before meshgrid: ','\n', y)

x, y = np.meshgrid(x, y)

print('x after meshgrid: ','\n', x)
print('y after meshgrid: ','\n', y)

## Plot x and y

fig, ax = plt.subplots()
nums = np.arange(0, 10, 1)
x = nums
y = nums

x, y = np.meshgrid(x, y)

ax.scatter(x, y)

plt.show()

## Apply logic to World Happiness Dataset

df = pd.read_csv('c:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')
fig = plt.figure()

correlations = df.corr()

x = y = nums = np.arange(0, len(correlations.columns), 1)
x, y = np.meshgrid(x, y)
z = correlations.values

ax3d = fig.add_subplot(111, projection = '3d')
ax3d.plot_surface(x, y, z, cmap = cm.coolwarm)

plt.yticks(range(0, len(correlations.index)), correlations.index, fontsize = 8)
plt.xticks(range(0, len(correlations.columns)), correlations.columns, fontsize = 8)

plt.show()

## Wireframe plots are a meshgrid where each marker ('vertex') is joined with 4 other vertices, but the surfaces between them arent filled. 
## Plot a wireframe plot

df = pd.read_csv('c:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')
fig = plt.figure()

correlations = df.corr()

x = y = nums = np.arange(0, len(correlations.columns), 1)
x, y = np.meshgrid(x, y)
z = correlations.values

ax3d = fig.add_subplot(111, projection = '3d')
ax3d.plot_wireframe(x, y, z, cmap = cm.coolwarm)

plt.yticks(range(0, len(correlations.index)), correlations.index, fontsize = 8)
plt.xticks(range(0, len(correlations.columns)), correlations.columns, fontsize = 8)

plt.show()

## PROJECTING SURFACE PLOTS WITH CONTOUR PLOTS
## Contour Plots are a type of 2D plot, used for visualizing 3D data. 
## Contour Plots consist of contours (Z-axis slices). They allow us to plot the relationship between 3 features on a 2D plot

## PLOTTING A CONTOUR PLOT

## Plot a Contour Plot by calling the contour() function on an Axes or plt instance, passing in the 3 features we would like to plot

df = pd.read_csv('AmesHousing.csv')
fig, ax = plt.subplots()
correlations = df.corr()

x = y = range(0, len(correlations.columns))

x, y = np.meshgrid(x, y)
z = correlations.values

ax.contour(x, y, z)

plt.yticks(range(0, len(correlations.index)),
           correlations.index, fontsize = 8)
plt.xticks(range(0, len(correlations.columns)),
           correlations.columns, fontsize = 8, rotation = 90)

plt.show()

## The contour() function has a similar, overloaded partner - contourf() filling in the spaces between the contours. 

df = pd.read_csv('AmesHousing.csv')
fig, ax = plt.subplots()
correlations = df.corr()

x = y = range(0, len(correlations.columns))

x, y = np.meshgrid(x, y)
z = correlations.values

ax.contourf(x, y, z)

plt.yticks(range(0, len(correlations.index)),
           correlations.index, fontsize = 8)
plt.xticks(range(0, len(correlations.columns)),
           correlations.columns, fontsize = 8, rotation = 90)

plt.show()

## Adjust the colormap

df = pd.read_csv('AmesHousing.csv')
fig, ax = plt.subplots()
correlations = df.corr()

x = y = range(0, len(correlations.columns))

x, y = np.meshgrid(x, y)
z = correlations.values

ax.contourf(x, y, z, cmap = cm.coolwarm)

plt.yticks(range(0, len(correlations.index)),
           correlations.index, fontsize = 8)
plt.xticks(range(0, len(correlations.columns)),
           correlations.columns, fontsize = 8, rotation = 90)

plt.show()

## PLOTTING A SURFACE PLOT WITH CONTOUR PLOTS

## Plot a Surface Plot with a Contour Plot

df = pd.read_csv('AmesHousing.csv')
fig, ax = plt.subplots()
correlations = df.corr()

x = y = range(0, len(correlations.columns))

x, y = np.meshgrid(x, y)
z = correlations.values

ax3d = fig.add_subplot(111, projection = '3d')

ax3d.plot_surface(x, y, z, cmap = cm.coolwarm)

ax3d.contourf(x, y, z, zdir = 'z', offset = -1, cmap = cm.coolwarm)


plt.yticks(range(0, len(correlations.index)),
           correlations.index, fontsize = 8, rotation = 90)
plt.xticks(range(0, len(correlations.columns)),
           correlations.columns, fontsize = 8, rotation = 90)

plt.show()

## 3D LINE PLOTS AND CP1919 RIDGE PLOT

df = pd.read_csv(r"https://raw.githubusercontent.com/StackABuse/CP1919/master/data-raw/clean.csv")
groups = df.groupby(['line'])

plt.style.use('dark_background')

fig = plt.figure(figsize = (6, 8))

ax = fig.add_subplot(111, projection = '3d')
## ax.axis('off')

for group in groups:
    ax.plot(group[1]['line'], group[1]['x'], group[1]['y'], color = 'white')
    
plt.show()

## EXPLORING EEG (BRAINWAVE) CHANNEL DATA WITH LINE PLOTS, SURFACE PLOTS AND SPECTROGRAMS

## DATASET : https://archive.ics.uci.edu/dataset/121/eeg+database : https://archive.ics.uci.edu/dataset/121/eeg+database

filenames_list = os.listdir('EEG-Alcoholics/SMNI_CMI_TRAIN/')

df = pd.DataFrame({})

i = 0
for filename in filenames_list:
    if not filename == 'Train':
        temp_df = pd.read_csv('EEG-Alcoholics/SMNI_CMI_TRAIN/' + filename)
        df = df.append(temp_df)
        i += 1
        print(f'Appended .csv file number {i}/{len(filenames_list)}')
        
df = df.drop(['Unnamed: 0'], axis = 1)

alcoholic_df = df[df['subject identifier'] == 'a']
control_df = df[df['subject identifier'] == 'c']

print(alcoholic_df.columns)
print(control_df)

## Create a DataFrame that is pivoted from long format to wide format

alcholic_df = pd.pivot_table(
    alcoholic_df[['sensor position', 'sample num', 'sensor value']],
    index = 'sensor position', columns = 'sample num', values = 'sensor value'
)

control_df = pd.pivot_table(
    control_df[['sensor position', 'sample num', 'sensor value']],
    index = 'sensor position', columns = 'sample num', values = 'sensor value'
)

print(alcoholic_df)
print(control_df)

print('Alcoholic group min and max: \n', np.min(alcoholic_df.values), np.max(alcoholic_df.values))
print('Control group min and max: \n', np.min(control_df.values), np.max(control_df.values))

## PLOTTING SURFACE PLOTS OF EEG CHANNELS

dataframes = [alcoholic_df, control_df]
fig = plt.figure()

i = 0
for dataframe in dataframes:
    ax3d = fig.add_subplot(1, 2, i+1, projection = '3d')
    ax3d.set_zlim(-15, 15)
    
    x = range(0, len(dataframe.columns))
    y = range(0, len(dataframe.index))
    
    x, y = np.meshgrid(x, y)
    z = dataframe.values
    
    ax3d.plot_surface(x, y, z, cmap = cm.coolwarm)
    ax3d.set_ylabel('EEG Channel')
    ax3d.set_xlabel('Sample Number')
    ax3d.set_zlabel('Sensor Value')
    
    ax3d.contour(x, y, z, zdir = 'z', offset = -15, cmap = cm.coolwarm)
    
    ax3d.set_yticks(np.arange(0, len(dataframe.index), 1))
    ax3d.set_yticklabels(dataframe.index, rotation = 90, fontsize = 6)
    ax3d.set_xticks(np.arange(0, len(dataframe.columns), 10))
    ax3d.set_xticklabels(dataframe.index, rotation = 90, fontsize = 6)
    i += 1
    
plt.show()