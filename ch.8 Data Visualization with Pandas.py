import pandas as pd
from matplotlib import pyplot as plt

## THE DataFrame.plot() FUNCTION

## The plot() function accepts x and y features, and a kind argument. To avoid the kind argument, you casn also call DataFrame.plot.kind() instead, 
## and pass in the x and y features. 
## The accepted values for the kind argument are: line, bar, barh (horizontal bar), hist, box, kde, density (synonym for kde), area, pie, scatter and hexbin 
## (similar to a Heatmap)

df = pd.read_csv('gdp_csv.csv')
df_eu = df.loc[df['Country Name'] == 'European Union']

df_eu.plot.bar(x = 'Year', y = 'Value')
df_eu.plot.line(x = 'Year', y = 'Value')
df_eu.plot.box(x = 'Value')

plt.show()

## Plot on one figure

df = pd.read_csv('gdp_csv.csv')
df_eu = df.loc[df['Country Name'] == 'European Union']

fig, ax = plt.subplots(3, 1)

df_eu.plot.box(x = 'Value', ax = ax[0])
df_eu.plot.line(x = 'Year', y = 'Value', ax = ax[1])
df_eu.plot.bar(x = 'Year', y = 'Value', rot = 45, ax = ax[2])

plt.show()

## PANDAS' PLOTTING MODULE

## BOOTSTRAP PLOT

## Bootstrapping is the process of randomly sampling (with replacement) a dataset, and calculating measures of accuracy such as bias, variance and confidence intervals 
## for the random samples. 
## 'With replacement', means that each randomly selected element can be selected again. 'Without replacement' means that after each randomly selected element, it is removed
## from the pool for the next sample. 
## A Bootstrap Plot bootstraps the mean, median and mid-range statistics of a dataset, based on the sample size, shown via plt.show()
## The default arguments for size and samples are 50 and 500 respectively. 

df = pd.read_csv('gdp_csv.csv')
df_eu = df.loc[df['Country Name'] == 'European Union']

pd.plotting.bootstrap_plot(df_eu['Value'])

plt.show()

## AUTOCORRELATION PLOT

## Autocorrelation Plots are used to check for data randomness, for time-series data. Multiple autocorrelations are calculated for differing timestamps,
## and if the data is truly random - the correlation will be near zero. If not - the correlation will be larger than zero. 

## The Autocorrelation Plot for the random_series should revolve around 0, since it's random data, while the plot for the Value feature won't. 

df = pd.read_csv('gdp_csv.csv')
df_eu = df.loc[df['Country Name'] == 'European Union']

random_series = pd.Series(np.random.randint(0, 100, size = 50))

pd.plotting.autocorrelation_plot(df_eu['Value'])
pd.plotting.autocorrelation_plot(random_series)

plt.show()

## SCATTER MATRICES

## Scatter Matrices plot a grid of Scatter Plots for all features against all features. Scatter Matrices are also known as Pair Plots. Seaborn offers a pairplot()

df = pd.read_csv('c:\dev\data_visualization_in_python\data-visualization-in-python\worldhappiness/2019.csv')

axes = pd.plotting.scatter_matrix(df, diagonal = 'hist')

for ax in axes.flatten():
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
    
plt.show()

## You can pass in the diagonal argument, which accepts 'hist' or 'kde' to specify what type of distribution plot you'd like to plot on the diagonal, as well as alpha, 
## specifying the translucency of the markers in the Scatter Plots