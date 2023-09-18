## Iport Libraries

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

## UNDERSTANDING MATPLOTLIB STYLESHEETS

## A stylesheet contains a set of parameters that change the look of Matplotlib's elements. 
## Each Matplotlib stylesheet is a file with a bunmch of tuned parameters, just like a CSS file. 
## These include various elements such as axes.spines.linewidth, grid.linestyle, xtick.labelsize, text.color, etc. 
## These are elements you can manually set through setter functions or by changing the Runtime COnfigurations(rc) parameters. 
## All of these settings reside in style_name.mplstyle files and can be created manually and used. 

## Print out the available stylesheets from the plt.style module

for style in plt.style.available:
    print(style)
    
## A lot of these stylesheets are inspired by Seaborn, another Data Visualization library that's built on top of Matplotlib

plt.style.use('seaborn')

fig, ax = plt.subplots(figsize = (6, 2))
plt.show()

## 

df = pd.read_csv('gdp_csv.csv')

plt.style.use('dark_background')


df_eu = df.loc[df['Country Name'] == 'European Union']
df_na = df.loc[df['Country Name'] == 'North America']
df_sa = df.loc[df['Country Name'] == 'South Asia']
df_ea = df.loc[df['Country Name'] == 'East Asia & Pacific']

fig, ax = plt.subplots(figsize = (6,2))
ax.plot(df_eu['Year'], df_eu['Value'], label = 'European Union GDP per Year')
ax.plot(df_na['Year'], df_na['Value'], label = 'North America GDP per Year')
ax.plot(df_sa['Year'], df_sa['Value'], label = 'South Asia GDP per Year')
ax.plot(df_ea['Year'], df_ea['Value'], label = 'East Asia & Pacific GDP per Year')

ax.legend()
plt.show()

## Style 'fivethirtyeight'

df = pd.read_csv('gdp_csv.csv')

plt.style.use('fivethirtyeight')


df_eu = df.loc[df['Country Name'] == 'European Union']
df_na = df.loc[df['Country Name'] == 'North America']
df_sa = df.loc[df['Country Name'] == 'South Asia']
df_ea = df.loc[df['Country Name'] == 'East Asia & Pacific']

fig, ax = plt.subplots(figsize = (6,2))
ax.plot(df_eu['Year'], df_eu['Value'], label = 'European Union GDP per Year')
ax.plot(df_na['Year'], df_na['Value'], label = 'North America GDP per Year')
ax.plot(df_sa['Year'], df_sa['Value'], label = 'South Asia GDP per Year')
ax.plot(df_ea['Year'], df_ea['Value'], label = 'East Asia & Pacific GDP per Year')

ax.legend()
plt.show()

## Style 'seaborn' 

df = pd.read_csv('gdp_csv.csv')

plt.style.use('seaborn')


df_eu = df.loc[df['Country Name'] == 'European Union']
df_na = df.loc[df['Country Name'] == 'North America']
df_sa = df.loc[df['Country Name'] == 'South Asia']
df_ea = df.loc[df['Country Name'] == 'East Asia & Pacific']

fig, ax = plt.subplots(figsize = (6,2))
ax.plot(df_eu['Year'], df_eu['Value'], label = 'European Union GDP per Year')
ax.plot(df_na['Year'], df_na['Value'], label = 'North America GDP per Year')
ax.plot(df_sa['Year'], df_sa['Value'], label = 'South Asia GDP per Year')
ax.plot(df_ea['Year'], df_ea['Value'], label = 'East Asia & Pacific GDP per Year')

ax.legend()
plt.show()

## Style 'ggplot'

df = pd.read_csv('gdp_csv.csv')

plt.style.use('ggplot')


df_eu = df.loc[df['Country Name'] == 'European Union']
df_na = df.loc[df['Country Name'] == 'North America']
df_sa = df.loc[df['Country Name'] == 'South Asia']
df_ea = df.loc[df['Country Name'] == 'East Asia & Pacific']

fig, ax = plt.subplots(figsize = (6,2))
ax.plot(df_eu['Year'], df_eu['Value'], label = 'European Union GDP per Year')
ax.plot(df_na['Year'], df_na['Value'], label = 'North America GDP per Year')
ax.plot(df_sa['Year'], df_sa['Value'], label = 'South Asia GDP per Year')
ax.plot(df_ea['Year'], df_ea['Value'], label = 'East Asia & Pacific GDP per Year')

ax.legend()
plt.show()

## MATPLOTLIB RUNTIME CONFIGURATION (RC) PARAMETERS

## Runtime Configuration Parameters control various aspects of Matplotlib plots and Figures. 

## You can access them via the plt instance: plt.rc('group', **kwargs) or via: plt.rcParams['group.param'] = value

## Returning them back to the default values: plt.rcdefaults() or via: plt.style.use('default')

## All available keys to set arguments for:

print('Number of keys: ', len(plt.rcParams.keys()))
print(plt.rcParams.keys())

## Run a few sets of runtime configuration parameters on the fly, in the code itself

df = pd.read_csv('gdp_csv.csv')

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.grid'] = True

plt.rc('lines', linestyle = '--', color = 'red')

settings = {
    'color' : 'g'
}

plt.rc('text', **settings)

df_eu = df.loc[df['Country Name'] == 'European Union']
fig, ax = plt.subplots(figsize = (6, 2))
ax.plot(df_eu['Year'], df_eu['Value'], label = 'European Union GDP per Year')

ax.legend()
plt.show()

## UNDERSTANDING MATPLOTLIB COLORS AND COLORMAPS

## The matplotlib.colors module is the module that takes care of mapping numbers and color arguments to RGB and RGBA values. 
## The colors API accepts various formats, and converts them into the underlying RGB values:

## RGB tuples - (0.0, 0.0, 0.0) denoting the RGB values
## Hex strings - #49CE3D, ##0055DFF, etc
## String representations of float values - 0 for black, 1 for white, and 0.[0-99] for values in-between
## Character representations of colors - r, g, b, c, m, y, k, w
## X11/CSS names - burlywood, bisque, black, darkcyan, etc
## T10/Tableau colors -= tab:blue, tab:orange, tab:gray, etc

fig, ax = plt.subplots(6, 1)
x = [1, 3, 4, 2, 5, 3, 4, 6, 1]

# Tuple RGB representation
ax[0].plot(x, color = (0.2, 0.5, 0.1))
# Hex String representation
ax[1].plot(x, color = '#005DFF')
# String-float representation
ax[2].plot(x, color = '0.67')
# Character representation
ax[3].plot(x, color = 'g')
# X11/CSS names
ax[4].plot(x, color = 'aquamarine')
# T10/Tableau colors
ax[5].plot(x, color = 'tab:orange')

plt.show()

## If a plot type allows for multiple values

fig, ax = plt.subplots(6, 1)

x = range(0, 10)
y = [1, 3, 4, 2, 5, 3, 4, 6, 1, 5]

ax[0].bar(x, y, color = [(0.2, 0.5, 0.1), (0.1, 0.2, 0.7), (0, 0, 0)])
ax[1].bar(x, y, color = ['#005DFF', '#DEB887', '#D2691E'])
ax[2].bar(x, y, color = ['0.67', '0.5', '0.2'])
ax[3].bar(x, y, color = ['r', 'g', 'b'])
ax[4].bar(x, y, color = ['aquamarine', 'burlywood', 'chocolate'])
ax[5].bar(x, y, color = ['tab:orange', 'tab:blue', 'tab:green'])

plt.show()

## COLORMAPS

## Perceptually Uniform - steps in data are equally reflected as steps in color
## The lightness of a color is perceptually correlated with the underlying data. On the other hand, the hue is much less so - because of this, most perceptually uniform
## colormaps rely mainly on the increase of the lightness, and less so on the parameters. 

## There are four types of colormaps:

## Sequential
## Diverging
## Cyclic
## Qualitative

## Sequential colorms are typically used for ordered data, that follows a range - 0..N. They linearly/sequentially increase the lightness and/or hue
## Diverging colormaps are typically used for data revolving around a certain value - such as zero or peak. They contain two colors/hues that are intense on the ends, 
## but reach an unsaturated point in the middle, that's shared. The coolwarm colormap is a diverging colormap since on one end, we've got hues of blue, on the other end,
## we've got hues of red, an din the center - we've got an unsaturated grey.
## Cyclic colormaps start and end with the same color, allowing us to repeat them over multiple values seamlessly when confronted with cyclic data, such as days
## Qualitative colormaps are used for unordered, qualitative data, where the changes in data don't reflect an equal change in the colors

## Sequential, diverging and cyclic colormaps can, but not always, be perceptually uniform, depending on the specific colormap itself.
## Qualitative colormaps aren't meant to be perceptually uniform, since the data they're being used on is typically hectic and doesn't have order

## Visualization

sequential_colormaps = ['Greys', 'Blues', 'Reds', 'Greens', 'GnBu', 'PuBu']
diverging_colormaps = ['PiYG', 'PuOr', 'coolwarm', 'seismic', 'Spectral']
cyclic_colormaps = ['twilight', 'hsv']
qualitative_colormaps = ['Pastel1', 'Pastel2', 'tab10', 'tab20', 'Paired']

colormaps = [sequential_colormaps, 
             diverging_colormaps, 
             cyclic_colormaps, 
             qualitative_colormaps]

x = np.linspace(0, 50, 100).reshape((10, 10))

for colormap_list in colormaps:
    fig = plt.figure()
    i = 0
    for colormap_name in colormap_list:
        ax = fig.add_subplot(1, len(colormap_list), i + 1)
        ax.imshow(x, cmap = colormap_name)
        i += 1
        
plt.show()

## CUSTOMIZING LAYOUTS WITH GridSpec

## Attempt 1

df = pd.read_csv('winequality-red.csv')

alcohol = df['alcohol']
quality =df['quality']

fig, ax = plt.subplots(2, 2)

ax[0][0].hist(alcohol)
ax[1][0].scatter(x = alcohol, y = quality)
ax[1][0].set_xlabel('Alcohol')

ax[1][1].hist(quality, orientation = 'horizontal')
ax[1][1].set_xlabel('Quality')

plt.show()

## GridSpec used to specify a layout into which we can place subpolots - Axes instances. The GridSpec instance is created separately from the Figure.
## It is used to create an Axes while adding its respective Figure

fig = plt.figure()
gs = gridspec.GridSpec(ncols = 2, nrows = 1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

plt.show()

## There is a high probability that some elements will overlap between Axes, so its' a good idea to use a constrained_layout for your figure.
## When setting the constrained_layout to True, we also have to define the optional figure argument of the GridSpec constructor to our fig.

fig = plt.figure(constrained_layout = True)
gs = gridspec.GridSpec(ncols = 2, nrows = 2, figure = fig)

ax1 = fig.add_subplot(gs[0:1])
ax1.set_title('gridspec[0:1]')

ax2 = fig.add_subplot(gs[1:2])
ax2.set_title('gridspec[1:2]')

ax2 = fig.add_subplot(gs[2:3])
ax2.set_title('gridspec[2:3]')

ax2 = fig.add_subplot(gs[3:4])
ax2.set_title('gridspec[3:4]')

plt.show()

## Adjust code to reflect ax1 taking up the space of 0..3

fig = plt.figure(constrained_layout = True)
gs = gridspec.GridSpec(ncols = 2, nrows = 2, figure = fig)

ax1 = fig.add_subplot(gs[0:3])
ax1.set_title('gridspec[0:3]')

ax2 = fig.add_subplot(gs[1:2])
ax2.set_title('gridspec[1:2]')

ax2 = fig.add_subplot(gs[3:4])
ax2.set_title('gridspec[3:4]')

plt.show()

## Slice notation can be used to define a range of positions in the layout

fig = plt.figure(constrained_layout = True)
gs = gridspec.GridSpec(ncols = 2, nrows = 2, figure = fig)

ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('gridspec[0, :]')

ax2 = fig.add_subplot(gs[3:])
ax2.set_title('gridspec[3:]')

plt.show()

## Negative indexing can be used to signify these slots in reverse order - in the above case - gridspec[-1: ] and gridspec[3: ] evaluate to the same slots

df = pd.read_csv('Iris.csv')

fig = plt.figure()
gs = gridspec.GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1: 4, 0: 3])
ax_hist_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

plt.show()

## 

df = pd.read_csv('winequality-red.csv')

alcohol = df['alcohol']
quality = df['quality']

fig = plt.figure()
gs = gridspec.GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0, 0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(x = alcohol, y = quality)
ax_hist_y.hist(alcohol)
ax_hist_x.hist(quality, orientation = 'horizontal')

plt.show()