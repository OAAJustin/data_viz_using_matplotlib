## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import TextBox
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

## The matplotlib.widgets module defines a Button class - to connect to it call the on_clicked() function, which executes the function we supply. 
## Once a click has been detected, the function executes. 

## ADDING BUTTONS

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)
plot = ax.scatter([], [])

class EventHandler:
    def add_random(self, event):
        x = np.random.randint(0, 100)
        y = np.random.randint(0, 100)
        ax.scatter(x, y)
        plt.draw()
        
button_ax = plt.axes([0.7, 0.05, 0.2, 0.07])

button = Button(button_ax, 'Add Random', color = 'green', hovercolor = 'red')
button.on_clicked(EventHandler().add_random)

plt.show()

## 

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)

df = pd.read_csv('winequality-red.csv')
plot = ax.scatter([], [])

class EventHandler:
    i = 0
    # Find and plot next feature, re-draw the Axes    
    def next_feature(self, event):
        # If the counter is at the end of the columns
        # Revert it back to 0 to cycle through again
        if self.i >= len(df.columns):
            self.i = 0
        ## Clear axes from last plot
        ax.cla()
        # Plot a feature against a feature located on the 'i' column
        ax.scatter(df['alcohol'], df.iloc[:, self.i])
        # Set labels
        ax.set_xlabel('Alcohol')
        ax.set_ylabel(df.columns[self.i])
        # Increment i
        self.i += 1
        # Update Figure
        plt.draw()
        
    def previous_feature(self, event):
        # If the counter is at the start of the columns
        # Rever it back to the last column to cycle through
        if self.i <= 0:
            self.i = len(df.columns)-1
        # Clear Axes from last plot
        ax.cla()
        ax.scatter(df['alcohol'], df.iloc[:, self.i])
        # Set labels
        ax.set_xlabel('Alcohol')
        ax.set_ylabel(df.columns[self.i])
        # Increment i
        self.i += 1
        # Update Figure
        plt.draw()
        
# Add Buttons
button1_ax = plt.axes([0.7, 0.02, 0.2, 0.07])
next_button = Button(button1_ax, 'Next Feature')
next_button.on_clicked(EventHandler().next_feature)

button2_ax = plt.axes([0.45, 0.02, 0.2, 0.07])
previous_button = Button(button2_ax, 'Preivous Feature')
previous_button.on_clicked(EventHandler().previous_feature)

plt.show()

## ADDING RADIO BUTTONS AND CHECK BOXES

## Radio Buttons are used to allow a user to slect one value out of several values. Only one radio button can be selected at a time, and they typically represent a choice.
## Check Boxes can be used if you'd like to let the user select multiple options at once. 

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)

df = pd.read_csv('winequality-red.csv')

# Plot two line plots for two features, and turn them invisible

line1, = ax.plot(df['fixed acidity'], visible = False)
line2, = ax.plot(df['citric acid'], visible = False)

class EventHandler:
        # set_range handler
    def set_range(label):
        if (label == 'Small Range'):
            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 25)
        else:
            ax.set_xlim(0, 1600)
            ax.set_ylim(0, 50)
        plt.draw()
        
    # Turn off, if on, and on if off
    def apply_features(label):
        if (label == 'Fixed Acidity'):
            line1.set_visible(not line1.get_visible())
        elif (label == 'Citric Acid'):
            line2.set_visible(not line2.get_visible())
        plt.draw()
        
# Add radio buttons and check boxes

range_ax = plt.axes([0.7, 0.02, 0.2, 0.1])
range_radio_buttons = RadioButtons(range_ax, ('Small Range', 'Large Range'))
range_radio_buttons.on_clicked(EventHandler.set_range)

checkboxes_ax = plt.axes([0.4, 0.02, 0.2, 0.1])
checkboxes = CheckButtons(checkboxes_ax, ('Fixed Acidity', 'Citric Acid'))
checkboxes.on_clicked(EventHandler.apply_features)

plt.show()

## Adding Textboxes

## Textboxes are used to collect data from the user - and we can alter the plots based on this data. 

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)

df = pd.read_csv('winequality-red.csv')

class EventHandler:
    def submit(feature_name):
        if feature_name != '' or feature_name != None:
            if feature_name in df:
                ax.cla()
                ax.plot(df[feature_name])
            else:
                if len(textbox_ax.texts) > 2:
                    del textbox_ax.texts[-1]
                textbox_ax.text(-2, 0.4, feature_name + 'was not found.')
        plt.draw()
                
textbox_ax = plt.axes([0.7, 0.02, 0.2, 0.1])
textbox = TextBox(textbox_ax, 'Feature Name')
textbox.on_submit(EventHandler.submit)

plt.show()

## ADDING SPAN SELECTORS

## Span Selectors can be used to allow the user to select a span of data and focus on it, setting hte axis limits based on that selection. 

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2)

df = pd.read_csv('AmesHousing.csv')

ax.scatter(x = df['Year Built'], y = df['Total Bsmt SF'], alpha = 0.6)

class EventHandler:
    def select_horizontal(x, y):
        ax.set_xlim(x, y)
        plt.draw()
        
    def reset(self):
        ax.set_xlim(df['Year Built'].min(), df['Year Built'].max())
        plt.draw()
        
span_horizontal = SpanSelector(ax, EventHandler.select_horizontal, 'horizontal', useblit = True, props = dict(alpha = 0.5, facecolor = 'blue'))
    
button_ax = plt.axes([0.7, 0.02, 0.2, 0.07])
button = Button(button_ax, 'Reset')
button.on_clicked(EventHandler.reset)

plt.show()

## ADDING SLIDERS

## Sliders allow users to select between many values intuitively by sliding a marker and selecting a value.

fig, ax = plt.subplots()
fig.subplots_adjust(bottom = 0.2, left = 0.2)

df = pd.read_csv('winequality-red.csv')
plot, = ax.plot(df['volatile acidity'])

class EventHandler:
    def update(val):
        ax.set_ylim(0, yslider.val)
        ax.set_xlim(0, xslider.val)
        plt.draw()
        
xslider_ax = plt.axes([0.35, 0.03, 0.5, 0.07])
xslider = Slider(
    ax = xslider_ax, 
    label = 'x-limit',
    valmin = 0,
    valmax = len(df['volatile acidity']),
    valinit = len(df['volatile acidity']),
    orientation = 'horizontal'
)

yslider_ax = plt.axes([0.03, 0.2, 0.07, 0.5])
yslider = Slider(
    ax = yslider_ax,
    label = 'y-limit',
    valmin = 0,
    valmax = len(df['volatile acidity']),
    valinit = len(df['volatile acidity']),
    orientation = 'vertical'
)

xslider.on_changed(EventHandler.update)
yslider.on_changed(EventHandler.update)

plt.show()