from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def curves(splits):
    return splits + 1

axis_color = 'lightgoldenrodyellow'

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the slider and button
fig.subplots_adjust(left=0.25, bottom=0.25)


splits_0 = 0
t = len(curves(splits_0))

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax.plot(t, curves(splits_0), linewidth=2, color='red')
ax.set_xlim([0, 1])
ax.set_ylim([-10, 10])

# Add slider for tweaking the parameters

# Define an axes area and draw a slider in it
splits_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axis_color)
split_slider = Slider(split_slider_ax, 'Splits', 0, 10, valinit=amp_0)

# Define an action for modifying the line when any slider's value changes
def slider_on_changed(val):
    line.set_ydata(curves(split_slider.val))
    fig.canvas.draw_idle()
split_slider.on_changed(slider_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    split_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color
color_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15], axisbg=axis_color)
color_radios = RadioButtons(color_radios_ax, ('red', 'blue', 'green'), active=0)
def color_radios_on_clicked(label):
    line.set_color(label)
    fig.canvas.draw_idle()
color_radios.on_clicked(color_radios_on_clicked)

plt.show()

#template taken from stackOverflow
