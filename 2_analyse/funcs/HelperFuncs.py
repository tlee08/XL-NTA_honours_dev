import os
from glob import glob
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ipywidgets import interact, widgets
from IPython.display import display


"""
**************************************************************************************************
            DEFINING CONSTANTS
**************************************************************************************************
"""


INT8 = "<8"
UINT8 = "<B"
UINT16 = "<H"
UINT32 = "<I"
UINT64 = "<L"

N_UINT8 = ">B"
N_UINT16 = ">H"
N_UINT32 = ">I"
N_UINT64 = ">L"


"""
**************************************************************************************************
            MANAGING FOLDERS AND FILES
**************************************************************************************************
"""


def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass


def clean_dir_junk(dir_name):
    for fp in glob(os.path.join(dir_name, "._*")):
        os.remove(fp)


def get_name(fp):
    return os.path.splitext(os.path.split(fp)[1])[0]


def remove_dir(dir):
    clean_dir_junk(dir)
    shutil.rmtree(dir)


"""
**************************************************************************************************
            INTERACTIVE PLOT WITH SLIDER (FOR TESTING THE BEST HYPERPARAMS FOR THE PLOT)
**************************************************************************************************
"""


def InteractivePlot(csi):
    # NOTE: must add %matplotlib ipympl
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Getting csi amplitude and phase values
    csi_amp = np.abs(csi)
    csi_phase = np.angle(csi)

    # Define a function to update the plot with a specified vmax
    def update(vmax):
        ax.cla()  # Clear the current axis
        counts, xedges, yedges, im = ax.hist2d(
            x=np.tile(np.arange(csi.shape[1]), csi.shape[0]),
            y=csi_phase.flatten(),
            bins=(csi.shape[1] - 1, 100),
            norm=LogNorm(vmax=vmax),
            # norm=Normalize(vmax=300),
        )
        ax.set_title(f"vmax={vmax}")

    # Create a slider widget
    # vmax_slider = widgets.FloatSlider(
    #     value=500,
    #     min=50,  # Set the minimum value
    #     max=1000,  # Set the maximum value
    #     step=1.0,  # Set the step size
    #     description='vmax:',
    # )
    vmax_slider = widgets.FloatLogSlider(
        value=1.5e3,
        base=10,  # Base of the logarithm (default is 10)
        min=1.0,  # Set the minimum value
        max=4.0,  # Set the maximum value (adjust as needed)
        step=0.1,  # Set the step size
        description="vmax:",
    )
    # Connect the slider to the update function
    interact(update, vmax=vmax_slider)
    # Display the initial plot
    update(vmax_slider.value)
    # Show the interactive plot
    plt.show()
