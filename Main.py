# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:49 2023

@author: Jwaad
"""

import pygame
from pygame.locals import *
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
from scipy.interpolate import interp1d
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication
from lib.py import DragPoint


class DementiaSimulator():

    def __init__(self):
        self.Running = True
        self.NumberOnScreen = 0
        self.MinPeople = 1   # How many people have to be on screen, before we change the audio
        self.MaxPeople = 10  # Num people on screen, where audio will be most transformed
        self.MonitorDPI = self.GetDPI()
        self.SpeakerDragPoints = []
        self.NoiseDragPoints = []
        self.SpeakerDragPoints, self.NoiseDragPoints = self.LoadDragPoints()

    # Use pickle to load our previously saved drag points
    def LoadDragPoints(self):
        dragPoints = [[1, 90], [3, 80], [4, 70], [6, 40], [10, 0]]  # TEMP
        return dragPoints, dragPoints

    # Get DPI of screen
    def GetDPI(self):
        app = QApplication(sys.argv)
        screen = app.screens()[0]
        dpi = screen.physicalDotsPerInch()
        app.quit()
        return dpi

    # Create a GUI using pygame
    def InitScreen(self):
        self.BackgroundColour = (155, 155, 155)
        (width, height) = (1000, 800)
        pygame.init()
        self.Screen = pygame.display.set_mode((width, height))  # , DOUBLEBUF)
        pygame.display.set_caption('Dementia Simulator')
        self.Screen.fill(self.BackgroundColour)
        pygame.display.update()

    # Create a plot in matlab, and convert it into an image
    def CreateCurve(self, set_points, plot_size=[400, 200]):
        """ 
        setpoints refers to the output of the dragable squares used for curve fitting.
        These must be in the format: [[x1,y1],[x2,y2],etc]
        """

        # Fit curve to setpoints, and take as Ydata
        x_data = []
        y_data = []
        for data_pos in set_points:
            x_data.append(data_pos[0])
            y_data.append(data_pos[1])

        trf = interp1d(x_data, y_data, 'cubic')

        # Create new plot using our TRF, in our min and max range
        interp_y_data = []
        step_size = 0.1
        x_data = np.round(np.arange(self.MinPeople, self.MaxPeople +
                                    step_size, step_size), 3)
        for x in x_data:
            print(x)
            interp_y_data.append(trf(x))

        # Plot params
        matplotlib.use("Agg")
        # Convert pixels to inches, using DPI
        x_inches = plot_size[0] / self.MonitorDPI
        y_inches = plot_size[1] / self.MonitorDPI
        fig = pylab.figure(figsize=[x_inches, y_inches], dpi=self.MonitorDPI)
        # fig.patch.set_facecolor("none")
        fig.tight_layout()
        ax = fig.gca()
        ax.margins(x=0, y=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.arange(self.MinPeople, self.MaxPeople + 1, 1))
        # ax.axis("off")

        # Plot interpolated data
        ax.plot(x_data, interp_y_data)

        # Convert plot to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        plot_array = np.asarray(canvas.buffer_rgba())
        plot = pygame.image.frombuffer(
            plot_array.tobytes(), plot_array.shape[1::-1], "RGBA")
        return (plot)

    # Draw the plot onto the surface of pygame
    def DrawPlot(self, plot, location):
        self.Screen.blit(plot, location)

    # Draw the background
    def DrawBackground(self):
        self.Screen.fill(self.BackgroundColour)

    # Check quit
    def CheckQuit(self):
        for event in pygame.event.get():
            # Quit on x pressed
            if event.type == pygame.QUIT:
                self.Running = False

    # Main loop
    def Main(self):
        # Initialise pygame
        self.InitScreen()

        # Create initial plots
        speaker_volume_curve = self.CreateCurve(
            self.Example_setpoint, plot_size=[500, 250])
        noise_volume_curve = self.CreateCurve(
            self.Example_setpoint, plot_size=[500, 250])

        while self.Running:
            # Change vars based on UI touches (events)
            pass

            # Draw on screen elements
            self.DrawBackground()
            self.DrawPlot(speaker_volume_curve, (25, 75))
            self.DrawPlot(noise_volume_curve, (25, 400))

            # Update display
            pygame.display.update()

            # See if use quit program, or clicked the X
            self.CheckQuit()


if __name__ == '__main__':
    dementiaSim = DementiaSimulator()
    dementiaSim.Main()
