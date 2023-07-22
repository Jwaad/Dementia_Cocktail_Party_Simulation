# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:49 2023

@author: Jwaad
"""
import mtcnn
import pygame
from pygame.locals import *
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
from scipy.interpolate import interp1d
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication
from GameElementsLib import DragPoint
from GameElementsLib import InputBox
from GameElementsLib import VideoCapture as VC
from GameElementsLib import Checkbox
import cv2 as cv
import pickle


class DementiaSimulator():

    def __init__(self):
        self.Running = True
        self.Crowdedness
        self.NumberOnScreen = 1
        self.MinPeople = 1   # How many people have to be on screen, before we change the audio
        self.MaxPeople = 15  # Num people on screen, where audio will be most transformed
        self.MonitorDPI = self.GetDPI()
        self.SpeakerPointsPos, self.NoisePointsPos = self.LoadData()
        self.FPS = 20
        self.Clock = pygame.time.Clock()
        self.SpeakerPoints = []
        self.SpeakerVolume = 1
        self.NoisePoints = []
        self.NoiseVolume = 0
        self.SpeakerTRF = None
        self.NoiseTRF = None
        self.Stream = None
        self.ShowCamera = True
        self.UseStream = False
        self.Detector = None
        self.OpenFigures = []

    # Detect Camera and establish a video stream
    def StartCameraStream(self):
        self.Detector = mtcnn.MTCNN()
        print("Starting Camera Stream. This might take a while...")
        self.Stream = VC(0)
        if not self.Stream.cap.isOpened():
            print("Cannot open camera")

    # Process frame
    def ProcessFrame(self):
        "Get latest frame, and count faces"
        # Get latest frame
        frame = self.Stream.read()
        frame = cv.pyrDown(frame)  # Downscale for less lag
        faces = self.Detector.detect_faces(frame)
        # print(len(faces))
        if self.ShowCamera:
            for face in faces:
                x, y, w, h = face['box']
                cv.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 3)
            cv.imshow('Face Counter', frame)
            if cv.waitKey(1) == ord('q'):
                self.ShowCamera = False
        return len(faces)

    # Use pickle to load our previously saved drag points

    def BackupLoadData(self):
        # USE DEFAULT CURVE IF COULDNT FIND SAVE
        speaker_Points = [90, 80, 70, 40, 10, 0]
        noise_points = speaker_Points.copy()
        noise_points.reverse()
        return speaker_Points, noise_points

    # Use pickle to load our previously saved drag points
    def LoadData(self):
        print("Loading saved data...")
        try:
            save_data = pickle.load(open("data.pkl", "rb"))
            speaker_Points = save_data["story"]
            noise_points = save_data["noise"]
        except Exception as e:
            print("FAILED TO LOAD SAVE, USING DEFAULT PLOTS")
            speaker_Points, noise_points = self.BackupLoadData()

        return speaker_Points, noise_points

    def SaveData(self):
        print("Saving data...")
        saveDict = {}
        saveDict["story"] = self.SpeakerPointsPos
        saveDict["noise"] = self.NoisePointsPos
        pickle.dump(saveDict, open("data.pkl", "wb"))

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
        self.Screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Dementia Simulator')
        self.Screen.fill(self.BackgroundColour)
        pygame.display.update()

    def SpawnDragObjects(self, graphSize, origin_xy, dragPoints):
        # Spawn drag points for speaker curve
        origin_x = origin_xy[0]
        origin_y = origin_xy[1]
        # len -1 cause we want both 0 and 100 x
        stepSize = graphSize[1] / (len(dragPoints) - 1)
        i = 0
        SpawnedPoints = []
        graphStepSize = (self.MaxPeople - self.MinPeople) / \
            (len(dragPoints) - 1)
        for yGraphPos in dragPoints:
            x_pos = origin_x + (stepSize * i)
            # Convert from % to ypos
            y_pos = origin_y + ((1 - (yGraphPos / 100)) * graphSize[0])
            clampMin = origin_y
            clampMax = origin_y + graphSize[0]
            clamp = (clampMin, clampMax)
            xGraphPos = self.MinPeople + (graphStepSize * i)
            dragObject = DragPoint((x_pos, y_pos), clamp, xGraphPos)
            SpawnedPoints.append(dragObject)
            i += 1
        return SpawnedPoints

    # Draw faint vertical line, showing current volume %
    def DrawGraphStats(self, screen):
        """Draws a line on graph and writes the current volume % above the line

        Args:
            screen (pygame.screen)
        """
        # Shared variables
        font = pygame.font.Font('freesansbold.ttf', 20)
        colour = (0, 100, 100)

        lineHeight = 193  # Note this is hard coded in 2 places
        graphWidth = 387
        x_start = 88
        stepSize = graphWidth / (self.MaxPeople - self.MinPeople)
        if self.NumberOnScreen <= self.MinPeople:
            x_pos = 0
        elif self.NumberOnScreen >= self.MaxPeople:
            x_pos = (self.MaxPeople - 1) * stepSize
        else:
            x_pos = (self.NumberOnScreen - self.MinPeople) * stepSize

        # Speaker
        y_origin = 105
        pygame.draw.line(screen, colour, (x_start + x_pos, y_origin),
                         (x_start + x_pos, (y_origin + lineHeight)), 2)
        text = font.render('{}%'.format(
            self.SpeakerVolume), True, colour)
        textRect = text.get_rect()
        textRect[0] = x_start + x_pos
        textRect[1] = y_origin - (textRect[3])
        screen.blit(text, textRect)

        """
        # Noise
        y_origin = 430
        pygame.draw.line(screen, colour, (x_start + x_pos, y_origin),
                         (x_start + x_pos, (y_origin + lineHeight)), 2)
        text = font.render('{}%'.format(
            self.NoiseVolume), True, colour)
        textRect = text.get_rect()
        textRect[0] = x_start + x_pos
        textRect[1] = y_origin - (textRect[3])
        screen.blit(text, textRect)
        """

    # Spawn drag points for speaker and noise audio
    def CreateDragPoints(self):
        # Shared vars
        graphSize = (193, 387)

        self.SpeakerPoints = self.SpawnDragObjects(
            graphSize, [88, 105], self.SpeakerPointsPos)
        self.NoisePoints = self.SpawnDragObjects(
            graphSize, [88, 430], self.NoisePointsPos)

    # For debugging, read the pos of the mouse in the screen
    def TrackMousePos(self, event):
        if event.type == MOUSEMOTION:
            print(event.pos)

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

        # Interpolate
        trf = interp1d(x_data, y_data, 'cubic')

        # Create new plot using our TRF, in our min and max range
        interp_y_data = []
        step_size = 0.1
        x_data = np.round(np.arange(self.MinPeople, self.MaxPeople +
                                    step_size, step_size), 3)
        for x in x_data:
            interp_y_data.append(trf(x))

        # Clear old plots:
        if self.OpenFigures != []:
            for i in range(len(self.OpenFigures) - 1, - 1, - 1,):
                fig = self.OpenFigures.pop(i)
                matplotlib.pyplot.close(fig)

        # Plot params
        matplotlib.use("Agg")
        # Convert pixels to inches, using DPI
        x_inches = plot_size[0] / self.MonitorDPI
        y_inches = plot_size[1] / self.MonitorDPI
        fig = pylab.figure(figsize=[x_inches, y_inches], dpi=self.MonitorDPI)
        fig.tight_layout()
        ax = fig.gca()
        ax.margins(x=0, y=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.arange(self.MinPeople, self.MaxPeople + 1, 1))
        ax.set_yticks(np.arange(0, 110, 10))
        ax.grid()

        # Plot interpolated data
        ax.plot(x_data, interp_y_data, linewidth=3.0)

        # Convert plot to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        plot_array = np.asarray(canvas.buffer_rgba())
        plot = pygame.image.frombuffer(
            plot_array.tobytes(), plot_array.shape[1::-1], "RGBA")
        return (plot, trf)

    # Draw the plot onto the surface of pygame
    def DrawPlot(self, plot, location):
        self.Screen.blit(plot, location)

    # Draw the background
    def DrawBackground(self):
        self.Screen.fill(self.BackgroundColour)

    # Check quit
    def CheckQuit(self, event):
        # Quit on x pressed
        if event.type == pygame.QUIT:
            self.Running = False
            self.OnExit()

    def OnNumPeopleChange(self, newNum):
        # If num not in range, dont do anything
        if newNum <= self.MinPeople:
            newNum = self.MinPeople
        elif newNum >= self.MaxPeople:
            newNum = self.MaxPeople
        self.SpeakerVolume = self.SpeakerTRF(newNum)
        self.NoiseVolume = self.NoiseTRF(newNum)
        self.NumberOnScreen = newNum
        self.channel1.set_volume(self.SpeakerVolume/100)
        self.channel2.set_volume(self.NoiseVolume/100)
        # update audio

    def StartAudio(self):
        """
        Get default volumes from our dragables
        Begin both audio tracks at default volumes
        """
        self.SpeakerVolume = (self.SpeakerPointsPos[0])
        self.NoiseVolume = (self.NoisePointsPos[0])

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2**12)
        # create separate Channel objects for simultaneous playback
        self.channel1 = pygame.mixer.Channel(0)  # argument must be int
        self.channel2 = pygame.mixer.Channel(1)

        self.channel1.set_volume(self.SpeakerVolume/100)
        self.channel2.set_volume(self.NoiseVolume/100)
        story = pygame.mixer.Sound("audio/story.mp3")
        noise = pygame.mixer.Sound("audio/noise.mp3")
        self.channel1.play(story, loops=-1)
        self.channel2.play(noise, loops=-1)

    # Do some stuff on exit
    def OnExit(self):
        self.SaveData()

    # Main method
    def Main(self):

        # Initialise
        # self.StartCameraStream()
        self.InitScreen()
        self.CreateDragPoints()
        numberPeopleText = InputBox(575, 100, 50, 32)
        useCameraToggle = Checkbox(self.Screen, 700, 100)
        self.StartAudio()  # Call on first loop to get %s to defaults
        numberOnScreen = self.NumberOnScreen

        # Workaround: Forces our first loop to calculate plots
        self.SpeakerPointsPos, self.NoisePointsPos = [], []

        # Main loop
        while self.Running:

            # Handle Events
            events = pygame.event.get()
            for event in events:
                # self.TrackMousePos(event)
                self.CheckQuit(event)
                numberPeopleText.handle_event(event)
                if numberPeopleText.returnInput != None:
                    numberOnScreen = int(numberPeopleText.returnInput)

                # Allow drag points to be dragged
                speakerPointsPos = []
                for dragPoint in self.SpeakerPoints:
                    dragPoint.handle_event(event)
                    speakerPointsPos.append(dragPoint.GetPercentageHeight())
                noisePointsPos = []
                for dragPoint in self.NoisePoints:
                    dragPoint.handle_event(event)
                    noisePointsPos.append(dragPoint.GetPercentageHeight())

                # Let use activate camera with button
                useCameraToggle.update_checkbox(event)
                if useCameraToggle.is_checked():
                    self.UseStream = True
                else:
                    self.UseStream = False

            # Display camera stream
            if self.UseStream:
                numFaces = self.ProcessFrame()
                numberOnScreen = numFaces

            # If number on screen changed, change volume too
            if self.NumberOnScreen != numberOnScreen:
                self.OnNumPeopleChange(numberOnScreen)

            # If the Y of any of the points change, redraw curves
            if self.SpeakerPointsPos != speakerPointsPos:
                newSetPoints = []
                for object in self.SpeakerPoints:
                    newSetPoints.append(object.GetGraphPos())
                speaker_volume_curve, self.SpeakerTRF = self.CreateCurve(
                    newSetPoints, plot_size=[500, 250])
                self.SpeakerPointsPos = speakerPointsPos
            if self.NoisePointsPos != noisePointsPos:
                newSetPoints = []
                for object in self.NoisePoints:
                    newSetPoints.append(object.GetGraphPos())
                noise_volume_curve, self.NoiseTRF = self.CreateCurve(
                    newSetPoints, plot_size=[500, 250])
                self.NoisePointsPos = noisePointsPos

            # Draw on screen elements
            self.DrawBackground()
            self.DrawPlot(speaker_volume_curve, (25, 75))
            self.DrawPlot(noise_volume_curve, (25, 400))
            self.DrawGraphStats(self.Screen)
            numberPeopleText.draw(self.Screen)
            for dragPoint in self.SpeakerPoints:
                dragPoint.Render(self.Screen)
            for dragPoint in self.NoisePoints:
                dragPoint.Render(self.Screen)
            useCameraToggle.render_checkbox()

            # Update display
            pygame.display.update()

            # Lock program to fps
            self.Clock.tick(self.FPS)


if __name__ == '__main__':
    print("Starting. This may take a while...")
    dementiaSim = DementiaSimulator()
    dementiaSim.Main()
