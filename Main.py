# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:49 2023

@author: Jwaad

Program to simulate what its like to be in group setting for some dementia sufferers.
Program alters 2 seperate audio tracks based on the position of the crowd
Noise volume changes with how close the crowd is to the speaker (on average)
Speech volume changes based on the amount of people in the room.
"""

import sys
import pickle
import pygame
from pygame.locals import *
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
from scipy.interpolate import interp1d
import numpy as np
from PyQt5.QtWidgets import QApplication
from GameElementsLib import DragPoint
from GameElementsLib import InputBox
from GameElementsLib import VideoCapture as VC
from GameElementsLib import Checkbox
from GameElementsLib import Button
import cv2 as cv


class DementiaSimulator():

    def __init__(self):
        self.Running = True
        self.Crowdedness = 0
        self.CrowdednessMin = 0
        self.CrowdednessMax = 1
        self.NumberOnScreen = 0
        self.MinPeople = 1   # How many people have to be on screen, before we change the audio
        self.MaxPeople = 10  # Num people on screen, where audio will be most transformed
        self.MonitorDPI = self.GetDPI()
        self.SpeakerPointsPos, self.NoisePointsPos = self.LoadData()
        self.FPS = 60
        self.Clock = pygame.time.Clock()
        self.SpeakerPoints = []
        self.SpeakerVolume = 1
        self.NoisePoints = []
        self.NoiseVolume = 0
        self.CameraIndex = 1
        self.SpeakerTRF = None
        self.NoiseTRF = None
        self.Stream = None
        self.ShowCamera = True
        self.UseStream = False
        self.Detector = None
        self.OpenFigures = []
        self.BackgroundImage = None
        self.AudioPath = "audio/"
        self.NoiseFileName = "noise5.wav"
        self.SpeechFileName = "speech.wav"
        self.graphSize = [195, 387]
        self.SpeakerGraphOrigin = [88, 105]
        self.NoiseGraphOrigin = [88, 430]

    # Detect Camera and establish a video stream
    def StartCameraStream(self):
        print("Starting Camera Stream. This might take a while...")
        self.Stream = VC(self.CameraIndex)
        if not self.Stream.cap.isOpened():
            print("Cannot open camera")
            return
        self.BackgroundImage = self.Stream.read()

    # Process frame
    def ProcessFrame(self):
        "Get latest frame, and count faces"
        # Get latest frame
        frame = self.Stream.read()
        #frame = cv.pyrDown(frame)  # Downscale for less lag
        hairRects = 5
        crowdedness = 0.5

        if self.ShowCamera:
            #for rect in hairRects:
            #    x, y, w, h = face['box']
            #    cv.rectangle(frame, (x, y), ((x+w), (y+h)), (255, 0, 0), 3)
            cv.imshow('Camera Feed', frame)
            if cv.waitKey(1) == ord('q'):
                self.ShowCamera = False

        return hairRects, crowdedness

    # Use pickle to load our previously saved drag points
    def BackupLoadData(self):
        # USE DEFAULT CURVE IF COULDNT FIND SAVE
        default_data = {'story': [104.66321243523315, 92.2279792746114, 84.4559585492228, 64.24870466321244, 40.932642487046635, 28.497409326424872], 'noise': [
            5.699481865284972, 30.569948186528496, 56.476683937823836, 75.12953367875647, 87.56476683937824, 105.18134715025906]}
        speaker_Points = default_data["story"]
        noise_points = default_data["noise"]
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
        print(saveDict)
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

    # Initialise setpoints / drag points
    def SpawnDragObjects(self, graphSize, origin_xy, dragPoints, min, max):
        # Spawn drag points for speaker curve
        origin_x = origin_xy[0]
        origin_y = origin_xy[1]
        # len -1 cause we want both 0 and 100 x
        stepSize = graphSize[1] / (len(dragPoints) - 1)
        i = 0
        SpawnedPoints = []
        # Distribute setpoints equidistant along graph
        graphStepSize = (max - min) / \
            (len(dragPoints) - 1)
        for yGraphPos in dragPoints:
            x_pos = origin_x + (stepSize * i)
            # Convert from % to ypos
            clampMin = origin_y
            clampMax = origin_y + graphSize[0]
            clamp = (clampMin, clampMax)
            xGraphPos = min + (graphStepSize * i)
            dragObject = DragPoint(x_pos, yGraphPos, clamp, xGraphPos)
            SpawnedPoints.append(dragObject)
            i += 1
        return SpawnedPoints

    # Draw thin vertical line, showing current volume %
    def DrawGraphStats(self, screen):
        """Draws a line on graph and writes the current volume % above the line

        Args:
            screen (pygame.screen)
        """
        # Shared variables
        font = pygame.font.Font('freesansbold.ttf', 20)
        colour = (0, 0, 0)
        lineHeight = self.graphSize[0] # 193 
        graphWidth = self.graphSize[1] # 387
        x_start = self.SpeakerGraphOrigin[0]

        # Step size for Speaker volume
        stepSize = graphWidth / (self.MaxPeople - self.MinPeople)
        if self.NumberOnScreen <= self.MinPeople:
            x_pos = 0
        elif self.NumberOnScreen >= self.MaxPeople:
            x_pos = (self.MaxPeople - 1) * stepSize
        else:
            x_pos = (self.NumberOnScreen - self.MinPeople) * stepSize

        # Speaker
        y_origin = self.SpeakerGraphOrigin[1]
        pygame.draw.line(screen, colour, (x_start + x_pos, y_origin),
                         (x_start + x_pos, (y_origin + lineHeight)), 2)
        # Draw volume
        text = font.render('{}%'.format(
            round(self.SpeakerVolume, 1)), True, colour)
        textRect = text.get_rect()
        textRect[0] = x_start + x_pos
        textRect[1] = y_origin - (textRect[3])
        screen.blit(text, textRect)
        
        # Draw title, speaker
        font = pygame.font.Font('freesansbold.ttf', 35)
        text = font.render('{}%'.format(
            "Speech Volume"), True, colour)
        textRect = text.get_rect()
        textRect[0] = x_start + (self.graphSize[1] / 2) - (textRect[2]/2)
        textRect[1] = y_origin - (textRect[3]) - 45
        screen.blit(text, textRect)

        # Step sizes and clamp noise to graph
        if self.Crowdedness <= self.CrowdednessMin:
            x_pos = 0
        elif self.Crowdedness >= self.CrowdednessMax:
            x_pos = graphWidth
        else:
            x_pos = (self.Crowdedness * graphWidth)

        # Noise
        font = pygame.font.Font('freesansbold.ttf', 20)
        y_origin = self.NoiseGraphOrigin[1]
        pygame.draw.line(screen, colour, (x_start + x_pos, y_origin),
                         (x_start + x_pos, (y_origin + lineHeight)), 2)
        text = font.render('{}%'.format(
            round(self.NoiseVolume,1)), True, colour)
        textRect = text.get_rect()
        textRect[0] = x_start + x_pos
        textRect[1] = y_origin - (textRect[3])
        screen.blit(text, textRect)
        
        # Draw title noise
        font = pygame.font.Font('freesansbold.ttf', 35)
        text = font.render('{}%'.format(
            "Noise Volume"), True, (0,0,0))
        textRect = text.get_rect()
        textRect[0] = x_start + (self.graphSize[1] / 2) - (textRect[2]/2)
        textRect[1] = y_origin - (textRect[3]) - 45
        screen.blit(text, textRect)
        

    # Spawn drag points for speaker and noise audio
    def CreateDragPoints(self):
        self.SpeakerPoints = self.SpawnDragObjects(
            self.graphSize, self.SpeakerGraphOrigin, self.SpeakerPointsPos, self.MinPeople, self.MaxPeople)
        self.NoisePoints = self.SpawnDragObjects(
            self.graphSize, self.NoiseGraphOrigin, self.NoisePointsPos, self.CrowdednessMin, self.CrowdednessMax)

    # For debugging, read the pos of the mouse in the screen
    def TrackMousePos(self, event):
        if event.type == MOUSEMOTION:
            print(event.pos)

    # Create a plot in matlab, and convert it into an image
    def CreateCurve(self, set_points, min, max, plot_size=[400, 200]):
        """ 
        setpoints refers to the output of the dragable squares used for curve fitting.
        These must be in the format: [[x1,y1],[x2,y2],etc]
        """
        
        # Fit curve to setpoints, and take as Ydata
        setpoint_x = []
        setpoint_y = []
        for data_pos in set_points:
            setpoint_x.append(data_pos[0])
            setpoint_y.append(data_pos[1])

        # Interpolate
        trf = interp1d(setpoint_x, setpoint_y, 'cubic')

        # Create new plot using our TRF, in our min and max range
        interp_y_data = []
        step_size = 0.1
        if max < 2:
            step_size = 0.01
        x_data = np.round(np.arange(min, max +
                                    step_size, step_size), 3)
        
        # Create interpolated data to smoothen the curve
        for x in x_data:
            # If x == setpoint x, then just use setpoint Y
            if x in setpoint_x:
                setpoint_index = setpoint_x.index(x)
                interp_y_data.append(np.array(set_points[setpoint_index][1]))
                print("using setpoint instead of interpolation")
            else:
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
        x_step = 1
        if max < 2:
            x_step = 0.1
        ax.set_xticks(np.arange(min, max + x_step, x_step))
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
        self.OpenFigures.append(fig)
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
        self.SpeakerVolume = float(self.SpeakerTRF(newNum))
        self.NumberOnScreen = newNum
        self.channel1.set_volume(self.SpeakerVolume/100)
        # update audio

    def OnCrowdednessChange(self, newCrowdedness):
        if newCrowdedness < self.CrowdednessMin:
            newCrowdedness = self.CrowdednessMin
        elif newCrowdedness > self.CrowdednessMax:
            newCrowdedness = self.CrowdednessMax
        self.NoiseVolume = float(self.NoiseTRF(newCrowdedness))
        self.Crowdedness = newCrowdedness
        self.channel2.set_volume(self.NoiseVolume/100)

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
        story = pygame.mixer.Sound(self.AudioPath + self.SpeechFileName)
        noise = pygame.mixer.Sound(self.AudioPath + self.NoiseFileName)
        self.channel1.play(story, loops=-1)
        self.channel2.play(noise, loops=-1)

    # Do some stuff on exit
    def OnExit(self):
        print("Now closing program")
        #self.SaveData()

    # Main method
    def Main(self):

        # Initialise
        self.StartCameraStream()
        self.InitScreen()
        self.CreateDragPoints()
        numberPeopleText = InputBox(575, 100, 50, 32)
        crowdednessText = InputBox(575, 400, 50, 32)
        useCameraToggle = Checkbox(self.Screen, 700, 100)
        self.StartAudio()  # Call on first loop to get %s to defaults
        numberOnScreen = self.NumberOnScreen
        crowdedness = self.Crowdedness
        save_button = Button(700, 500, 50, 200, "Save Curves")

        # Workaround: Forces our first loop to calculate plots
        self.SpeakerPointsPos, self.NoisePointsPos = [], []

        # Main loop
        while self.Running:
            # Handle Events
            events = pygame.event.get()
            for event in events:
                # self.TrackMousePos(event)
                self.CheckQuit(event)
                # Handle manual input for num of people
                numberPeopleText.handle_event(event)
                if numberPeopleText.returnInput != None:
                    numberOnScreen = int(float(numberPeopleText.returnInput)) #float -> Int, to handle decimals
                # Handle manual input for crowdedness
                crowdednessText.handle_event(event)
                if crowdednessText.returnInput != None:
                    crowdedness = float(crowdednessText.returnInput)

                # Allow drag points to be dragged
                speakerPointsPos = []
                for dragPoint in self.SpeakerPoints:
                    dragPoint.handle_event(event)
                    speakerPointsPos.append(dragPoint.GetPercentageHeight())
                noisePointsPos = []
                for dragPoint in self.NoisePoints:
                    dragPoint.handle_event(event)
                    noisePointsPos.append(dragPoint.GetPercentageHeight())

                # Let user activate camera with button 
                useCameraToggle.update_checkbox(event)
                if useCameraToggle.is_checked():
                    self.UseStream = True
                else:
                    self.UseStream = False
                
                # Let user save graphs if they like them
                save_graph = save_button.handle_event(event)
                if save_graph:
                    self.SaveData()

            # Get num of ppl and crowdedness from camera stream
            if self.UseStream:
                numberOnScreen, crowdedness = self.ProcessFrame()

            # If number on screen changed, change volume of speech
            if self.NumberOnScreen != numberOnScreen:
                self.OnNumPeopleChange(numberOnScreen)
            # If average crowdedness changed, change volume of noise
            if self.Crowdedness != crowdedness:
                self.OnCrowdednessChange(crowdedness)

            # Draw curve for speech, if setpoints change
            if self.SpeakerPointsPos != speakerPointsPos:
                newSetPoints = []
                for object in self.SpeakerPoints:
                    newSetPoints.append(object.GetGraphPos())
                print("Creating speech curve")
                speaker_volume_curve, self.SpeakerTRF = self.CreateCurve(
                    newSetPoints, self.MinPeople, self.MaxPeople, plot_size=[500, 250])
                self.SpeakerPointsPos = speakerPointsPos
            # Draw curve for noise, if setpoints change
            if self.NoisePointsPos != noisePointsPos:
                newSetPoints = []
                print("Creating noise curve")
                for object in self.NoisePoints:
                    newSetPoints.append(object.GetGraphPos())
                noise_volume_curve, self.NoiseTRF = self.CreateCurve(
                    newSetPoints, self.CrowdednessMin, self.CrowdednessMax, plot_size=[500, 250])
                self.NoisePointsPos = noisePointsPos

            # Draw on screen elements
            self.DrawBackground()
            self.DrawPlot(speaker_volume_curve, (25, 75))
            self.DrawPlot(noise_volume_curve, (25, 400))
            self.DrawGraphStats(self.Screen)
            numberPeopleText.draw(self.Screen)
            crowdednessText.draw(self.Screen)
            for dragPoint in self.SpeakerPoints:
                dragPoint.Render(self.Screen)
            for dragPoint in self.NoisePoints:
                dragPoint.Render(self.Screen)
            useCameraToggle.render_checkbox()
            save_button.Render(self.Screen)

            # Update display
            pygame.display.update()

            # Lock program to fps
            self.Clock.tick(self.FPS)


if __name__ == '__main__':
    print("Starting. This may take a while...")
    dementiaSim = DementiaSimulator()
    dementiaSim.Main()
