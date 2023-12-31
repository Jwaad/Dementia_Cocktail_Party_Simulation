# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:49 2023

@author: Jwaad

Program to simulate what its like to be in group setting for some dementia sufferers.
Program mixes volumes of 2 seperate audio tracks based on the position of the crowd
Noise volume changes with how close the crowd is to the speaker (on average)
Speech volume changes based on the amount of people in the room.

To Do:
Fix padding on plots
"""

import pickle
import pygame
import time
from pygame.locals import *
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
from scipy.interpolate import interp1d
import numpy as np
import cv2 as cv
from GameElementsLib import DragPoint
from GameElementsLib import InputBox
from GameElementsLib import VideoCapture as VC
from GameElementsLib import Button
from pathlib import Path
import os
from ctypes import windll
import cython
import remove_background

class DementiaSimulator():

    def __init__(self):
        self.Running = True
        self.Crowdedness = 0
        self.CrowdednessMin = 0
        self.CrowdednessMax = 1
        self.NumberOnScreen = 0
        self.MinPeople = 0   # How many people have to be on screen, before we change the audio
        self.MaxPeople = 1  # Num people on screen, where audio will be most transformed
        self.DPI_H, self.DPI_V = self.GetDPI()
        self.SpeakerPointsPos, self.NoisePointsPos = self.LoadData()
        self.FPS = 30
        self.Clock = pygame.time.Clock()
        self.SpeakerPoints = []
        self.SpeakerVolume = 1
        self.NoisePoints = []
        self.NoiseVolume = 0
        self.SpeakerTRF = None
        self.NoiseTRF = None
        self.Stream = None
        self.ShowCamera = True
        self.CameraDown = False
        self.UseStream = False
        self.OpenFigures = []
        self.BackgroundImage = None
        #self.AudioPath = "audio/"
        self.AudioPath = os.path.join(Path().absolute(), "audio/")
        self.graphSize = [195, 387]
        self.SpeakerGraphOrigin = [88, 105]
        self.NoiseGraphOrigin = [88, 430]
        self.PreviousFaces = None
        self.LostLog = {} # record of face lost at what pos, and at what time
        
        # Variables we need to tweak
        self.CameraIndex = 0 #1
        self.down_scale_num = 0
        self.WidthFurthest = 70 # width of bounding box in pixels, when we consider person at max distance away
        self.WidthClosest = 135 # width of bounding box in pixels, when we consider person at max distance away
        self.dist_to_F = 3 # distance in m that we consider them, furthest away
        self.dist_to_C = 0.5 # distance in m that we consider them maximally close
        self.NoiseFileName = "noise.wav"
        self.SpeechFileName = "speech.wav"
        
        
    # Detect Camera and establish a video stream
    def StartCameraStream(self):
        print("Starting Camera Stream. This might take a while...")
        self.Stream = VC(self.CameraIndex)
        self.CameraDown = False
        if not self.Stream.cap.isOpened():
            print("Cannot open camera...")
            self.CameraDown = True
            return
        self.BackgroundImage = self.Stream.read()
        self.BackgroundImage = cv.flip(self.BackgroundImage, 1) # flip to match original image
        #self.BackgroundImage= self.preprocess(self.BackgroundImage, downscale_num = self.down_scale_num)
        print("Camera started successfully")
    
    
    def EndCameraSteam(self):
        print("Attempting to close current stream")
        if not self.Stream.cap.isOpened():
            print("No Steam found...")
            return
        else:
            self.Stream.EndSteam()
        print("Camera ended successfully")
    
    
    def preprocess(self, frame, downscale_num = 0 ):
        """ Grey scale and pyramid down image"""
        
        # Convert to grey scale to save processing
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # downscale as many times as specified
        for i in range (0, downscale_num):
            #print("image was downscaled")
            frame = cv.pyrDown(frame)  # Downscale for less lag

        return frame


    def count_people(self, frame):
        """ Use selected technique (face track or HOG) to track how many people are on screen and return bounding boxes"""
        person_boxes = []
        
        #self.PersonDetector = MTCNN()
        self.PersonDetector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        #self.PersonDetector = cv.HOGDescriptor() # Pedestrian detection
        #self.PersonDetector.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        
        # MTCNN
        #results = self.PersonDetector.detect_faces(frame)
        #for person in results:
        #    person_boxes.append(person["box"])
        
        # CASCASE CLASSIFIERS
        person_boxes = self.PersonDetector.detectMultiScale(frame, 1.1, 7)
        #(person_boxes, _) = self.PersonDetector.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        return person_boxes
    
    
    def compute_crowdedness(self, person_boxes):
        """ Take in bounding boxes and do take avg distance
        distance is given as a percentage of their distance between the min and max values inputted
        """
        # Compute crowdedness
        if len(person_boxes) <= 1:
            crowdedness = 0
        else:
            people_distance = []
            for box in person_boxes:
                w= box[2]
                if w < self.WidthFurthest:
                    people_distance.append(0)
                elif w > self.WidthClosest:
                    people_distance.append(1)
                else:
                    # percentage closeness within min and max. 0.5 = 50% of the distance from min - max
                    closeness = (w - self.WidthFurthest) / (self.WidthClosest - self.WidthFurthest)
                    people_distance.append(closeness)
            crowdedness = np.mean(people_distance)
        return crowdedness
    
    
    def Estimate_distance(self, person_box):
        """ Use rect of user to guess their distance in MM, using estimated min and max distance
        """
        # Compute crowdedness
        w = person_box[2]
        if w < self.WidthFurthest:
            distance = "> {}m".format(round(self.dist_to_C + self.dist_to_F, 3))
        elif w > self.WidthClosest:
            distance = "< {}m".format(round(self.dist_to_C, 3))
        else:
            # percentage closeness within min and max. 0.5 = 50% of the distance from min - max
            closeness = (w - self.WidthFurthest) / (self.WidthClosest - self.WidthFurthest)
            distance = "~ {}m".format(round(((self.dist_to_F * (1 - closeness)) + self.dist_to_C), 3))
        return distance

        
    def checkRectSimilar(self, rect1, rect2, percent_diff = 0.3):
        """ 
        Check if rect1 and rect2 are similar in x y w h by optional percentage given 
        NOTE: moving too fast would mean you're now a missing person
        """
        # Loop through x y w h, and check if each of them are similar by given percente
        for i in range(0,3):
            if not (rect2[i] - (rect2[i]*percent_diff))  <= rect1[i] <= (rect2[i] + (rect2[i]*percent_diff)):
                return False
        return True
        
        
    def check_face_lost(self, new_faces):
        """ 
        Compare new and previous faces, if we lose 1, then log when it's lost.
        """        
        # Guard statement
        if self.PreviousFaces is None or len(self.PreviousFaces) < 1:
            return
        
        # Check if face was lost
        for prev_face in self.PreviousFaces:
            # Find similar rect in new faces, else log lost face
            similar_found = False
            for new_face in new_faces:
                similar = self.checkRectSimilar(prev_face, new_face)
                if similar:
                    similar_found = True
            # If after looping through all faces, no similar rect was found -> make missing persons report
            if not similar_found:
                self.LostLog[time.time()] = prev_face
                print("Lost a face")
               
               
    def check_face_found(self, new_faces):
        """ Check if face has been reclaimed
            i.e.: Check if new faces are in the prev frame. if not. assume that face is new
            If it's new, check if it's one of the lost faces
        """  
                      
        # Guard statement
        if new_faces is None or len(new_faces) < 1:
            return
        
        # Note: not using len, incase person A appears as person B disapears
        # Check for any unaccounted for faces i.e: face that we didnt see last frame
        for new_face in new_faces:
            similar_found = False
            if self.PreviousFaces is not None:
                for prev_face in self.PreviousFaces:
                    similar = self.checkRectSimilar(new_face, prev_face)
                    if similar:
                        similar_found = True
            # Assume face is new person and check if they are in range to reclaim lost face
            if not similar_found:
                print("New Person Detected")
                for (lost_face_key, lost_face_rect) in self.LostLog.items():
                    similar = self.checkRectSimilar(new_face, lost_face_rect)
                    if similar:
                        print("Reclaimed a lost face")
                        del self.LostLog[lost_face_key]
                        break
    
        
    def cullLostFaces(self, time_til_cull = 1.5):
        """ If 5 seconds haved passed and a lost face is unclaimed, delete it"""
        cull_times = self.LostLog.copy().keys()
        for cull_time in cull_times:
            if time.time() - cull_time > time_til_cull:
                print("Culled a lost face")
                del self.LostLog[cull_time]
    
    
    def drawFaceBoxes(self, image, person_boxes, downscale_num = 0):
        # Draw bounding boxes
        #dn = downscale_num if downscale_num < 1 else 1
        if person_boxes is None:
            return image
        if len(person_boxes) < 1:
            return image
        
        dn = downscale_num + 1
        for (x, y, w, h) in person_boxes:
            
            # Estimate their distance away, for fun
            distance = self.Estimate_distance((x, y, w, h))
            
            # Scale up bounding box for original image
            xs, ys, ws, hs = [x * dn, y * dn, w * dn, h * dn] 
            colour = (0, 0, 255)
            if w > self.WidthClosest:
                colour = (255, 0, 0)
            elif w < self.WidthFurthest:
                colour = (0, 255, 0)
            cv.rectangle(image, (xs, ys), (xs + ws, ys + hs), colour, 3)
            cv.putText(image, "w = {}, h = {} d = {}".format(w, h, distance), (xs + 15, ys - 15),
                        cv.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255))
        return image
    
    
    def drawLostFaceBoxes(self, image, person_boxes, downscale_num = 0):
        # Draw bounding boxes
        # dn = downscale_num if downscale_num < 1 else 1
        if person_boxes is None:
            return image
        if len(person_boxes) < 1:
            return image
            
        dn = downscale_num + 1
        for (x, y, w, h) in person_boxes:
            
            # Scale up bounding box for original image
            xs, ys, ws, hs = [x * dn, y * dn, w * dn, h * dn] 
            colour = (255, 255, 0)
            cv.rectangle(image, (xs, ys), (xs + ws, ys + hs), colour, 3)
            cv.putText(image, "LOST FACE", (xs + 15, ys - 15),
                        cv.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255))
        return image
    
    
    # Process frame
    def ProcessFrame(self, downscale_num = 0):
        "Get latest frame, and count faces"
        if self.CameraDown:
            print("Camera was not detected. Please restart the application")
            time.sleep(0.2)
            return 0, 0
        
        # Get latest frame
        t0 = time.time()
        original_image = self.Stream.read()
        original_image = cv.flip(original_image, 1) # For displaying
        frame = original_image.copy() # For processing
        
        # Remove background image 
        t10 = time.time()
        frame = remove_background.cRemove_background(frame, self.BackgroundImage, 0.4)
        
        # Preprocess frame
        t20 = time.time()        
        frame = self.preprocess(frame, downscale_num = downscale_num)
            
        # Detect people
        t30 = time.time()
        crowdedness = 0.0
        person_boxes = self.count_people(frame)
        people_count = len(person_boxes)
                
        # Check if a face that was previously lost, has reappeared
        t32 = time.time()
        self.check_face_found(person_boxes) 
        
        # Check if between this and last frame, we lost a face
        self.check_face_lost(person_boxes)
        
        # Get rid of lingering ghousts
        self.cullLostFaces()
        self.PreviousFaces = person_boxes
        
        # Display lost faces along with tracked faces
        lost_faces = np.array(list(self.LostLog.values()))
        persons = person_boxes # default = just person faces
        # If there's lost faces and detected faces, use both
        if len(self.LostLog) > 0 and len(person_boxes) > 0:
            persons = np.vstack((lost_faces, person_boxes))
        # If there's just lost faces use only those
        elif len(self.LostLog) > 0 and len(person_boxes) < 0 :
            persons = lost_faces
            
        # Get average distance / closeness to target
        t40 = time.time()
        crowdedness = self.compute_crowdedness(persons)

        # Display stream to user
        t50 = time.time()
        if self.ShowCamera:
            # Draw bounding boxes for faces and lost faces
            original_image = self.drawFaceBoxes(original_image, person_boxes, downscale_num=downscale_num)
            original_image = self.drawLostFaceBoxes(original_image, lost_faces, downscale_num=downscale_num)
            
            # Show img
            cv.imshow('Original Image', original_image)
            cv.imshow('Processed frame', frame)
            # if click q then close stream. (keep processing though)
            if cv.waitKey(1) == ord('q'): # or cv.getWindowProperty('Camera Feed', 0) == -1:
                self.ShowCamera = False

        t60 = time.time()
        t_total = t60 - t0
        print("---\nMax FPS = {}s \nTotal time = {}s\n   Read frame: {}s \n   Remove background: {}s \n   Preprocess: {}s \n   Detect people: {}s \n   Get Lost people: {}s \n   Get crowdedness: {}s \n   Display image: {}s".format( round(1 / t_total, 0), round(t_total, 4), round(t10 - t0, 4), round(t20 - t10, 4), round(t30 - t20, 4), round(t32 - t30, 4), round(t40 - t32, 4), round(t50 - t40, 4), round(t60 - t50, 4)))
        
        return people_count, crowdedness

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
        LOGPIXELSX = 88
        LOGPIXELSY = 90
        user32 = windll.user32
        user32.SetProcessDPIAware()
        dc = user32.GetDC(0)   
        pix_per_inch = windll.gdi32.GetDeviceCaps(dc, LOGPIXELSX)
        dpi_h = windll.gdi32.GetDeviceCaps(dc, LOGPIXELSX)
        dpi_v =windll.gdi32.GetDeviceCaps(dc, LOGPIXELSY)
        user32.ReleaseDC(0, dc)
        return dpi_h, dpi_v

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

        # ---

        # Step sizes and clamp noise to graph
        if self.Crowdedness <= self.CrowdednessMin:
            x_pos = 0
        elif self.Crowdedness >= self.CrowdednessMax:
            x_pos = graphWidth
        else:
            x_pos = (self.Crowdedness * graphWidth)

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
        if event.type == pygame.MOUSEBUTTONDOWN():
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
                #print("using setpoint instead of interpolation")
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
        x_inches = plot_size[0] / self.DPI_H
        y_inches = plot_size[1] / self.DPI_V
        fig = pylab.figure(figsize=[x_inches, y_inches], dpi=self.DPI_H)
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
        fig.tight_layout()

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
        self.SpeakerVolume = float(self.SpeakerTRF(newCrowdedness))
        self.Crowdedness = newCrowdedness
        self.channel1.set_volume(self.SpeakerVolume/100)
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
        # Release the video capture object
        self.Stream.cap.release()

        # Close all OpenCV windows
        cv.destroyAllWindows()
        print("Now closing program")

    # Main method
    def Main(self):

        # Initialise
        self.StartCameraStream()
        self.InitScreen()
        self.CreateDragPoints()
        numberPeopleText = InputBox(575, 100, 50, 32)
        crowdednessText = InputBox(575, 400, 50, 32)
        self.StartAudio()  # Call on first loop to get %s to defaults
        #numberOnScreen = self.NumberOnScreen
        crowdedness = self.Crowdedness
        save_button = Button(700, 500, 50, 200, "Save Curves")
        toggle_camera_button = Button(650, 50, 100, 300, "TOGGLE CAMERA")
        button_rect = toggle_camera_button.rect
        
        # Let user change camera index
        minus_one = Button(725, 300, 25, 25, "<-")
        camera_index = InputBox(750, 300, 25, 25)
        plus_one = Button(775, 300, 25, 25, "->")
        camera_index.setText(str(self.CameraIndex))

        # Workaround: Forces our first loop to calculate plots
        self.SpeakerPointsPos, self.NoisePointsPos = [], []

        # Main loop
        while self.Running:
            t0 = time.time() # fps tracking
            # Handle Events
            events = pygame.event.get()
            for event in events:
                # self.TrackMousePos(event)
                self.CheckQuit(event)
                
                # Handle manual input for num of people
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
                
                # Change 
                plus = plus_one.handle_event(event)
                minus = minus_one.handle_event(event)
                if plus:
                    if self.CameraIndex < 5:
                        self.CameraIndex += 1
                        self.EndCameraSteam()
                        self.StartCameraStream()
                        camera_index.setText(str(self.CameraIndex))
                elif minus:
                    if self.CameraIndex > 0:
                        self.CameraIndex -= 1
                        self.EndCameraSteam()
                        self.StartCameraStream()
                        camera_index.setText(str(self.CameraIndex))
                    
                # Let user save graphs if they like them
                save_graph = save_button.handle_event(event)
                if save_graph:
                    self.SaveData()
                
                # Start and stop camera processing
                camera_pressed = toggle_camera_button.handle_event(event)
                if camera_pressed:
                    self.UseStream = not self.UseStream
                    # If stream was just turned on
                    if self.UseStream:
                        self.ShowCamera = True # Re-enable the display

            # Get num of ppl and crowdedness from camera stream
            if self.UseStream:
                numberOnScreen, crowdedness = self.ProcessFrame(downscale_num = self.down_scale_num)

            # If avg closeness changed
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
            save_button.Render(self.Screen)
            toggle_camera_button.Render(self.Screen)
            camera_index.draw(self.Screen)
            plus_one.Render(self.Screen)
            minus_one.Render(self.Screen)
            # While camera is active, draw red border around button
            if self.UseStream:
                lw = 3
                x, y, w, h = button_rect
                pygame.draw.rect(self.Screen, (255,0,0), [x - lw, y - lw, w + (lw * 2), h + (lw * 2)], lw)
            
            # Update display
            pygame.display.update()

            # Lock program to fps
            fTime = time.time() - t0
            #print("FPS =", 1/fTime)
            #self.Clock.tick(self.FPS)

#def profiler_run():
    #cProfile.run('DementiaSimulator().Main()')

if __name__ == '__main__':
    print("Starting. This may take a while...")#
    DementiaSimulator().Main()
        
    """    
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    DementiaSimulator().Main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.dump_stats('profile_data')
    """
    