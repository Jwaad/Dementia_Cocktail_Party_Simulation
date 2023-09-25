# -*- coding: utf-8 -*-

#python setup.py build_ext --inplace
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN

print("Person Tracker has been Imported")

cdef class PersonTracker:
    cdef int WidthFurthest
    cdef int width_closest
    cdef int ShowCamera # declare bools as ints
    
    def __init__(self, vc, width_furthest, width_closest):
        self.vc = vc
        
        self.WidthFurthest = width_furthest
        self.WidthClosest = width_closest
        self.ShowCamera = True
        
        # Set a technique for person detection
        self.PersonDetector = MTCNN()
        #self.PersonDetector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        #self.PersonDetector = cv.HOGDescriptor() # Pedestrian detection
        #self.PersonDetector.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
      
      
    cdef preprocess(self, frame, int downscale_num = 0 ):
        """ Take frame and pre process according to our needs"""
        
        # Convert to grey scale to save processing
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # downscale as many times as specified
        for i in range (0, downscale_num):
            #print("image was downscaled")
            frame = cv.pyrDown(frame)  # Downscale for less lag
        
        return frame


    def count_people(self, frame):
        """ Use selected technique (face track or HOG) to track how many people are on screen and return bounding boxes"""
        person_boxes = []
        results = self.PersonDetector.detect_faces(frame)
        #results = self.PersonDetector.detectMultiScale(frame, 1.1, 3)
        for person in results:
            person_boxes.append(person["box"])
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
                w = box[2]
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


    def remove_background(self, frame):
        """ Remove the background image from the current frame"""
        # todo loop through and remove background here
        #forground_mask = self.BackSub.apply(frame)
        #frame = cv.bitwise_and(frame, frame, mask= forground_mask)
        return frame
        

    # Process frame
    def ProcessFrame(self, downscale_num = 0):
        "Get latest frame, and count faces"
        
        # Get latest frame
        try:
            original_image = self.vc.read()
            original_image = cv.flip(original_image, 1)
        except Exception as e:
            print("Stream could not start. Please restart the application")
            return 0, 0
        
        # Preprocess frame
        frame = self.preprocess(original_image, downscale_num = downscale_num)
        
        # Remove background image
        frame = self.remove_background(frame)        
        
        # Detect people
        crowdedness = 0.0
        person_boxes = self.count_people(frame)
        people_count = len(person_boxes)
        
        # Get average distance / closeness to target
        crowdedness = self.compute_crowdedness(person_boxes)

        # Display stream to user
        if self.ShowCamera:
            # Draw bounding boxes
            for (x, y, w, h) in person_boxes:
                dn = downscale_num if downscale_num < 1 else 1
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # Scale up bounding box for original image
                x, y, w, h = [x * dn, y * dn, w * dn, h * dn]
                cv.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(original_image, "w = {}, h = {}".format(w, h), (x + 15, y - 15),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                
            # Show img
            cv.imshow('Processed Stream', frame)
            cv.imshow('Original Image', original_image)
            # if click q or on the X, then close stream. (keep processing though)
            if cv.waitKey(1) == ord('q'):# or cv.getWindowProperty('Camera Feed', 0) == -1:
                self.ShowCamera = False

        return people_count, crowdedness