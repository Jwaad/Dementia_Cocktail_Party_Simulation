# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:00 2023

@author: Jwaad

Contained within is the class that holds all the useful methods for the creation of the app, to control the dementia simulation.
Run to "Main.py" To actually run the program
"""
import time
import pygame


class DragPoint():

    def __init__(self, position, clampRange, x):
        self.ID = time.time
        size = (10, 10)  # 10 pix by 10
        self.Rect = pygame.rect.Rect(
            position[0], position[1], size[0], size[1])
        self.ClampRange = clampRange
        self.x = x  # It's X value in respect to the curve

    # Move the pos of this rect, depending on drag + clamp ranges
    def HandleEvents(self, events):
        for event in events:
            # If left mouse clicked
            if event.type == pygame.MOUSEBUTTONDOWN & event.button == 1:
                # If mouse hovering over this point
                if self.Rect.collidepoint(event.pos):
                    OnDrag

    # Return the % of this objects position, between its min and max clamp. E.G. 50 / 100 = 0.5
    def GetPercentageHeight(self):
        percentHeight = (self.ClampRange[1] -
                         self.ClampRange[0]) / (self.Rect.y - self.ClampRange[0])
        return percentHeight

    # Return X and Y coordinates in respect to it's graph placement
    def GetXYPos(self):
        x = self.x
        y = self.GetPercentageHeight()
        return [x, y]
