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

    def __init__(self, position, clampRange, xGraphPos):
        """Creates a dragable. 10x10 red square, 
        that returns it's position in regards to 
        the graph it's sort of attached to.

        Args:
            position (List / Tuple): X and Y values of top left of square
            clampRange (List / Tuple): Min and Max Y values, the square is allowed to move. X is disabled
            xGraphPos (Float / Int): The X position in regards to the graph. not it's position on screen
        """
        self.ID = time.time
        size = (10, 10)  # 10 pix by 10
        self.Rect = pygame.rect.Rect(
            (position[0] - (size[0]/2)), (position[1] - (size[1]/2)), size[0], size[1])
        self.ClampRange = clampRange
        self.x = xGraphPos  # It's X value in respect to the curve
        self.BeingDragged = False

    # Move the pos of this rect, depending on drag + clamp ranges
    def handle_event(self, event):
        # If left mouse clicked
        if event.type == pygame.MOUSEBUTTONDOWN:

            # If mouse hovering over this point
            if event.button == 1 & self.Rect.collidepoint(event.pos):
                self.BeingDragged = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.BeingDragged = False

        elif event.type == pygame.MOUSEMOTION:
            mouse_y = event.pos[1]
            # If object is being dragged, and mouse is in range
            if self.BeingDragged & (mouse_y > self.ClampRange[0]) & (mouse_y < self.ClampRange[1]):
                self.Rect.y = mouse_y
            else:
                self.BeingDragged = False

    # Render this object
    def Render(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.Rect)

    # Return the % of this objects position, between its min and max clamp. E.G. 50 / 100 = 0.5
    def GetPercentageHeight(self):
        percentHeight = ((1 - (
            self.Rect.y - self.ClampRange[0]) / (self.ClampRange[1] - self.ClampRange[0])) * 100)
        return percentHeight

    # Return X and Y coordinates in respect to it's graph placement
    def GetGraphPos(self):
        x = self.x
        y = self.GetPercentageHeight()
        return [x, y]


class InputBox:

    def __init__(self, x, y, w, h, text=''):
        """THIS CLASS WAS FOUND ONLINE, I DIDNT WRITE IT.
        SOURCE: https://stackoverflow.com/questions/46390231/how-can-i-create-a-text-input-box-with-pygame

        """
        self.COLOR_INACTIVE = pygame.Color('lightskyblue3')
        self.COLOR_ACTIVE = pygame.Color('dodgerblue2')
        self.FONT = pygame.font.Font(None, 32)
        self.rect = pygame.Rect(x, y, w, h)
        self.color = self.COLOR_INACTIVE
        self.text = text
        self.txt_surface = self.FONT.render(text, True, self.color)
        self.active = False
        self.returnInput = None

    def handle_event(self, event):
        self.returnInput = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = self.COLOR_ACTIVE if self.active else self.COLOR_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.returnInput = self.text
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = self.FONT.render(
                    self.text, True, self.color)

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)
