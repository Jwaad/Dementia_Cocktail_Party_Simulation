# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:00 2023

@author: Jwaad

Contained within is the class that holds all the useful methods for the creation of the app, to control the dementia simulation.
Run to "Main.py" To actually run the program
"""
import pygame as pg
import threading
import cv2
import time
import pygame


class Button:
    """Gotten online: Grepper answer https://www.google.com/search?q=pygame+button&rlz=1C1ONGR_en-GBGB993GB993&oq=pygame+button&aqs=chrome..69i57j0i512l9.1554j0j4&sourceid=chrome&ie=UTF-8    
    """
    def __init__(self, x, y, height, width, text, border = 1, curve = 0, textColour = (255,255,255), colour = (50,50,0), font_size = 12):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.colour = colour
        self.border = border
        self.curve = curve
        self.text = text
        self.textColour = textColour
        self.font = pygame.font.Font('freesansbold.ttf', font_size)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        

    def Render(self, screen):
        pygame.draw.rect(screen, self.colour, self.rect, self.border, self.curve)
        if self.text != "":
            self.drawText(screen)
        pygame.display.flip()

        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False
    
    def drawText(self, screen):
        text_surf = self.font.render(self.text, True, self.textColour)
        text_rect = text_surf.get_rect(center=(self.x+self.width//2, self.y+self.height//2))
        screen.blit(text_surf, text_rect)
    

class DragPoint():

    def __init__(self, x_position, y_percentage, clampRange, xGraphPos, size= [10,10]):
        """Creates a dragable. 10x10 red square, 
        that returns it's position in regards to 
        the graph it's sort of attached to.

        Args:
            x_position (Float): X pos in pixels
            y_percentage (Float): percetage between clamp min and max 
            clampRange (List / Tuple): Min and Max Y values, the square is allowed to move. X is disabled
            xGraphPos (Float / Int): The X position in regards to the graph. not it's position on screen
        """
        self.ID = time.time
        self.ClampRange = clampRange
        self.Size = size
        self.x = xGraphPos  # It's X value in respect to the curve
        y_position = self.ConvertPercentToPosition(y_percentage)
        self.Rect = pygame.rect.Rect(
            (x_position - (self.Size[0]/2)), (y_position - (self.Size[1]/2)), self.Size[0], self.Size[1])
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
            elif self.BeingDragged & (mouse_y >= self.ClampRange[0]):
                self.Rect.y = self.ClampRange[1]
            elif self.BeingDragged & (mouse_y <= self.ClampRange[1]):
                self.Rect.y = self.ClampRange[0]

    # Render this object
    def Render(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.Rect)

    # Return the % of this objects position, between its min and max clamp. E.G. 50 / 100 = 0.5
    def GetPercentageHeight(self):
        clamp_length = self.ClampRange[1] - self.ClampRange[0]
        percentHeight = (1 - ((self.Rect.centery - self.ClampRange[0]) / clamp_length) ) * 100 # this percentage is flipped cause pygame y0 = top left
        if percentHeight > 100:
            percentHeight = 100
        elif percentHeight < 0 :
            percentHeight = 0
        return percentHeight

    def ConvertPercentToPosition(self, percentage):
        if percentage > 100:
            percentage = 100
        elif percentage < 0 :
            percentage = 0
        clamp_length = self.ClampRange[1] - self.ClampRange[0]
        y_pos = (self.ClampRange[0] + ( (1 - (percentage / 100)) * clamp_length))# + (self.Size[1] / 2)
        return y_pos
    
    # Return X and Y coordinates in respect to it's graph placement
    def GetGraphPos(self):
        x = self.x
        y = self.GetPercentageHeight()
        return [x, y]


class InputBox:

    def __init__(self, x, y, w, h, text=''):
        """THIS CLASS WAS FOUND ONLINE,(I EDITED IT QUITE A BIT)
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
                allowed_chars = ["0", "1", "2", "3",
                                 "4", "5", "6", "7", "8", "9", "."]
                if event.key == pygame.K_RETURN:
                    # Check all chars are letters not numbers
                    rectified_text = ""
                    for char in self.text:
                        if char in allowed_chars:
                            rectified_text += char
                    self.text = rectified_text
                    # Handle sending an empty string
                    self.returnInput = self.text
                    if self.text == "":
                        self.returnInput = None
                    # Reset text for next input
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


# bufferless VideoCapture
class VideoCapture:
    """
    THIS CLASS WAS FOUND ONLINE, I DIDNT WRITE IT.
    SOURCE: https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    """

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        with self.lock:
            _, frame = self.cap.retrieve()
        return frame


pg.init()
class Checkbox:
    """
    THIS CLASS WAS FOUND ONLINE, I DIDNT WRITE IT.
    SOURCE: https://stackoverflow.com/questions/38551168/radio-button-in-pygame
    """

    def __init__(self, surface, x, y, color=(230, 230, 230), caption="", outline_color=(0, 0, 0),
                 check_color=(0, 0, 0), font_size=22, font_color=(0, 0, 0), text_offset=(28, 1)):
        self.surface = surface
        self.x = x
        self.y = y
        self.color = color
        self.caption = caption
        self.oc = outline_color
        self.cc = check_color
        self.fs = font_size
        self.fc = font_color
        self.to = text_offset
        # checkbox object
        self.checkbox_obj = pg.Rect(self.x, self.y, 12, 12)
        self.checkbox_outline = self.checkbox_obj.copy()
        # variables to test the different states of the checkbox
        self.checked = False
        self.active = False
        self.unchecked = True
        self.click = False

    def _draw_button_text(self):
        self.font = pg.font.Font(None, self.fs)
        self.font_surf = self.font.render(self.caption, True, self.fc)
        w, h = self.font.size(self.caption)
        self.font_pos = (self.x + 12 / 2 - w / 2 +
                         self.to[0], self.y + 12 / 2 - h / 2 + self.to[1])
        self.surface.blit(self.font_surf, self.font_pos)

    def render_checkbox(self):
        if self.checked:
            pg.draw.rect(self.surface, self.color, self.checkbox_obj)
            pg.draw.rect(self.surface, self.oc, self.checkbox_outline, 1)
            pg.draw.circle(self.surface, self.cc, (self.x + 6, self.y + 6), 4)

        elif self.unchecked:
            pg.draw.rect(self.surface, self.color, self.checkbox_obj)
            pg.draw.rect(self.surface, self.oc, self.checkbox_outline, 1)
        self._draw_button_text()

    def _update(self, event_object):
        x, y = event_object.pos
        # self.x, self.y, 12, 12
        px, py, w, h = self.checkbox_obj  # getting check box dimensions
        if px < x < px + w and px < x < px + w:
            self.active = True
        else:
            self.active = False

    def _mouse_up(self):
        if self.active and not self.checked and self.click:
            self.checked = True
        elif self.checked:
            self.checked = False
            self.unchecked = True

        if self.click is True and self.active is False:
            if self.checked:
                self.checked = True
            if self.unchecked:
                self.unchecked = True
            self.active = False

    def update_checkbox(self, event_object):
        if event_object.type == pg.MOUSEBUTTONDOWN:
            self.click = True
            # self._mouse_down()
        if event_object.type == pg.MOUSEBUTTONUP:
            self._mouse_up()
        if event_object.type == pg.MOUSEMOTION:
            self._update(event_object)

    def is_checked(self):
        if self.checked is True:
            return True
        else:
            return False

    def is_unchecked(self):
        if self.checked is False:
            return True
        else:
            return False


