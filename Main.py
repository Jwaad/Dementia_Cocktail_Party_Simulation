# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:10:49 2023

@author: Jwaad
"""

import pygame


class DementiaSimulator():

    def __init__(self):
        self.NumberOfPeople = 0
        self.Running = True

    def InitScreen(self):
        self.BackgroundColour = (55, 55, 55)
        (width, height) = (800, 800)
        self.Screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Dementia Simulator')
        self.Screen.fill(self.BackgroundColour)

    def DrawScreen(self):
        while self.Running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.Running = False

    def Main(self):
        self.InitScreen()
        self.DrawScreen()


if __name__ == '__main__':
    dementiaSim = DementiaSimulator()
    dementiaSim.Main()
