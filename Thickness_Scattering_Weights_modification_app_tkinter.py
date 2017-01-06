# -*- coding: utf-8 -*-
import Tkinter as tk
from Tkinter import *
from tkFileDialog   import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk
import sys,os,io
import numpy as np 
import cv2
from KS_helper import *
import warnings
import json
from Constant_Values import *




class Thickness_Scattering_Weights_modification_app:
    
    def __init__(self, controller):
        self.master=controller.master
        self.canvas=controller.canvas
        self.mask=controller.mask

        if self.mask==None:
            print "no input mask"
        else:
            print self.mask.shape

        self.AllData=controller.AllData