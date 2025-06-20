#!/usr/bin/python
# -*- coding: UTF-8 -*-
from src.core import Participant
from src.core import ImageData
from src.core import Framework
from typing import List

class FixationTask(object):
	def FixationTask(self, participant : Participant, img : ImageData, data : np.Dataframe):
		pass

	def getXcoord(self):
		pass

	def getYcoord(self):
		pass

	def getDuration(self):
		pass

	def getPartID(self):
		pass

	def getImgID(self):
		pass

	def plotScanPath(self):
		pass

	def plotAttentionMap(self):
		pass

	def plotFixations(self):
		pass

	def __init__(self):
		self.__part : Participant = None
		self.__img : ImageData = None
		self.__data : pd.Dataframe = None
		self.__group : int = None
		self.__condition : int = None
		self.experiment : Framework = None
		self.participant : Participant = None
		self.imageData : ImageData = None

