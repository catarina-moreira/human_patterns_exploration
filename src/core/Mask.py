#!/usr/bin/python
# -*- coding: UTF-8 -*-
from src.core import ImageData
from typing import List

class Mask(object):
	def Mask(self):
		pass

	def __init__(self):
		self.__imgID : int = None
		self.__maskID : int = None
		self.__path : str = None
		self.__score : float = None
		self.__prompt : np.array = None
		self.__x1 : int = None
		self.__x2 : int = None
		self.__y1 : int = None
		self.__y2 : int = None
		self.__area : float = None
		self.__logits : float = None
		self.__prompt_type : str = None
		self.__perimeter : float = None
		self.__labels : List = None
		self.__bestLabel : str = None
		self.imageData : ImageData = None

