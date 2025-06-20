#!/usr/bin/python
# -*- coding: UTF-8 -*-
from src.core import Mask
from src.core import FixationTask
from typing import List

class ImageData(object):
	def ImageData(self, path : str):
		pass

	def __init__(self):
		self.__path : str = None
		self.__iD : int = None
		self.__masks : dict = None
		self.__image : np.array = None
		self.__target : List = None
		self.__width : float = None
		self.__height : float = None
		self.mask : Mask = None
		"""# @AssociationKind Composition"""
		self.fixationTask : FixationTask = None
		"""# @AssociationKind Aggregation"""

