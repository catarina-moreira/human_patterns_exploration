#!/usr/bin/python
# -*- coding: UTF-8 -*-
from src.core import FixationTask
from typing import List

class PyGazeAnalyzer(object):
	def PyGazeAnalyzer(self, fixations : FixationTask):
		pass

	def __init__(self):
		self.__fixations : FixationTask = None
		self.__screen_dist_cm : float = None
		self.__angle_width_deg : float = None
		self.__angle_height_deg : float = None
		self.__refresh_rate : float = None
		self.__sampling_rate_hz : float = None
		self.__nominal_monitor_inch : float = None
		self.__target_dpi : float = None

