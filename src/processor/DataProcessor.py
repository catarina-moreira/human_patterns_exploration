
from typing import List

import pandas as pd

class DataProcessor:
    
	def __init__(self, path : str, condition : int = None, group : int = None):
		self.path = path
		self.data = self.load_data()
		self.condition = condition
		self.group = group

	def load_data(self):
		data = pd.read_csv(self.path)
		return data

	def process_participant_data(self):

		if not(self.group is None or self.condition is None):
			filtered_data = self.data[(self.data['Condition'] == self.condition) & (self.data['Group'] == self.group)]
		else:
			filtered_data = self.data
		return filtered_data.drop(["Condition", "Group"], axis=1)

	def get_participant_ids(self, filtered_data):
		return filtered_data['Participant_ID'].unique()










