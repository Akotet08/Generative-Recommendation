"""
The Dataset is the Amazon Review Dataset for Beauty, Sports and Outdoors, and Toys and Games categories
"""

import json
import os
import torch
import numpy as np

class AmazonDataset:
    """Dataset class for Amazon Review Dataset for Beauty, Sports and Outdoors, and Toys and Games 
    categories. This class loads the preprocessed json with {[user-id]: [item-ids sorted by timestamp]} """

    def __init__(self, dataset_name):
        """Initialize the dataset by loading the data from the specified directory.

        Args:
            dataset_name (str): The name of the dataset to load.
        """
        self.dataset_name = dataset_name
        self.data = self.load_data()
    
    def load_data(self):
        with open(f"data/{self.dataset_name}_user_item_dict.json", "r") as f:
            return json.load(f)
    
    def __len__(self):
        """Return the number of users in the dataset."""
        return len(self.data)
    
    def __getitem__(self, user_id):
        """Return the list of item ids for a given user id.

        Args:
            user_id (str): The user id to retrieve the item ids for.

        Returns:
            list: The list of item ids for the given user id.
        """
        return self.data.get(user_id, [])