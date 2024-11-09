""" This module provides utility functions for Pytorch models """

import torch
import yaml

class ModelUtils:
    """ ModelUtils class to provide utility functions for Pytorch models """

    def get_device(self):
        """
        Determines if a CUDA-capable GPU is available and returns the appropriate device.

        Returns:
            torch.device:
                'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'.
        """
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model_config(self, config_path):
        """
        Loads a configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: The loaded configuration.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
