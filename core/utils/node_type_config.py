"""

This contains data mapping for the node type to be expected from the network topology image and its color map
These two are combined to create node_feature matrix for the graph neural network
"""

NODE_TYPE = ['ATN', 'RTN', 'Router', 'Switch', 'HubSite']

COLOR_MAP = {'Red': [1, 0, 0, 0, 0, 0], 'Green': [0, 1, 0, 0, 0, 0], 'Blue': [0, 0, 1, 0, 0, 0],
             'Yellow': [0, 0, 0, 1, 0, 0], 'Orange': [0, 0, 0, 0, 1, 0], 'Gray': [0, 0, 0, 0, 0, 1]}

# Minimum fuzzy similarity (%) for matching a site ID.
FUZZY_PERCENTAGE = 70