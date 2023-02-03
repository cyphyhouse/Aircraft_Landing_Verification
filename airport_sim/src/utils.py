"""!@brief Utility file containing a test light array.

@details Constructs a light array fitted for the Aarhus airport.

@file utils.py Utility file for the airport_sim package.

@author Martin Schuck

@date 19.09.2020
"""

import numpy as np


positions = [((-1169, 80), (1446, -525), 100), ((-1163, 143), (1448, -464), 100),  # Big runway. 200 lights.
             ((1542, -253), (-1191, 384), 100), ((1547, -222), (-1190, 415), 100),  # Smaller runway. 200 lights.
             ((1442, -440), (1505, -269), 25), ((1463, -461), (1535, -280), 25),  # Small runway connector 1. 50 lights.
             ((-1207, 157), (-1188, 376), 25), ((-1177, 148), (-1161, 367), 25),  # Small runway connector 2. 50 lights.
             ((-647, 25), (-609, 241), 25), ((-613, 16), (-578, 236), 25),  # Small runway connector 3. 50 lights.
             ((725, -286), (757, -78), 25), ((767, -301), (796, -84), 25),  # Small runway connector 4. 50 lights.
             ((768, -34), (799, 176), 25), ((796, -45), (826, 182), 25),  # Small runway connector 5. 50 lights.
             ((297, 72), (429, 270), 25), ((336, 65), (467, 270), 25)]  # Small runway connector 6. 50 lights.
line_array = list()
for position in positions:
    line_array.append(np.linspace(position[0], position[1], position[2]))


def generate_test_light_array():
    """!@brief Generates a test light array.

    @details Test light array is fitted to the Aarhus airport simulation. Light colors are evenly distributed.

    @return The test light array.
    """
    light_array = np.zeros((700, 3))
    idx = 0
    color = 0
    for line in line_array:
        for pos in line:
            light_array[idx][0:2] = pos
            light_array[idx][2] = color % 3
            color += 1
            idx += 1
        color = 0
    return light_array


test_light_array = generate_test_light_array()
