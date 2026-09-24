import unittest
import layer_lib
import numpy as np
import pandas as pd
import pyomo.environ as pyo


class TestLayerLib(unittest.TestCase):

    # def test_sort_rows(self):
    #     route = pd.DataFrame({
    #         "i": [8, 9, 10, 9, 11, 16, 18],
    #         "j": [9, 8, 9, 16, 10, 18, 19],
    #         "remaining energy": [22, 24, 31, 24, 49, 13, 7],
    #         "charged energy": [3, 0, 0, 0, 0, 0, 0]
    #     })
    #     expected_route_after_sort = pd.DataFrame({
    #         "i": [11, 10, 9, 8, 9, 16, 18],
    #         "j": [10, 9, 8, 9, 16, 18, 19],
    #         "remaining energy": [49, 31, 24, 22, 24, 13, 7],
    #         "charged energy": [0, 0, 0, 3, 0, 0, 0],
    #     })
    #     print('expected_route_after_sort:', expected_route_after_sort)
    #     self.assertTrue(
    #         layer_lib.sort_route((11, 19, 0), route).equals(expected_route_after_sort)
    #     )

    def test_sort_rows2(self):
        route = pd.DataFrame({
            "i": [5, 2, 3, 4, 4, 8],
            "j": [1, 3, 4, 5, 8, 4],
            "remaining energy": [10, 49, 34, 27, 27, 18],
            "charged energy": [0, 0, 0, 0, 0, 17]
        })
        print(route)
        expected_route_after_sort = pd.DataFrame({
            "i": [2, 3, 4, 8, 4, 5],
            "j": [3, 4, 8, 4, 5, 1],
            "remaining energy": [49, 34, 27, 18, 27, 10],
            "charged energy": [0, 0, 0, 17, 0, 0],
        })
        print('expected_route_after_sort:', expected_route_after_sort)
        self.assertTrue(
            layer_lib.sort_route((2, 1, 0), route).equals(expected_route_after_sort)
        )



if __name__ == "__main__":
    unittest.main()
