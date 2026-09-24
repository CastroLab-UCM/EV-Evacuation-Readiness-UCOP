import unittest
import sumo_lib
import layer_lib
import numpy as np
import pandas as pd


class TestSumoLib(unittest.TestCase):


    # def test_build_path_and_stations_with_added_charging_station(self):
    #     od_plans = {
    #         (11, 19, 0): pd.DataFrame(
    #             {
    #                 "i": [8, 9, 10, 9, 11, 16, 18],
    #                 "j": [9, 8, 9, 16, 10, 18, 19],
    #                 "remaining energy": [22, 24, 31, 24, 49, 13, 7],
    #                 "charged energy": [3, 0, 0, 0, 0, 0, 0]
    #             }
    #         )}

    #     data = layer_lib.RunData()
    #     data.od_plans = od_plans
    #     expected_path = 'e11_10 e10_9 e9_8 8_CS8 CS8_8 e8_9 e9_16 e16_18 e18_19 '
    #     expected_edge_to_stations = [[(9, 8), '284']]

    #     self.assertEqual(
    #         sumo_lib.build_path_and_stations(data, (11, 19, 0)),
    #         (expected_path, expected_edge_to_stations)
    #     )

    # def test_build_path_and_stations_without_added_charging_station(self):
    #     od_plans = {
    #         (2, 17, 0): pd.DataFrame(
    #             {
    #                 "i": [2, 3, 4, 8, 9, 15],
    #                 "j": [3, 4, 8, 9, 15, 17],
    #                 "remaining energy": [49, 34, 26, 18, 20, 9],
    #                 "charged energy": [0, 0, 0, 3, 0, 0]
    #             }
    #         )}
    #     data = layer_lib.RunData()
    #     data.od_plans = od_plans
    #     expected_path = 'e2_3 e3_4 e4_8 8_CS8 CS8_8 e8_9 e9_15 e15_17 '
    #     expected_edge_to_stations = [[(4, 8), '284']]

    #     self.assertEqual(
    #         sumo_lib.build_path_and_stations(data, (2, 17, 0)),
    #         (expected_path, expected_edge_to_stations)
    #     )




if __name__ == "__main__":
    unittest.main()