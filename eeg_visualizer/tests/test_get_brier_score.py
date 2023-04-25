import unittest
import numpy as np
from survival_metrics.metrics import get_brier_score


class TestGetBrierScore(unittest.TestCase):
    """
    unit test class for creating working folder function
    """

    def test_main(self):
        """
        the main test scenario
        :return: None
        """
        # test inputs
        # case 1 : all
        prediction = np.array([0.2, 0.4, 0.9, 1.0])
        death = np.array([1])
        survival_time = np.array([2])
        time = np.array([10])
        brier_score = survival_metrics.get_brier_score(prediction=prediction,
                                                       survival_time=survival_time,
                                                       death=death,
                                                       time=time)
        self.assertAlmostEqual(1.0, 1.0)


if __name__ == '__main__':
    unittest.main(TestGetBrierScore())
