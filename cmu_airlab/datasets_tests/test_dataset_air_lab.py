import unittest

from cmu_airlab.datasets.dataset_air_lab import AirLabClassSegBase

class TestAirLabClassSegBase(unittest.TestCase):

    def test_get_one(self):
        dataset_root = "/home/david/workspaces/cmu/intern_assignment/task-5-2-3/resources/test"
        ds = AirLabClassSegBase(root=dataset_root)
        img, lbl = ds.__getitem__(0)

        ds = AirLabClassSegBase(root=dataset_root, k_fold_val=1)
        img, lbl = ds.__getitem__(0)

        ds = AirLabClassSegBase(root=dataset_root, k_fold_val=2)
        img, lbl = ds.__getitem__(0)

        ds = AirLabClassSegBase(root=dataset_root, k_fold_val=3)
        img, lbl = ds.__getitem__(0)




if __name__ == '__main__':
    unittest.main()