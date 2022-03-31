from Magnification import MotionMag

MM = MotionMag(9, 0.1, 5, 0.1, 0.3, 4)
MM.compute_micro_movement_mag("D:/DataSets/casme2/sub14/EP04_04f.avi", "./test_outout.avi")
