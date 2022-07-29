from Magnification import MotionMag
from os.path import join

if __name__ == "__main__":
    MM = MotionMag(9, 0.1, 5, 0.1, 0.3, 4)
    # Put the base folder of CASME2-RAW here:
    casme_folder = "./CASME2-RAW"
    MM.compute_micro_movement_mag(join(casme_folder, "sub14/EP04_04f.avi"), "./test_output.avi")
