import numpy as np
import torch
import cv2
from raft import RAFTOpticalFlow
import argparse
from collections import OrderedDict
from tqdm import tqdm
import os


class RaftOF:
    """
    This class hadles the RAFT model to quickly compute the optical flow on videos
    """
    def __init__(self):
        """
        The raft model gets loaded onto the available device and put into evaluation mode
        """
        self.raft_of = RAFTOpticalFlow()

    def get_optical_flow(self, frame1, frame2):
        """
        :param frame1: np.ndarray(n, m, 3), first image
        :param frame2: np.ndarray(n, m, 3), second image
        :return: np.ndarray(n, m, 2), optical flow between the input images estimated using raft
        """
        return self.raft_of.calc(frame1, frame2)

    def compute_optical_flow_on_video(self, video_path, saving_path=None):
        """
        :param video_path: str, path to read video from
        :param saving_path: str or None, path to write frames and optical flow arrays to, if None, a new path is
                                         created in this directory with the video name
        :return: ((int, int), int, str), returns the image size as well as the number of frames from the video
                                         as well as the path where all frames are saved
        """
        if saving_path is None:
            video_name = video_path.split('/')[-1].split('\\')[-1].split('.')[0]
            saving_path = os.path.join('./', video_name)
            if not os.path.isdir(saving_path):
                os.mkdir(saving_path)

        print('Reading Video..')
        cap = cv2.VideoCapture(video_path)
        readable = True
        frame_list = []
        while readable:
            readable, frame = cap.read()
            
            if readable:
                frame_list.append(frame.copy())

        n = len(frame_list)

        flows = []
        prog_bar = tqdm(total = n)
        for i in range(n):
            flows.append(self.get_optical_flow(frame_list[0], frame_list[i]))
            prog_bar.update()

        for i in range(n):
            np.save(saving_path + '/img_{}'.format(i), frame_list[i])
            if i >= 1:
                flow = flows[i] - flows[i-1]
            else:
                flow = flows[i]
            np.save(saving_path + '/flow_{}'.format(i), flow)

            prog_bar.update()

        return (frame_list[0].shape[:2]), n, saving_path


