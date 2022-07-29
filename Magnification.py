import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
import os

from OF import RaftOF
from FrameWarper import FrameWarper
from sparseDecomposition import sparsesparsePCA


def visualize_flow(flow):
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    norm = np.log(np.linalg.norm(flow, axis=2) + 1)
    max_norm = np.max(norm)
    max_norm = 1 if max_norm == 0 else max_norm
    hsv_img = np.ones((*flow.shape[:2], 3), dtype=np.uint8) * 255
    hsv_img[..., 0] = ((angle + np.pi) * 180 / (2 * np.pi)).astype(np.uint8)
    hsv_img[..., 1] = (norm * 255 / max_norm).astype(np.uint8)

    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb_img


class MotionMag:
    """
    Object implementing the whole presented algorithm
    """
    def __init__(self, num_components, lamb, eta, lower_threshold, upper_threshold, alpha):
        """
        :param num_components: int, number of components to use in the decomposition
        :param lamb: float, l1 regularization parameter
        :param eta: float, size of the l21 ball to use as a constraint
        :param lower_threshold: float, lower threshold for a motion to be classified as a micro-motion
        :param upper_threshold: float, upper threshold for a motion to be classified as a micro-motion
        :param alpha: float, magnification factor
        """
        self.OF_module = RaftOF()
        self.sparse_decomp = sparsesparsePCA(num_components, alpha=lamb, eta=eta)

        self.lower_theshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.alpha = alpha

    def is_micro_movement(self, code):
        """
        :param code: np.ndarray(k, 2), array containing d_x(t), d_y(t) for one component
        :return: bool, a boolean indicating of this component is likely to correspond to a micro-movement
        """
        code_mag = np.max(np.linalg.norm(code, axis=1))
        return code_mag > self.lower_theshold and code_mag < self.upper_threshold

    def load_flow_and_image(self, loading_path, ind):
        """
        :param loading_path: str, path to load image and corresponding optical flow from
        :param ind: int, index of the image and flow in the video sequence which should be loaded
        :return: np.ndarray(n, m, 3), np.ndarray(n, m, 2), loaded frame and flow field
        """
        img = np.load(os.path.join(loading_path, 'img_{}.npy'.format(ind)))[..., ::-1]
        flow = np.load(os.path.join(loading_path, 'flow_{}.npy'.format(ind)))
        return img, flow

    def normalize_code_and_components(self, code, components):
        """
        :param code: np.ndarray(T, 2, num_components), sparse matrix containing for each component the (d_x(t), d_y(t))
        :param components: np.ndarray(num_components, n, m), sparse images containing for each component
                                                             the sptial motion extent
        :return: np.ndarray(2*T, num_components),  np.ndarray(num_components, n, m), renormalized code and components

        This function computes a normalization constant such that the entries in code containing the (d_x(t), d_y(t))
        actually reflect the motion magnitude observed in the video.
        """
        m = np.max(np.abs(components), axis=(1, 2))
        m = np.where(m > 0, m, 1)
        return code * m[None, None, :], components / m[:, None, None]

    def compute_micro_movement_mag(self, video_path, output_path, component_saving_path=None):
        """
        :param video_path: str, path to load the video from
        :param output_path: str, path to save the magnified video to

        This method implements our whole pipeline. This starts by computing the optical flow and loading all frames
        and flows. We then use our decomposition to decompose the flow into different components. Only
        the components corresponding to micro-movements are amplified and the resulting video is saved
        """
        (n, m), num_frames, saving_path = self.OF_module.compute_optical_flow_on_video(video_path)

        tmp = [self.load_flow_and_image(saving_path, i) for i in range(num_frames)]
        frames = np.array([x[0] for x in tmp])
        flows = np.array([x[1] for x in tmp])
        flows = gaussian_filter1d(flows, 2, axis=0)

        code = self.sparse_decomp.fit_transform(flows.transpose(0, 3, 1, 2).reshape(flows.shape[0] * 2, -1))
        num_used_components = code.shape[1]
        code = code.reshape(-1, 2, num_used_components)
        components = self.sparse_decomp.get_components().reshape(num_used_components, *flows.shape[1:3])
        code, components = self.normalize_code_and_components(code, components)

        if component_saving_path is not None:
            np.save(os.path.join(component_saving_path, 'components.npy'), components)
            np.save(os.path.join(component_saving_path, 'code.npy'), code)

        micro_exp_comps = np.array([self.is_micro_movement(code[:, :, i]) for i in range(num_used_components)])
        micro_exp_displacement = np.cumsum(np.einsum('ijk,klm->ijlm', code[:, :, micro_exp_comps],
                                                     components[micro_exp_comps, :, :]), axis=0).astype(np.float64)
        micro_exp_displacement = (1 + self.alpha) * micro_exp_displacement.transpose(0, 2, 3, 1)[:, :, :, ::-1]
        xx, yy = np.indices((n, m), dtype=np.float64)
        micro_exp_displacement[:, :, :, 0] += xx[None, :, :].astype(np.float64)
        micro_exp_displacement[:, :, :, 1] += yy[None, :, :].astype(np.float64)

        frame_warper = FrameWarper((n, m))
        warped_frames = []
        for k, frame in enumerate(frames):
            warped_frame = frame_warper.warp_image(frame.astype(np.float64) / 256, micro_exp_displacement[k],
                                                   np.linalg.norm(micro_exp_displacement[k], axis=2))
            warped_frames.append(warped_frame)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 31.0, (m, n))
        for warped_frame in warped_frames:
            out.write(warped_frame[:, :, ::-1])
        out.release()





