import numpy as np
import torch


def compute_rotation_error(gt, rotation):
    """
    Function to compute the angular error between predicted and ground truth rotation using tensors.

    Parameters
    ----------
    :param gt: ground truth quaternion as tensor
    :param rotation: predicted rotation as tensor

    :return: difference in orientation in degrees
    """
    q = rotation
    if q[0] < 0:
        q = -1 * q
    d = torch.abs(torch.dot(q, gt[:4]))
    d = torch.clamp(d, max=1.0)
    angle_error = 2 * torch.acos(d) * 180 / torch.pi

    return angle_error

def evaluate_pose_batch(gt, rotations, translations):
    """
    Function to evaluate predicted poses in comparison to the ground truth ones using tensors.

    Parameters
    ----------
    :param gt: ground truth pose given as tensor [N, 7] (quaternion and translation)
    :param rotations: predicted rotations as quaternion tensor [N, 4]
    :param translations: predicted translations as tensor [N, 3]

    :return: difference in rotation (angle) and translation (||gt - predicted_t||_2)
    """
    assert gt.shape[0] == rotations.shape[0], 'batch size of ground truth and prediction must be equal'
    
    angle_errors = torch.zeros(gt.shape[0], 1, device=gt.device)
    for i in range(gt.shape[0]):
        angle_errors[i] = compute_rotation_error(gt[i, :4], rotations[i])
    translation_errors = torch.norm(translations - gt[:, 4:7], dim=1)
    
    return angle_errors, translation_errors


def close_to_all_pose(samples, gt):
    """
    Function to retrieve the closest pose to the ground truth

    Parameters
    ----------
    :param samples: pose predictions
    :param gt: ground truth pose

    :return closest pose and index of it
    """

    diff_t = np.zeros([samples.shape[0], samples.shape[0]])
    for i in range(0, samples.shape[0]):
        for j in range(0, samples.shape[0]):
            diff_t[i, j] = np.linalg.norm(samples[i, 4:] - gt[j, 4:])

    dot = np.minimum(np.matmul(samples[:, :4], gt[:, :4].T), 1.0)
    diff_r = np.rad2deg(2 * np.arccos(np.abs(dot)))

    sum_r = np.sum(diff_r, axis=1)
    sum_t = np.sum(diff_t, axis=1)
    sums = (sum_r / np.max(sum_r)) + (sum_t / np.max(sum_t))
    idx = np.argmin(sums)
    return samples[idx], idx

def close_to_all_pose_split(samples, gt):
    """
    Function to retrieve the closest pose to the ground truth

    Parameters
    ----------
    :param samples: pose predictions
    :param gt: ground truth pose

    :return closest pose and index of it
    """

    diff_t = np.zeros([samples.shape[0], samples.shape[0]])
    for i in range(0, samples.shape[0]):
        for j in range(0, samples.shape[0]):
            diff_t[i, j] = np.linalg.norm(samples[i, 4:] - gt[j, 4:])

    dot = np.minimum(np.matmul(samples[:, :4], gt[:, :4].T), 1.0)
    diff_r = np.rad2deg(2 * np.arccos(np.abs(dot)))

    sum_r = np.sum(diff_r, axis=1)
    sum_t = np.sum(diff_t, axis=1)
    # sums = (sum_r / np.max(sum_r)) + (sum_t / np.max(sum_t))
    idx_r = np.argmin(sum_r)
    idx_t = np.argmin(sum_t)
    
    return samples[idx_r, :4], samples[idx_t, 4:], idx_r, idx_t

