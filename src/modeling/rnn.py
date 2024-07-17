import torch.nn
from torch.nn.functional import relu
from config import *
import articulate as art

class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)

class RNN(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x, h = self.rnn(relu(self.linear1(self.dropout(x))).unsqueeze(1), h)
        return self.linear2(x.squeeze(1)), h



class IMU_backbone(torch.nn.Module):
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9
        self.pose_s1 = RNN(n_imu,                         joint_set.n_leaf * 3,       256)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_imu,  joint_set.n_full * 3,       64)