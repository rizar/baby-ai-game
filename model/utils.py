from __future__ import print_function
import numpy as np
from collections import namedtuple, deque
import os
import time
import shutil
import torch


ReplayFrame = namedtuple(
    'ReplayFrame',
    ('state',
     'action_logit',
     'next_state',
     'reward',
     'value',
     'terminal'))

State = namedtuple('State', ('image', 'mission', 'advice'))


class ReplayMemory(object):

    def __init__(self, capacity, num_frame_rp=3):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.top_index = 0
        self.zero_reward_indices = []  # frame indices for zero rewards
        self.nonzero_reward_indices = []  # frame indices for non zero rewards
        self.num_frame_rp = num_frame_rp  # num of frames for reward prediction

    def push(self, frame):
        # ReplayFrame(*args)
        if (frame.terminal and len(self.memory) > 0) and (
                self.memory[-1].terminal):
                # Discard if terminal frame continues
            print("Terminal frames continued.")
            return

        frame_index = self.top_index + len(self.memory)
        was_full = self.full()

        # append frame
        self.memory.append(frame)

        # append index
        if frame_index >= self.num_frame_rp:
            if frame.reward == 0:
                self.zero_reward_indices.append(frame_index)
            else:
                self.nonzero_reward_indices.append(frame_index)

        if was_full:
            self.top_index += 1

            min_frame_index = self.top_index + self.num_frame_rp
            # Remove frame if its index is lower than min_frame_index.
            if len(self.zero_reward_indices) > 0 and \
               self.zero_reward_indices[0] < min_frame_index:
                self.zero_reward_indices.pop(0)

            if len(self.nonzero_reward_indices) > 0 and \
               self.nonzero_reward_indices[0] < min_frame_index:
                self.nonzero_reward_indices.pop(0)

    def sample(self, batch_size):
        start_index = np.random.randint(0, len(self.memory) - batch_size)
        sampled_frames = []
        for i in range(batch_size):
            frame = self.memory[start_index + i]
            sampled_frames.append(frame)
        return sampled_frames

    def sample_sequence(self, sequence_size):
        # -1 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        start_pos = np.random.randint(
            0, len(self.memory) - sequence_size - 1)

        if self.memory[start_pos].terminal:
            start_pos += 1
            # Assuming that there are no successive terminal frames.

        sampled_frames = []

        for i in range(sequence_size):
            frame = self.memory[start_pos + i]
            sampled_frames.append(frame)
            if frame.terminal:
                break

        return sampled_frames

    def skewed_sample(self, batch_size):
        """
        Sample self.num_frames_rew_pre+1 successive frames for reward prediction.
        """
        sampled_batch = []

        size = min(batch_size, len(self.nonzero_reward_indices) +
                   len(self.zero_reward_indices))

        for k in range(size):
            if np.random.randint(2) == 0:
                from_zero = True
            else:
                from_zero = False

            if len(self.zero_reward_indices) == 0:
                # zero rewards container was empty
                from_zero = False
            elif len(self.nonzero_reward_indices) == 0:
                # non zero rewards container was empty
                from_zero = True

            if from_zero:
                index = np.random.randint(len(self.zero_reward_indices))
                end_frame_index = self.zero_reward_indices[index]
            else:
                index = np.random.randint(len(self.nonzero_reward_indices))
                end_frame_index = self.nonzero_reward_indices[index]

            start_frame_index = end_frame_index - self.num_frame_rp
            raw_start_frame_index = start_frame_index - self.top_index

            sampled_frames = []

            for i in range(self.num_frame_rp + 1):
                frame = self.memory[raw_start_frame_index + i]
                sampled_frames.append(frame)

            sampled_batch.extend(sampled_frames)

        return sampled_batch

    def __len__(self):
        return(len(self.memory))

    def full(self):
        if (len(self.memory) >= self.capacity):
            return True
        return False

    def clear(self):
        self.memory.clear()
        self.top_index = 0
        self.zero_reward_indices.clear()
        self.nonzero_reward_indices.clear()


def apply_flatten_parameters(m):
    if hasattr(m, 'flatten_parameters'):
        if callable(m.flatten_parameters):
            m.flatten_parameters()


class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during
    training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using
    sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following
    args; then call that object's save() method to write parameters to disk.

    Args:
        model: model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch
        step (int): number of examples seen within the current epoch

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    def __init__(
            self, model, optimizer, epoch, step, path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir):
        """
        Saves the current model and related training parameters into
        a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in
        Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(
            experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer
                    },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from
            those stored on disk
        """
        print("Loading checkpoints from {}".format(path))
        try:
            resume_checkpoint = torch.load(
                os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        except:
            resume_checkpoint = torch.load(
                os.path.join(path, cls.TRAINER_STATE_NAME),
                map_location=lambda storage, loc: storage)
            model = torch.load(
                os.path.join(path, cls.MODEL_NAME),
                map_location=lambda storage, loc: storage)
        # make RNN parameters contiguous
        model = model.apply(apply_flatten_parameters)
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to
        the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made
        (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(
            experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
