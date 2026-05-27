import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from constants import Constants as const

from dataloader.feature_io import find_segment_feature_npz, load_segment_features_from_npz


class CaptainCookSubStepDataset(Dataset):

    def __init__(self, config, phase, split):
        self._config = config
        self._backbone = self._config.backbone
        self._phase = phase
        self._split = split

        if self._split is None:
            self._split = "recordings"

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"
        self._features_directory = self._config.segment_features_directory

        split_ids_file = f"{self._split}_data_split_combined.json"
        with open(f'annotations/data_splits/{split_ids_file}', 'r') as file:
            split_ids_json = json.load(file)

        if self._phase == 'train':
            phase_ids = split_ids_json['train'] + split_ids_json['val']
        else:
            phase_ids = split_ids_json[self._phase]

        with open('annotations/annotation_json/step_annotations.json', 'r') as f:
            self._annotations = json.load(f)

        self._sub_step_dict = {}
        if self._split == const.STEP_SPLIT:
            self._build_sub_step_dict_from_step_keys(phase_ids)
        else:
            self._build_sub_step_dict_from_recording_ids(phase_ids)

    @staticmethod
    def _parse_step_split_key(step_key):
        recording_id, step_id_str = step_key.rsplit('_', 1)
        return recording_id, int(step_id_str)

    def _append_sub_steps_for_step(self, sub_step_id, recording_id, step):
        if step['start_time'] < 0 or step['end_time'] < 0:
            return sub_step_id

        start_time = math.floor(step['start_time'])
        end_time = math.floor(step['end_time'])
        for sub_step_time in range(start_time, end_time):
            self._sub_step_dict[sub_step_id] = (
                recording_id, (sub_step_time, sub_step_time + 1), step['has_errors'])
            sub_step_id += 1
        return sub_step_id

    def _build_sub_step_dict_from_step_keys(self, step_keys):
        sub_step_id = 0
        for step_key in step_keys:
            recording_id, step_id = self._parse_step_split_key(step_key)
            if recording_id not in self._annotations:
                continue
            step = next(
                (s for s in self._annotations[recording_id]['steps'] if s['step_id'] == step_id),
                None,
            )
            if step is None:
                continue
            sub_step_id = self._append_sub_steps_for_step(sub_step_id, recording_id, step)

    def _build_sub_step_dict_from_recording_ids(self, recording_ids):
        sub_step_id = 0
        for recording_id in recording_ids:
            if recording_id not in self._annotations:
                continue
            for step in self._annotations[recording_id]['steps']:
                sub_step_id = self._append_sub_steps_for_step(sub_step_id, recording_id, step)

    def __len__(self):
        assert len(self._sub_step_dict) > 0, "No data found in the dataset"
        return len(self._sub_step_dict)

    def __getitem__(self, idx):
        recording_id = self._sub_step_dict[idx][0]
        start_time, end_time = self._sub_step_dict[idx][1]
        has_errors = self._sub_step_dict[idx][2]
        if self._backbone in [const.OMNIVORE, const.SLOWFAST, const.PERCEPTION_ENCODER]:
            features_path = find_segment_feature_npz(
                self._features_directory,
                self._backbone,
                recording_id,
            )
            recording_features = load_segment_features_from_npz(features_path)
        elif self._backbone == const.EGOVLP:
            features_path = os.path.join(self._features_directory, "features", self._backbone, f'{recording_id}.npy')
            recording_features = np.load(features_path)  # return ndarray，no need to close(),close() get error
        else:
            raise ValueError(f"Backbone {self._backbone} not supported for sub-step dataset.") 

        sub_step_features = recording_features[start_time:end_time]
        sub_step_features = torch.from_numpy(sub_step_features).float()

        if has_errors:
            sub_step_labels = torch.ones(1, 1)
        else:
            sub_step_labels = torch.zeros(1, 1)
        return sub_step_features, sub_step_labels


def collate_fn(batch):
    # batch is a list of tuples, and each tuple is (step_features, step_labels)
    step_features, step_labels = zip(*batch)

    # Stack the step_features and step_labels
    step_features = torch.cat(step_features, dim=0)
    step_labels = torch.cat(step_labels, dim=0)

    return step_features, step_labels
