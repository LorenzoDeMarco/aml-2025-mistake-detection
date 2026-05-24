import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from dataloader.feature_io import find_segment_npz_in_directory, load_segment_features_from_npz

from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("error")
class ErrorDataset(Dataset):
	def __init__(
			self,
			is_training,  # if in training mode
			split,  # split, a tuple/list allowing concat of subsets
			feat_folder,  # folder for features
			json_file,  # json file for annotations
			feat_stride,  # temporal stride of the feats
			num_frames,  # number of frames for each feat
			default_fps,  # default fps
			downsample_rate,  # downsample rate for feats
			max_seq_len,  # maximum sequence length during training
			trunc_thresh,  # threshold for truncate an action segment
			crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
			input_dim,  # input feat dim
			num_classes,  # number of action categories
			file_prefix,  # feature file prefix if any
			file_ext,  # feature file extension if any
			force_upsampling  # force to upsample to max_seq_len
	):
		# Initialize file-path related attributes
		self._init_file_paths(feat_folder, json_file, file_prefix, file_ext)
		# Initialize split / training mode
		self._init_split_mode(split, is_training)
		# Initialize feature metadata
		self._init_features_meta(feat_stride, num_frames, input_dim, default_fps, downsample_rate, max_seq_len, trunc_thresh, crop_ratio)
		self.label_dict = None
		# Load database and select subset
		self._init_database(json_file, num_classes)
		# Initialize dataset-specific attributes
		self._init_db_attributes(num_classes)

	def _init_file_paths(self, feat_folder, json_file, file_prefix, file_ext):
		"""
		Initialize file-path related attributes.

		Args:
			feat_folder (str): Directory containing feature files
			json_file (str): Path to the annotation JSON file
			file_prefix (str or None): Prefix for feature filenames
			file_ext (str or None): Extension for feature filenames

		Sets:
			self.feat_folder, self.file_prefix (empty string if None),
			self.file_ext, self.json_file
		"""
		# file path
		assert os.path.exists(feat_folder) and os.path.exists(json_file)
		self.feat_folder = feat_folder
		if file_prefix is not None:
			self.file_prefix = file_prefix
		else:
			self.file_prefix = ''
		self.file_ext = file_ext
		self.json_file = json_file

	def _init_split_mode(self, split, is_training):
		"""
		Initialize split and training-mode attributes.

		Args:
			split (tuple or list): Dataset split(s); multiple subsets may be concatenated
			is_training (bool): Whether the dataset is used in training mode

		Sets:
			self.split, self.is_training
		"""
		# split / training mode
		assert isinstance(split, tuple) or isinstance(split, list)
		self.split = split
		self.is_training = is_training

	def _init_features_meta(self, feat_stride, num_frames, input_dim, default_fps, downsample_rate, max_seq_len, trunc_thresh, crop_ratio):
		"""
		Initialize feature metadata attributes.

		Args:
			feat_stride (int): Temporal stride of features
			num_frames (int): Number of frames per feature vector
			input_dim (int): Input feature dimension
			default_fps (float or None): Default frames per second
			downsample_rate (int): Feature downsampling rate
			max_seq_len (int): Maximum sequence length during training
			trunc_thresh (float): Threshold for truncating action segments
			crop_ratio (tuple or None): Random crop ratio as (start, end)

		Sets:
			self.feat_stride, self.num_frames, self.input_dim, self.default_fps,
			self.downsample_rate, self.max_seq_len, self.trunc_thresh, self.crop_ratio
		"""
		# features meta info
		assert crop_ratio == None or len(crop_ratio) == 2
		self.feat_stride = feat_stride
		self.num_frames = num_frames
		self.input_dim = input_dim
		self.default_fps = default_fps
		self.downsample_rate = downsample_rate
		self.max_seq_len = max_seq_len
		self.trunc_thresh = trunc_thresh
		self.crop_ratio = crop_ratio

	def _init_database(self, json_file, num_classes):
		# load database and select the subset
		dict_db, label_dict = self._load_json_db(json_file)
		# "empty" noun categories on epic-kitchens
		assert len(label_dict) <= num_classes
		self.data_list = dict_db
		self.label_dict = label_dict

	def _init_db_attributes(self, num_classes):

		# dataset specific attributes
		empty_label_ids = self.find_empty_cls(self.label_dict, num_classes)
		self.db_attributes = {
			'dataset_name': 'error',
			'tiou_thresholds': np.linspace(0.1, 0.5, 5),
			'empty_label_ids': empty_label_ids
		}
	
	def find_empty_cls(self, label_dict, num_classes):
		# find categories with out a data sample
		if len(label_dict) == num_classes:
			return []
		empty_label_ids = []
		label_ids = [v for _, v in label_dict.items()]
		for id in range(num_classes):
			if id not in label_ids:
				empty_label_ids.append(id)
		return empty_label_ids
	
	def get_attributes(self):
		return self.db_attributes
	
	def _load_json_db(self, json_file):
		# load database and select the subset
		with open(json_file, 'r') as fid:
			json_data = json.load(fid)
		json_db = json_data['database']

		# if label_dict is not available
		if self.label_dict is None:
			label_dict = {}
			for key, value in json_db.items():
				for act in value['annotations']:
					label_dict[act['label']] = act['label_id']
		else:
			label_dict = self.label_dict
		# fill in the db (immutable afterwards)
		dict_db = tuple()
		skipped_missing_feat = []
		file_ext = self.file_ext or ".npz"
		file_prefix = self.file_prefix or ""
		for key, value in json_db.items():
			# skip the video if not in the split
			if value['subset'].lower() not in self.split:
				continue

			try:
				find_segment_npz_in_directory(
					self.feat_folder, key, file_prefix, file_ext
				)
			except FileNotFoundError:
				skipped_missing_feat.append(key)
				continue
			
			# get fps if available
			if self.default_fps is not None:
				fps = self.default_fps
			elif 'fps' in value:
				fps = value['fps']
			else:
				assert False, "Unknown video FPS."
			
			# get video duration if available
			if 'duration' in value:
				duration = value['duration']
			else:
				duration = 1e8
			
			# get annotations if available
			if ('annotations' in value) and (len(value['annotations']) > 0):
				num_acts = len(value['annotations'])
				segments = np.zeros([num_acts, 2], dtype=np.float32)
				labels = np.zeros([num_acts, ], dtype=np.int64)
				for idx, act in enumerate(value['annotations']):
					segments[idx][0] = act['segment'][0]
					segments[idx][1] = act['segment'][1]
					labels[idx] = label_dict[act['label']]
			else:
				segments = None
				labels = None
			dict_db += ({'id': key,
			             'fps': fps,
			             'duration': duration,
			             'segments': segments,
			             'labels': labels
			             },)
		if skipped_missing_feat:
			examples = ", ".join(skipped_missing_feat[:5])
			suffix = " ..." if len(skipped_missing_feat) > 5 else ""
			print(
				f"[ErrorDataset] Skipped {len(skipped_missing_feat)} videos with no "
				f"matching feature under {self.feat_folder} (e.g. {examples}{suffix})"
			)
		if len(dict_db) == 0:
			raise RuntimeError(
				f"No videos left after split/filter under {self.feat_folder}. "
				f"Check json_file, train/val split names, and feature files."
			)
		return dict_db, label_dict
	
	def __len__(self):
		return len(self.data_list)
	
	def __getitem__(self, idx):
		# directly return a (truncated) data point (so it is very fast!)
		# auto batching will be disabled in the subsequent dataloader
		# instead the model will need to decide how to batch / preporcess the data
		video_item = self.data_list[idx]
		# load features (canonical path + glob fallback for alternate stems, e.g. PE extractor)
		filename = find_segment_npz_in_directory(
			self.feat_folder,
			video_item["id"],
			self.file_prefix or "",
			self.file_ext or ".npz",
		)
		feats = load_segment_features_from_npz(filename)

		# deal with downsampling (= increased feat stride)
		feats = feats[::self.downsample_rate, :]
		feat_stride = self.feat_stride * self.downsample_rate
		feat_offset = 0.5 * self.num_frames / feat_stride
		# T x C -> C x T
		feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

		if video_item['segments'] is not None:
			segments = torch.from_numpy(
				video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
			)
			labels = torch.from_numpy(video_item['labels'])
		else:
			segments, labels = None, None
		
		# return a data dict
		data_dict = {'video_id': video_item['id'],
		             'feats': feats,  # C x T
		             'segments': segments,  # N x 2
		             'labels': labels,  # N
		             'fps': video_item['fps'],
		             'duration': video_item['duration'],
		             'feat_stride': feat_stride,
		             'feat_num_frames': self.num_frames}
		
		# truncate the features during training
		if self.is_training and (segments is not None):
			data_dict = truncate_feats(
				data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
			)
		return data_dict
