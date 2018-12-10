import torch as t
import os
from torch.utils import data
from sidekit import FeaturesServer
import numpy as np
import torchvision.transforms as transforms

class TestSet(data.Dataset):
	def __init__(self, root, transforms = None):
		speakers_dir = os.listdir(root)
		self.speakers = len(speakers_dir)
		self.speakers_dir = np.asarray(speakers_dir)
		for i in range(len(speakers_dir)):
			speakers_dir[i] = root + speakers_dir[i]
		speech = []
		for i in speakers_dir:
			speech_dir = os.listdir(i)
			for j in range(len(speech_dir)):
				speech.append(speech_dir[j].split('.')[0])
				if j == 19:
				    break
		for i in range(len(speech)):
			speech[i] = speech[i].split('_')[0] + '/' + speech[i]
		self.speech = np.asarray(speech)
		self.trans = transforms

	def __getitem__(self, index):
		features_server = FeaturesServer(features_extractor = None,
										 feature_filename_structure = './all_feature/{}.h5',
										 sources = None,
										 dataset_list = ['fb'],
										 mask = None, 
										 feat_norm = 'cms',
										 global_cmvn = None,
										 dct_pca = False,
										 dct_pca_config = None,
										 sdc = False,
										 sdc_config = None,
										 delta = False,
										 double_delta = False,
                                         delta_filter = None,
										 context = None,
										 traps_dct_nb = None,
										 rasta = True,
										 keep_all_features = False)
		show_list = self.speech[index]
		speaker = show_list.split('/')[0]
		features, _ = features_server.load(show_list, channel = 0)
		features = features.astype(np.float32)
		ind = np.argwhere(self.speakers_dir == speaker)[0]
		label = ind.astype(np.int64)[0] #这里只要指出label所在的索引就好了，比如是第20个说话人说的，那么label就是[20]
		features = features.reshape(1, features.shape[1], features.shape[0])
		features = t.tensor(features)
		img = transforms.ToPILImage()(features)
		features = transforms.Resize((24, 400))(img)
		if self.trans:
			features = self.trans(features)
		else:
		    features = transforms.ToTensor()(features)
		return features.view(features.size()[1], features.size()[2]), label

	def __len__(self):
		return len(self.speech)
