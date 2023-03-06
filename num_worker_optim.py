import os
import argparse
import monai
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import ThreadDataLoader
from time import time


class MonaiSet(monai.data.Dataset):
	# Custom pytorch dataset, simply indexes imgs and gts
	def __init__(self, imgs, GTs):
		self.imgs = imgs
		self.GTs = GTs

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		img = torch.tensor(self.imgs[index], dtype=torch.float)
		GT = torch.tensor(self.GTs[index], dtype=torch.float)
		return img, GT


class TorchSet(torch.utils.data.Dataset):
	# Custom pytorch dataset, simply indexes imgs and gts
	def __init__(self, imgs, GTs):
		self.imgs = imgs
		self.GTs = GTs

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		img = torch.tensor(self.imgs[index], dtype=torch.float)
		GT = torch.tensor(self.GTs[index], dtype=torch.float)
		return img, GT


def worker_optimizer(config):
	print('Loading Data')
	train_img_data = np.load(config.train_img_path, allow_pickle=True)
	train_GT_data = np.load(config.train_GT_path, allow_pickle=True)
	monai_ds = MonaiSet(train_img_data, train_GT_data)
	torch_ds = TorchSet(train_img_data, train_GT_data)
	print('Dataset compiled')

	for num_workers in range(2, os.cpu_count(), 2):
		monai_loader = ThreadDataLoader(monai_ds, num_workers=num_workers, batch_size=config.batch_size, shuffle=True)
		start_monai = time()
		for epoch in range(1, 3):
			for i, data in enumerate(monai_loader, 0):
				pass
		end_monai = time()
		print("Monai finish with:{} second, num_workers={}".format(end_monai - start_monai, num_workers))

	for num_workers in range(2, os.cpu_count(), 2):
		torch_loader = DataLoader(torch_ds, num_workers=num_workers, batch_size=config.batch_size, shuffle=True)
		start_torch = time()
		for epoch in range(1, 3):
			for i, data in enumerate(torch_loader, 0):
				pass
		end_torch = time()
		print("Torch finish with:{} second, num_workers={}".format(end_torch - start_torch, num_workers))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Training hyper-parameters
	parser.add_argument('--batch_size', type=int, default=1)

	# Paths
	parser.add_argument('--train_img_path', type=str, default='./tiny_data/train_img_data.npy')
	parser.add_argument('--train_GT_path', type=str, default='./tiny_data/train_GT_data.npy')

	config = parser.parse_args()
	worker_optimizer(config)
