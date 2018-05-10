import math
import torch
import observations
import numpy as np
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from torchvision.models.resnet import BasicBlock

class ResNet(nn.Module):
	def __init__(self, flags, num_classes):
		# Modified from pytorch/vision
		self.inplanes = 64

		self.conv_dropout = flags.conv_dropout
		self.fc_dropout = flags.fc_dropout

		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = self._make_layer(BasicBlock, 64, flags.layers[0], stride=1)
		self.layer2 = self._make_layer(BasicBlock, 128, flags.layers[1], stride=2)
		self.layer3 = self._make_layer(BasicBlock, 256, flags.layers[2], stride=2)
		self.layer4 = self._make_layer(BasicBlock, 512, flags.layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(4, stride=1)
		self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		# x = self.maxpool(x)

		x = self.layer1(x)
		x = nn.functional.dropout(x, p=self.conv_dropout)
		x = self.layer2(x)
		x = nn.functional.dropout(x, p=self.conv_dropout)
		x = self.layer3(x)
		x = nn.functional.dropout(x, p=self.conv_dropout)
		x = self.layer4(x)

		x = nn.functional.dropout(x, p=self.fc_dropout)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return nn.functional.softmax(x, dim=1)


class Cifar10(object):
	def __init__(self):
		self.use_cuda = torch.cuda.is_available()

	def __call__(self, flags):
		(train_X, train_Y), (test_X, test_Y) = observations.cifar10('data/')

		train = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y))
		test = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_Y))

		train_loader = DataLoader(train, batch_size=flags.batch_size, shuffle=True, pin_memory=self.use_cuda)

		model = ResNet(flags, np.unique(train_Y).size)

		if self.use_cuda:
			model = model.cuda()

		optimizer = SGD(model.parameters(), momentum=float(flags.momentum), lr=float(flags.lr))
		for epoch in range(50):
			print("Epoch {}".format(epoch))

			running_avg_loss = 0.0
			running_avg_weight = 0.01

			for X, Y in train_loader:
				X = Variable(X)
				Y = Variable(Y.long())
				if self.use_cuda:
					X = X.cuda()
					Y = Y.cuda()

				prediction = model(X)

				assert Y.size(0) == prediction.size(0)

				loss = nn.functional.cross_entropy(prediction, Y)

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				running_avg_loss = running_avg_loss * (1 - running_avg_weight) + (
					running_avg_weight * loss.data.cpu().numpy())

			print("Loss: ", running_avg_loss)

		return self._test_accuracy(model, test)

	def _test_accuracy(self, model, test):
		model.train(False)
		test_loader = DataLoader(test, batch_size=512, shuffle=True, pin_memory=self.use_cuda)
		correct = 0
		for X, Y in test_loader:
			X = Variable(X)
			if self.use_cuda:
				X = X.cuda()
				Y = Y.cuda()

			prediction = model(X).max(dim=1)[1]
			correct += (prediction.data == Y.long()).sum()

		model.train(True)
		return (correct / Y.numel())

	def optimal_input(self):
		return 0, 0

