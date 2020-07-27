import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, padding=0)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=2, kernel_size=2, padding=0)
        x = x.view(x.size(0), 6400)
        return x


class Predictor(nn.Module):
    def __init__(
        self, 
        num_classifiers_train=2, 
        num_classifiers_test=1, 
        use_init=False
    ):
        super(Predictor, self).__init__()
        self.num_classifiers_train = num_classifiers_train
        self.num_classifiers_test = num_classifiers_test
        self.init = init
        function_init = {
            'kaiming_u': nn.init.kaiming_uniform_,
            'kaiming_n': nn.init.kaiming_normal_,
            'xavier': nn.init.xavier_normal_
        }

        self.fc1 = nn.Linear(6400, 512)
        self.bn1_fc = nn.BatchNorm1d(512)

        self.fc2_mu = Parameter(torch.randn(43, 512))
        self.fc2_sigma = Parameter(torch.zeros(43, 512))
        self.fc2_bias = Parameter(torch.zeros(43))

        if use_init:
            function_init[self.init](self.fc2_mu)

    def forward(self, x, only_mu=True):

        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)

        fc2_sigma_pos = F.softplus(self.fc2_sigma - 2)
        fc2_distribution = normal.Normal(self.fc2_mu, fc2_sigma_pos)

        if self.training:
            classifiers = []
            for index in range(self.num_classifiers_train):
                fc2_w = fc2_distribution.rsample()
                classifiers.append(fc2_w)

            outputs = []
            for index in range(self.num_classifiers_train):
                out = F.linear(x, classifiers[index], self.fc2_bias)
                outputs.append(out)
            return outputs
        else:
            if only_mu:
                # Only use mu for classification
                out = F.linear(x, self.fc2_mu, self.fc2_bias)
                return [out]
            else:
                classifiers = []
                for index in range(self.num_classifiers_test):
                    fc2_w = fc2_distribution.rsample()
                    classifiers.append(fc2_w)

                outputs = []
                for index in range(self.num_classifiers_test):
                    out = F.linear(x, classifiers[index], self.fc2_bias)
                    outputs.append(out)
                return outputs
