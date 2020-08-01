import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x, 1).view(x.size()[0], 1, x.size()[2], x.size()[3])

        x = F.max_pool2d(
            F.relu(self.bn1(self.conv1(x))), 
            stride=2, 
            kernel_size=2, 
            dilation=(1, 1)
        )

        x = F.max_pool2d(
            F.relu(self.bn2(self.conv2(x))), 
            stride=2, 
            kernel_size=2, 
            dilation=(1, 1)
        )

        x = x.view(x.size(0), 48*4*4)
        return x


class Predictor(nn.Module):
    def __init__(
        self, 
        num_classifiers_train=2, 
        num_classifiers_test=20,
        init='kaiming_u', 
        use_init=False, 
        prob=0.5
    ):
        super(Predictor, self).__init__()
        self.num_classifiers_train = num_classifiers_train
        self.num_classifiers_test = num_classifiers_test
        self.prob = prob
        self.init = init

        function_init = {
            'kaiming_u': nn.init.kaiming_uniform_,
            'kaiming_n': nn.init.kaiming_normal_,
            'xavier': nn.init.xavier_normal_
        }

        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)

        # Use distribution in the last layer
        self.fc3_mu = Parameter(torch.randn(10, 100))
        self.fc3_sigma = Parameter(torch.zeros(10, 100))
        self.fc3_bias = Parameter(torch.zeros(10))

        if use_init:
            function_init[init](self.fc3_mu)

    def forward(self, x, only_mu=True):
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))

        # Distribution sample for the fc layer
        fc3_sigma_pos = F.softplus(self.fc3_sigma - 2)
        fc3_distribution = normal.Normal(self.fc3_mu, fc3_sigma_pos)

        if self.training:
            classifiers = []
            for index in range(self.num_classifiers_train):
                fc3_w = fc3_distribution.rsample()
                classifiers.append(fc3_w)

            outputs = []
            for index in range(self.num_classifiers_train):
                out = F.linear(x, classifiers[index], self.fc3_bias)
                outputs.append(out)

            return outputs

        else:
            if only_mu:
                # Only use mu for classification
                out = F.linear(x, self.fc3_mu, self.fc3_bias)
                return [out]
            else:
                classifiers = []
                for index in range(self.num_classifiers_test):
                    fc3_w = fc3_distribution.rsample()
                    classifiers.append(fc3_w)

                outputs = []
                for index in range(self.num_classifiers_test):
                    out = F.linear(x, classifiers[index], self.fc3_bias)
                    outputs.append(out)

                return outputs
