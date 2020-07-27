import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class Predictor(nn.Module):
    def __init__(
            self, 
            num_classifiers_train=2,
            num_classifiers_test=20,
            init='kaiming_u',
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

        self.mu1 = Parameter(torch.randn(3072, 2048))
        self.sigma1 = Parameter(torch.zeros(3072, 2048))

        self.mu2 = Parameter(torch.randn(2048, 10))
        self.sigma2 = Parameter(torch.zeros(2048, 10))

        if use_init:
            all_parameters = [self.mu1, self.sigma1, self.mu2, self.sigma2]
            for item in all_parameters:
                function_init[self.init](item)

        self.b1 = Parameter(torch.zeros(2048))
        self.b2 = Parameter(torch.zeros(10))
        self.bn1_fc = nn.BatchNorm1d(2048)

    def forward(self, x, only_mu=True):
        # Keep sigma values be positive
        sigma1_pos = torch.sigmoid(self.sigma1)
        sigma2_pos = torch.sigmoid(self.sigma2)

        fc1_distribution = normal.Normal(self.mu1, sigma1_pos)
        fc2_distribution = normal.Normal(self.mu2, sigma2_pos)

        if self.training:
            classifiers = []
            for index in range(self.num_classifiers_train):
                fc1_w = fc1_distribution.rsample()
                fc2_w = fc2_distribution.rsample()
                one_classifier = [fc1_w, self.b1, fc2_w, self.b2]

                classifiers.append(one_classifier)

            outputs = []
            for index in range(self.num_classifiers_train):
                out = torch.matmul(x, classifiers[index][0]) + classifiers[index][1]
                out = F.relu(self.bn1_fc(out))
                out = torch.matmul(out, classifiers[index][2]) + classifiers[index][3]
                outputs.append(out)

            return outputs
        else:
            if only_mu:
                # Only use mu for classification
                mus = [self.mu1, self.mu2, self.b1, self.b2]

                out = torch.matmul(x, mus[0]) + mus[2]
                out = F.relu(self.bn1_fc(out))
                out = torch.matmul(out, mus[1]) + mus[3]

                return [out]
            else:
                classifiers = []
                for index in range(self.num_classifiers_test):
                    fc1_w = fc1_distribution.rsample()
                    fc2_w = fc2_distribution.rsample()
                    one_classifier = [fc1_w, self.b1, fc2_w, self.b2]

                    classifiers.append(one_classifier)

                outputs = []
                for index in range(self.num_classifiers_test):
                    out = torch.matmul(x, classifiers[index][0]) + classifiers[index][1]
                    out = F.relu(self.bn1_fc(out))
                    out = torch.matmul(out, classifiers[index][2]) + classifiers[index][3]
                    outputs.append(out)

                return outputs
