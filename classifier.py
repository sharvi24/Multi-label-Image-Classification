import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 21


class SimpleClassifier(nn.Module):
   def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

   def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Classifier(nn.Module):
#     # TODO: implement me
#     def __init__(self):
#         super(Classifier, self).__init__()
        
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
#             nn.Conv2d(16, 16, kernel_size=3), nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(16),

#             nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(32),

#             nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(64),

#             nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.BatchNorm2d(128),
            
#             #nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
#             #nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(),
#             #nn.MaxPool2d(2,2),
#             #nn.BatchNorm2d(256),

#         ).to(device)
        
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(0.25),
#             #nn.Linear(3*3*256,256),
#             nn.Linear(10*10*128, 128),
#             nn.ReLU(),

#             nn.Dropout(0.5),
#             #nn.Linear(256,num_classes)
#             nn.Linear(128, num_classes)
#         ).to(device)
        
#     def forward(self, x):
#         x = self.model(x)
#         x = self.classifier(x)
#         return x
        



class Classifier(nn.Module):
    # TODO: implement me
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(1024)
        self.dropouts = nn.Dropout(p=0.1)

        self.conv9 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(2048)
        self.conv10 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(2048)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1179648, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024,num_classes)
        self.dropout = nn.Dropout(p=0.505)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = (F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = (F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = (F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(F.relu(self.bn8(self.conv8(x))))
        x = self.dropouts(x)
        x = (F.relu(self.bn9(self.conv9(x))))
        x = self.pool(F.relu(self.bn10(self.conv10(x))))


        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
