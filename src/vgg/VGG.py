import torch
import torch.nn as nn


class VGG19FromScratch(nn.Module):

    def __init__(self , num_classes , in_channels=3):
        super().__init__()

        self.conv_layers = nn.Sequential(

            #1st block
            nn.Conv2d(in_channels=in_channels , out_channels= 64 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 , out_channels= 64 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),

            #2nd block

            nn.Conv2d(in_channels=64 , out_channels= 128 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128 , out_channels= 128 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),

            #3rd block

            nn.Conv2d(in_channels=128 , out_channels= 256 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256 , out_channels= 256 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256 , out_channels= 256 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256 , out_channels= 256 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),

            #4th block

            nn.Conv2d(in_channels=256 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),

            #5th
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512 , out_channels= 512 , padding=1 , kernel_size = (3,3) , stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2),
        )

        # classifier
        self.classifier =  nn.Sequential(   
            nn.Linear(512*7*7 , 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

      # Apply He (Kaiming) initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for Conv layers (fan_out mode is better for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # He initialization for Linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


    def forward(self , input):

        x = self.conv_layers(input)
        x = torch.flatten(x , 1)
        logits = self.classifier(x)

        return logits