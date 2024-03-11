import os, sys
from libs import *

from dassl.modeling.network import fcn_3x64_gctx

class fcn_resnet18(nn.Module):
    def __init__(self, 
        num_classes = 0, 
    ):
        super(fcn_resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18(
            pretrained = True, 
        )
        self.backbone.fc = nn.Identity()

        if num_classes > 0:
            self.classifier = nn.Linear(
                512, num_classes, 
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, 
        input, 
    ):
        output = self.classifier(self.backbone(input))

        return output