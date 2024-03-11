import os, sys
from libs import *

class Generator():
    def __init__(self, 
        ckp_dir, 
    ):
        self.GS = torch.load(
            ckp_dir, 
            map_location = "cpu", 
        )

        self.transform = A.Compose(
            [
                A.Resize(
                    height = 128, width = 128, 
                ), 
                A.Normalize(), AT.ToTensorV2(), 
            ]
        )
        self.denormalize = A.Normalize(
            mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255, ], std = [1/0.229, 1/0.224, 1/0.255, ], 
            max_pixel_value = 1.0, 
        )

    def generate(self, 
        image, 
    ):
        image = self.transform(image = image)["image"].unsqueeze(0)

        generated_image = self.GS(image).detach().squeeze(0).permute(1, 2, 0, ).numpy()
        generated_image = self.denormalize(image = generated_image)["image"]*255

        return generated_image.astype(int).clip(0, 255)