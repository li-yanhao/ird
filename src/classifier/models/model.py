import torch, torch.nn as nn, torch.nn.functional as F, random
import lightning as L
# from .config import Config
import torchvision.models as models


class ModelBase(nn.Module):
    def __init__(self, name):
        super(ModelBase, self).__init__()
        self.name = name

    def copy_params(self, state_dict):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

    def boost_params(self, scale=1.0):
        if scale == 1.0:
            return self.state_dict()
        for (name, param) in self.state_dict().items():
            self.state_dict()[name].copy_((scale * param).clone())
        return self.state_dict()

    # self - x
    def sub_params(self, x):
        own_state = self.state_dict()
        for (name, param) in x.items():
            if name in own_state:
                own_state[name].copy_(own_state[name] - param)

    # self + x
    def add_params(self, x):
        a = self.state_dict()
        for (name, param) in x.items():
            if name in a:
                a[name].copy_(a[name] + param)


class EfficientNet(L.LightningModule):
    def __init__(self, model:str, num_class:int):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging

        # Load a pre-trained EfficientNet model
        if model.startswith("efficient-b0"):
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if model.startswith("efficient-b2"):
            self.efficientnet = models.efficientnet_b2(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if model.startswith("efficient-b3"):
            self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B0_Weights.DEFAULT)

        self.const_head = False
        if model.endswith("const"):
            self.const_head = True
            self.register_parameter("const_weight", None)
            self.const_weight = nn.Parameter(torch.randn(size=[3, 3, 5, 5]), requires_grad=True)

        # self.preconv = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        
        # Replace the final classification layer for your specific number of classes
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_class)

    def normalized_F(self):
        for j in range(3):
            central_pixel = (self.const_weight.data[:, j, 2, 2])
            for i in range(3):
                sumed = self.const_weight.data[i,j].sum() - central_pixel[i]
                self.const_weight.data[i, j] /= sumed
                self.const_weight.data[i, j, 2, 2] = -1.0

    def forward(self, x):
        if self.const_head:
            self.normalized_F()
            x = F.conv2d(x, self.const_weight)

        return self.efficientnet(x), None


class LightningModel(L.LightningModule):
    def __init__(self, model, class_names=None):
        super(LightningModel, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_names = class_names

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # # Dynamically import the config file
    import importlib.util
    spec = importlib.util.spec_from_file_location("config1", "configs/config1.py")
    config1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config1)
    config = config1.Config()

    ## Statically import the config file
    # from configs.config1 import Config
    # config = Config()

    print(config)

    model = ConstNet(conf=config)
    print(model)

    x = torch.randn(1, 1, 256, 256)
    logit, output = model(x)
    print(logit.shape)
