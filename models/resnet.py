import torch
import torchvision

# Resnets have avgpooling after feature extractor
# So that after CNN blocks Tensor goes to [Batch, Channel, 1, 1]
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
        
class ResNetSimCLR(torch.nn.Module):

    def __init__(self, base_model = 'resnet18', mode = 'SimCLR', out_dim = 128):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": torchvision.models.resnet18(weights = None, num_classes=out_dim),
                            "resnet34": torchvision.models.resnet34(weights = None, num_classes=out_dim),
                            "resnet50": torchvision.models.resnet50(weights = None, num_classes=out_dim)}

        self.encoder, dim_in = self.get_encoder(base_model)

        if mode == 'SimCLR':
            self.head = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(dim_in, out_dim))
        elif mode == 'Linear':
            # Here, out_dim is number of classes
            self.head = torch.nn.Linear(dim_in, out_dim)

    def get_encoder(self, model_name):
        model = self.resnet_dict[model_name]
        # Note that this model react any size due to [nn.AdaptiveAvgPool2d((1, 1))]
        # Thus, dim_in is same with different input sizes.
        dim_in = model.fc.in_features
        model.fc = Identity()
        return model, dim_in

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        x = torch.nn.functional.normalize(x, dim = 1)
        return x

if __name__ == '__main__':
    test_model = ResNetSimCLR()
    print(test_model)
    input = torch.zeros([10,3,64,64])
    output = test_model(input)
    print(input.shape,'-->>>',output.shape)