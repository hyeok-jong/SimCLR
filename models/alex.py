import torch
import torchvision

# AlexNet input size in fixed 256
# In LPIPS it doesn't matter
# For SimCLR change first module of FC layer.


class AlexSimCLR(torch.nn.Module):

    def __init__(self, base_model = 'alex', mode = 'SimCLR', out_dim = 128):
        super(AlexSimCLR, self).__init__()
        self.alexnet_dict = {"alex": torchvision.models.alexnet(weights = None, num_classes=out_dim)}

        self.encoder = self.get_encoder(base_model)

        if mode == 'SimCLR':
            self.head = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(256, out_dim))
        elif mode == 'Linear':
            # Here, out_dim is number of classes
            self.head = torch.nn.Linear(256, out_dim)

    def get_encoder(self, model_name):
        model = self.alexnet_dict[model_name]
        return model.features

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.head(x)
        x = torch.nn.functional.normalize(x, dim = 1)
        return x

if __name__ == '__main__':
    test_model = AlexSimCLR()
    print(test_model)
    input = torch.zeros([10,3,64,64])
    output = test_model(input)
    print(input.shape,'-->>>',output.shape)