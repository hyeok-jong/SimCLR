import torch
import torchvision

'''
VGG models have avgpool after features
So that for any input after avgpooling -> [batch, channel, 7, 7]
'''

class VGGSimCLR(torch.nn.Module):
    
    def __init__(self, base_model = 'vgg16', mode = "SimCLR", out_dim = 128) :
        super(VGGSimCLR, self).__init__()
        self.vgg_dict = {"vgg11": torchvision.models.vgg11(weights = None, num_classes = out_dim),
                        "vgg16": torchvision.models.vgg16(weights = None, num_classes = out_dim),
                        "vgg19": torchvision.models.vgg19(weights = None, num_classes = out_dim)}

        self.encoder = self.get_encoder(base_model)

        if mode == 'SimCLR':
            self.head = torch.nn.Sequential(torch.nn.Linear(512*2*2, 128),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(128, out_dim))
        elif mode == 'Linear':
            # Here, out_dim is number of classes
            self.head = torch.nn.Linear(512*2*2, out_dim)
        

    def get_encoder(self, model_name):
        model = self.vgg_dict[model_name]
        return model.features

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.head(x)
        x = torch.nn.functional.normalize(x, dim = 1)
        return x

if __name__ == '__main__':
    test_model = VGGSimCLR()
    print(test_model)
    input = torch.zeros([10,3,64,64])
    output = test_model(input)
    print(input.shape,'-->>>',output.shape)