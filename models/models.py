import torch
import torch.nn as nn

class Myresnext50(nn.Module):
    def __init__(self, my_pretrained_model, num_classes = 3):
        super(Myresnext50, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, num_classes))
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        
        pred = torch.sigmoid(x.reshape(x.shape[0], 1,self.num_classes))
        return pred

# all the network I mentioned today :) haha


