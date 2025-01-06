import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self,inchan,outchan):
        super().__init__()
        augment  = inchan!=outchan
        self.augment = augment
        self.outchan = outchan
        self.inchan = inchan
        stride = 1
        if augment:
            stride = 2
        self.inchan = inchan
        self.outchan = outchan
        self.conv1 = nn.Conv2d(inchan,outchan,3,padding=1,stride = stride)
        self.bn1 = nn.BatchNorm2d(outchan)
        self.conv2 = nn.Conv2d(outchan,outchan,3,padding=1,stride = 1)
        self.bn2 = nn.BatchNorm2d(outchan)
        self.skip = nn.Sequential()
        if augment:
            self.skip = nn.Sequential(nn.Conv2d(inchan,outchan,kernel_size=1,stride=2,bias=False)
            , nn.BatchNorm2d(outchan)
            )
        
        self.relu1 = nn.ReLU()
        self.relu2  = nn.ReLU()
    def forward(self,x):
        
        straight_path = self.relu1(self.bn1(self.conv1(x)))
        straight_path = self.bn2(self.conv2(straight_path))
        return self.relu2(self.skip(x)+straight_path)

class Resnet3(nn.Module):
    def __init__(self,inchan,vec_len):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchan,64,3),
            nn.BatchNorm2d(64)

        )

        self.block1 = nn.Sequential(ResidualBlock(64,64),ResidualBlock(64,64),ResidualBlock(64,64),ResidualBlock(64,128))
        self.block2 = nn.Sequential(ResidualBlock(128,128),ResidualBlock(128,128),ResidualBlock(128,128),ResidualBlock(128,512))
        self.block3 = nn.Sequential(ResidualBlock(512,512),ResidualBlock(512,512),ResidualBlock(512,512),ResidualBlock(512,2**10))
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024,vec_len)


    def forward(self,x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        vec = self.pool(x)
        vec = torch.flatten(vec,1)
        vec = self.fc(vec)
        return vec
    
class Resnetmini(nn.Module):
    def __init__(self,inchan,vec_len):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchan,64,3),
            nn.BatchNorm2d(64)

        )

        self.block1 = nn.Sequential(ResidualBlock(64,64),ResidualBlock(64,64),ResidualBlock(64,128))
        self.block2 = nn.Sequential(ResidualBlock(128,128),ResidualBlock(128,128),ResidualBlock(128,512))
        self.block3 = nn.Sequential(ResidualBlock(512,512),ResidualBlock(512,512),ResidualBlock(512,2**10))
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(1024,1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024,vec_len)
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        vec = self.pool(x)
        vec = torch.flatten(vec,1)
        vec = self.fc1(vec)
        vec = self.relu(vec)

        vec = self.fc2(vec)
        return vec
# test if this crap works



if __name__=="__main__":
    from torchvision import transforms
    from PIL import Image
    transform = transforms.Compose( [transforms.PILToTensor(),transforms.ConvertImageDtype(torch.float)] )

    i = Image.open("image.png")
    width, height = i.size
    i=i.crop((0,0,min(width,height),min(width,height)))
    data = transform(i.convert("RGB"))
    train_set = torch.stack([data,data])

    model = Resnetmini(3,600)
    print(model(train_set))