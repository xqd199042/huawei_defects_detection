import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms, Normalize


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Unit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1,3), (1, stride), (0,1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (3,1), (stride, 1), (1,0)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes, drop_rate):
        super(CNN, self).__init__()
        self.layer1 = Unit(3, 16)       #16 64 64
        self.layer2 = Unit(16, 32, 2)   #32 32 32
        self.layer3 = nn.Sequential(
            Unit(32, 64, 2),
            nn.Dropout(drop_rate)
        )   #64 16 16
        self.layer4 = nn.Sequential(
            Unit(64, 64, 2),
            nn.Dropout(drop_rate)
        )  #128 8 8
        self.layer5 = nn.Sequential(
            Unit(64, 128, 2),
            nn.Dropout(drop_rate)
        ) #256 4 4
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

if __name__ == '__main__':
    epochs = 2000
    num_classes = 6
    best_performance = 0.8
    batch_size = 200
    save_path = 'model'
    drop_rate= 0.3

    train_data = datasets.ImageFolder('out/train', transform=transforms.Compose([
        transforms.ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    test_data = datasets.ImageFolder('out/test', transform=transforms.Compose([
        transforms.ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0')
    net = CNN(num_classes, drop_rate)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    # modify...
    # optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for img, label in train_data:
            img = img.to(device)
            label = label.to(device)
            out = net(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predict = out.max(1)
            num_correct = (predict==label).sum().item()
            acc = num_correct/img.shape[0]
            train_acc += acc
        train_loss /= len(train_data)
        train_acc /= len(train_data)

        test_acc = 0
        net.eval()
        for img, label in test_data:
            img = img.to(device)
            label = label.to(device)
            out = net(img)
            _, predict = out.max(1)
            num_correct = (predict == label).sum().item()
            acc = num_correct / img.shape[0]
            test_acc += acc
        test_acc /= len(test_data)
        if test_acc > best_performance:
            best_performance = test_acc
            torch.save(net.state_dict(), save_path)
        print('Epoch {}, training loss: {}, training accuracy: {}, testing accuracy: {}'.format(epoch+1, train_loss, train_acc, test_acc))
        lr_decay.step()


























