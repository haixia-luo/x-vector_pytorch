import torch.nn as nn
import sys
import torch.nn.functional as F
import torch
from torchvision import transforms
import trainset
import testset
import TDNN as tdnn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

class Time_Delay(nn.Module):
    def __init__(self, context, input_dim, output_dim, node_num, full_context):
        super(Time_Delay, self).__init__()
        self.tdnn1 = tdnn.TDNN(context[0], input_dim, node_num[0], full_context[0])
        self.tdnn2 = tdnn.TDNN(context[1], node_num[0], node_num[1], full_context[1])
        self.tdnn3 = tdnn.TDNN(context[2], node_num[1], node_num[2], full_context[2])
        self.tdnn4 = tdnn.TDNN(context[3], node_num[2], node_num[3], full_context[3])
        self.tdnn5 = tdnn.TDNN(context[4], node_num[3], node_num[4], full_context[4])
        self.fc1 = nn.Linear(node_num[5], node_num[6])
        self.fc2 = nn.Linear(node_num[6], node_num[7])
        self.fc3 = nn.Linear(node_num[7], output_dim)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.batch_norm5 = nn.BatchNorm1d(512)
        self.batch_norm6 = nn.BatchNorm1d(512)
        self.batch_norm7 = nn.BatchNorm1d(256)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self,x):
        a1 = F.relu(self.batch_norm1(self.tdnn1(x)))
        a2 = F.relu(self.batch_norm2(self.tdnn2(a1)))
        a3 = F.relu(self.batch_norm3(self.tdnn3(a2)))
        a4 = F.relu(self.batch_norm4(self.tdnn4(a3)))
        a5 = F.relu(self.batch_norm5(self.tdnn5(a4)))
        a6 = self.statistic_pooling(a5)
        a7 = F.relu(self.batch_norm6(self.fc1(a6)))
        a8 = F.relu(self.batch_norm7(self.fc2(a7)))
        output = self.fc3(a8)
        return output
    
    def statistic_pooling(self, x):
        mean_x = x.mean(dim = 2)
        std_x = x.std(dim = 2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

def train():
    context = [[-2,2],[-2,0,2],[-3,0,3],[0],[0]]
    node_num = [256,256,256,256,512,1024,512,256]
    full_context = [True, False, False, True, True]
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    net = Time_Delay(context, 24, 300, node_num, full_context)
    net = net.to(device)
    train_set = trainset.TrainSet('../all_feature/')
    trainloader = DataLoader(train_set, batch_size = 128, num_workers = 16, shuffle = True)
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)
    torch.set_num_threads(16)
    for epoch in range(80):
        running_loss = 0
        total = 0
        correct = 0
        if epoch % 16 == 15:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        for i, inputs in enumerate(trainloader, 0):
            data, label = inputs
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = nn.CrossEntropyLoss()(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum()            
            if i % 10 == 9:
                print(epoch + 1, i + 1, 'loss:', running_loss, 'accuracy:{:.2%}'.format(correct.item() / total))
                running_loss = 0
                total = 0
                correct = 0
    torch.save(net.state_dict(), 'net.pth')
    return net

def test(net):
    net = net.to('cuda:1')
    total = 0
    correct = 0
    test_set = testset.TestSet('../all_feature/')
    testloader = DataLoader(test_set, batch_size = 128, num_workers = 16, shuffle = True)
    torch.set_num_threads(16)
    with torch.no_grad():
        for data in testloader:
            feature, label = data
            feature = feature.to('cuda:1')
            label = label.to('cuda:1')
            outputs = net(feature)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
    print('correct:', correct.item())
    print('total:', total)
    print('testset中的准确率为: %d %%' % (100 * correct / total))
                
if __name__ == '__main__':
    net = train()
    context = [[-2,2],[-2,0,2],[-3,0,3],[0],[0]]
    node_num = [256,256,256,256,512,1024,512,256]
    full_context = [True, False, False, True, True]
    net = Time_Delay(context, 24, 300, node_num, full_context)
    net.load_state_dict(torch.load(sys.argv[1]))
    test(net)
