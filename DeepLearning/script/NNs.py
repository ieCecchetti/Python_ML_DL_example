# function to define an old style fully connected network (multilayer perceptrons)
class old_nn(nn.Module):
    def __init__(self):
        super(old_nn, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes) #last FC for classification 

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
      
#function to define the convolutional network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #conv2d first parameter is the number of kernels at input (you get it from the output value of the previous layer)
        #conv2d second parameter is the number of kernels you wanna have in your convolution, so it will be the n. of kernels at output.
        #conv2d third, fourth and fifth parameters are, as you can read, kernel_size, stride and zero padding :)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, n_classes) #last FC for classification 

    def forward(self, x):
        x = F.relu(self.conv1(x))                  #32-5/2+1=14
        x = F.relu(self.conv2(x))                  #14-3/1+1=12
        x = F.relu(self.conv3(x))                  #10
        x = F.relu(self.pool(self.conv_final(x)))  #10-3
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))                    #7-3=4
        #hint: dropout goes here!
        x = self.fc2(x)
        return x
      
      
#function to define the convolutional network
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        #conv2d first parameter is the number of kernels at input (you get it from the output value of the previous layer)
        #conv2d second parameter is the number of kernels you wanna have in your convolution, so it will be the n. of kernels at output.
        #conv2d third, fourth and fifth parameters are, as you can read, kernel_size, stride and zero padding :)
        self.conv1 = nn.Conv2d(3, 256, kernel_size=5, stride=2, padding=0)   
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)  
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(256)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.BatchNorm2d(256) 
        self.dropout = nn.Dropout(p=0.7)
        self.conv_final = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(512 * 4 * 4, 8192)
        self.fc2 = nn.Linear(8192, n_classes) #last FC for classification 

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))                   #32-5/2+1=14    
        x = F.relu(self.conv2_bn(self.conv2(x)))                   #14-3/1+1=12 
        x = F.relu(self.conv3_bn(self.conv3(x)))                   #12-3/1+1=10
        x = F.relu(self.pool(self.conv_final(x)))                  #10-3/1+1=8  & 8-2/2+1=4
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
