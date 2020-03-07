# ESE 587, Stony Brook University
# Handout Code for PyTorch Warmup 1

# Based on PyTorch tutorial code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

############################
# Parameters you can adjust
showImages = 0          # Will show images as demonstration if = 1
batchSize = 64          # The batch size used for learning
learning_rate = 0.01    # Learning rate used in SGD
momentum = 0.5          # Momentum used in SGD
epochs = 3              # Number of epochs to train for


############################################
# Set up our training and test data
# The torchvision package gives us APIs to get data from existing datasets like MNST
# The "DataLoader" function will take care of downloading the test and training data

import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######################################
# Let's look at a few random images from the training data

import matplotlib.pyplot as plt
import numpy as np

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  #undo normalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if (showImages>0):

    # Grab random images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    images = images[0:4]
    labels = labels[0:4]

    # print labels
    print(' '.join('%s' % classes[labels[j]] for j in range(4)))
    # Show images
    imshow(torchvision.utils.make_grid(images))


##################################
# Define our network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*14*14, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 5*14*14)                
        x = F.log_softmax(self.fc1(x), dim=1) 
        return x

    # Some simple code to calculate the number of parametesr
    def num_params(self):
        numParams = 0
        for param in myNet.parameters():
            thisLayerParams=1
            for s in list(param.size()):
                thisLayerParams *= s
            numParams += thisLayerParams

        return numParams


myNet = Net()
print(myNet)
print("Total number of parameters: ", myNet.num_params())
 
###################################
# Training

import torch.optim as optim

# Loss function: negative log likelihood
criterion = nn.NLLLoss()

# Configuring stochastic gradient descent optimizer
optimizer = optim.SGD(myNet.parameters(), lr=learning_rate, momentum=momentum)

# Each epoch will go over training set once; run two epochs
for epoch in range(epochs): 

    running_loss = 0.0

    # iterate over the training set
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data

        # Clear the parameter gradients
        optimizer.zero_grad()

        #################################
        # forward + backward + optimize

        # 1. evaluate the current network on a minibatch of the training set
        outputs = myNet(inputs)              

        # 2. compute the loss function
        loss = criterion(outputs, labels)  

        # 3. compute the gradients
        loss.backward()                    

        # 4. update the parameters based on gradients
        optimizer.step()                   

        # Update the average loss
        running_loss += loss.item()

        # Print the average loss every 256 minibatches ( == 16384 images)
        if i % 256 == 255:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 256))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():       # this tells PyTorch that we don't need to keep track
                                # of the gradients because we aren't training
        for data in testloader:
            images, labels = data
            outputs = myNet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Epoch %d: Accuracy of the network on the %d test images: %d/%d = %f %%' % (epoch+1, total, correct, total, (100 * correct / total)))


print('Finished Training!')



###################################
# Let's look at some test images and see what our trained network predicts for them

if (showImages > 0):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images[0:4]
    labels = labels[0:4]
    outputs = myNet(images)
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%10s' % classes[predicted[j]] for j in range(4)))

    imshow(torchvision.utils.make_grid(images))


##################################
# Let's comptue the total accuracy across the training set

correct = 0
total = 0
with torch.no_grad():       # this tells PyTorch that we don't need to keep track
                            # of the gradients because we aren't training
    for data in trainloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d training images: %f %%' % (total, (100 * correct / total)))



##################################
# Now we want to compute the total accuracy across the test set

correct = 0
total = 0
with torch.no_grad():       # this tells PyTorch that we don't need to keep track
                            # of the gradients because we aren't training
    for data in testloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d/%d = %f %%' % (total, correct, total, (100 * correct / total)))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = myNet(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %10s : %f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

