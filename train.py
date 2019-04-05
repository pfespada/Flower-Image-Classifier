import argparse
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import helper
import json


def train_network(args):
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #define the dataloaders using a funtion loaders defined in helper.py
    train_loaders, valid_loaders,  test_loaders, train_datasets = helper.loaders(train_dir, valid_dir, test_dir)

    # read the cat_to_name.json, it gives you a dictionary mapping the integer encoded categories to the actual names of the flowers.
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # define the device (GPU or CPU) to be used based on the availability
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        

    #load the model selected from the pre-trained networks availables (vgg16,vgg13)
    if args.arch=='vgg16':
        model= models.vgg16(pretrained=True)
    else:
        model= models.vgg13(pretrained=True)
        
        
    # freeze the features parameter of the model in order to avoid update them and I modify the classifier(turn off gradients)
    for param in model.parameters():
        param.requires_grad = False

    # build a new classifier to be change in the model. The pre-calssifier has 25088 input features and we need 102 classes as an output

    from collections import OrderedDict
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, args.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.4)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier=classifier
    #set the criterion and optimizer

    criterion= nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    #convert model based on the device available 
    model.to(device)
    
    epochs= args.epochs
    steps = 0
    print_every = 30

    for e in range(epochs):
        train_loss = 0 # reset the training loss in each epoch
        model.train()
        for inputs, labels in train_loaders:
             steps+=1
             #use the device available
             inputs, labels = inputs.to(device), labels.to(device)

             optimizer.zero_grad()

             log_probs=model.forward(inputs)
             loss = criterion(log_probs,labels)
             loss.backward()
             optimizer.step()

             train_loss += loss.item()

             if steps % print_every == 0:
                 model.eval()

                 valid_loss = 0 
                 accuracy = 0

                 with torch.no_grad():

                     valid_loss = 0 
                     accuracy = 0
                     #model.to(device)
                     for images, labels in valid_loaders:

                         images, labels = images.to(device), labels.to(device)

                         log_prob=model.forward(images)
                         loss= criterion(log_prob,labels) #loss of the batch of images
                         valid_loss += loss.item() #loss acumulation

                         prob= torch.exp(log_prob)
                       
                         equals = (prob.max(dim=1)[1] == labels.data) # it gives us a tensor with True/False. it a comparation of results and targets
                         accuracy += equals.type(torch.FloatTensor).mean()

                 model.train()


                 print("Epoch: {}/{}.. ".format(e+1, epochs),
                         "Training Loss: {:.3f}.. ".format(train_loss/print_every),
                         "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loaders)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loaders)))

                 train_loss = 0
                
    #save the model's checkpoint
    # TODO: Save the checkpoint 
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint= {'classifier':model.classifier, 
                'state_dict': model.state_dict(),
                'clas_state_dict': model.classifier.state_dict(),
                'map_classes': model.class_to_idx,
                'name_classes':cat_to_name,
                'num_epoch':epochs,
                'arch':args.arch}

    torch.save(checkpoint, args.save_dir)

def main():
    
    """Launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', action='store', default= 'flowers', help='Directory of the pictures' )
    parser.add_argument('--save_dir', type=str, default = 'checkpoint_app.pth', help='Directory of checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'vgg13'], help='Network architecture')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help="Learning rate")
    parser.add_argument('--epochs', type= int, default= 1, help="Number of epochs")
    parser.add_argument('--hidden_units', type = int, default = 12595, help ='Number of hidden units')
    parser.add_argument('-gpu', action="store_true", help = 'gpu enable')
    
    args = parser.parse_args()
    
    
    print(args.data_directory)
    print(args.save_dir)
    print(args.arch)
    print(args.learning_rate)
    print(args.epochs)
    print (args.gpu)
    print(args.hidden_units)
    
    train_network(args)


if __name__ == "__main__":
    main()
