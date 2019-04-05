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



def predict_image(image_path):
    #load the model
    model=load_checkpoint(args.checkpoint)
    
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #Implement the code to predict the class from an image file
    
    image = helper.process_image(image_path)
    image = torch.from_numpy(image).float()
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    
    model=model.to(device)
    model.eval()
    image=image.to(device)
    
    
    output= model.forward(image)
    prob= torch.exp(output)
    prob, classes = prob.topk(arg.top_k, dim=1)
    
    top_prob=prob.data[0].tolist()
    top_classes = []
    for e in classes.data[0].cpu().numpy():
        for key, value in model.class_to_idx.items():
            if e==value:
                top_classes.append(int(key))
    
    
    return top_prob, top_classes
    


def load_checkpoint (file):
    checkpoint=torch.load(file)
    classifier= checkpoint['classifier']
    #classifier= classifier.load_state_dict(checkpoint['clas_state_dict'])
    if checkpoint['arch'] =="vgg16": 
        model=models.vgg16(pretrained=True)
    if checkpoint['arch'] == "vgg13":
        model=models.vgg13(pretrained=True)
        
    model.classifier=classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['map_classes']
      
    return model    

def main():
    
    """Launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument('image', action='store', help='Directory of the picture to be predicted' )
    parser.add_argument('checkpoint', default = 'checkpoint_app.pth', help='Directory of checkpoint')
    parser.add_argument('--top_k', type = int, default = 5 , help='Number of top K')
    parser.add_argument('--category_names', default = 'cat_to_name.json', help="Learning rate")
    parser.add_argument('-gpu', action="store_true", help = 'gpu enable')
    
    args = parser.parse_args()
    
    
    print(args.image)
    print(args.checkpoint)
    print(args.top_k)
    print(args.category_names)
    print (args.gpu)
    
    probs, classes = predict_image(args.image)
    names=[]
    for e in classes:
        for key, value in cat_to_name.items():
            if str(e) == key:
                names.append(value)
                
    Print(names , probs)


if __name__ == "__main__":
    main()
