from PIL import Image
from torchvision import datasets, transforms, models
import torch

#funtion to process the image to be entered 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256,256))
    im = im.crop((0,0,224,224))
    np_image = np.array(im)
    np_image= np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image= (np_image-mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image


def loaders(train_dir, valid_dir,test_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    # I create a transform for training dataset. 

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop (224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
  
# I create a transform for test and validation dataset. In this case it is only needed to resized, converted to tensor and normalizedt
    test_valid_transforms = transforms.Compose([
                                        transforms.Resize (225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

    

    # Load the data set for each one (train,validation and test)  
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_valid_transforms)


    #define the dataloaders for each one
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=32,shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    test_loaders =  torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return train_loaders, valid_loaders,  test_loaders, train_datasets