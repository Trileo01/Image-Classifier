import torch
import argparse
from torch import nn, optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from model_loader import load_model

parser = argparse.ArgumentParser(description="Train.py")
parser.add_argument('data_dir', action='store')
parser.add_argument('--save_dir', action='store', default='./', dest = 'dir')
parser.add_argument('--arch',action='store', default='densenet', choices=['vgg','densenet'], dest='model')                    
parser.add_argument('--learning rate', action='store', default=0.001, type=int, dest='lr')
parser.add_argument('--hidden units', action='store', default=768, type=int, dest='hidden')
parser.add_argument('--epochs', type=int, action='store',
                    default=8, dest='epochs')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu')
args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(35),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
}

model = load_model(args.model, args.hidden)
optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
criterion = nn.NLLLoss()

if args.gpu:
    model.to("cuda")

steps = 0
print_every = 5
step = 0

with active_session():
    for epoch in range(args.epochs):
        for images, labels in dataloaders['train']:
            running_loss = 0
            steps += 1
            if args.gpu:
                images, labels = images.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to("cuda"), labels.to("cuda")
                        output = model(images)
                        vlost = criterion(output, labels)
                        predict = torch.exp(output)
                        top_p, top_class = predict.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                vlost = vlost / len(dataloaders['valid'])
                accuracy = accuracy /len(dataloaders['valid'])
                print("Epoch: {}/{}... ".format(epoch+1, args.epochs),"Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                      "Accuracy: {:.4f}".format(accuracy))
                model.train()
                
checkpoint = {'model' : args.model,
              'hidden_layer' : args.hidden,
              'class_to_idx' : image_datasets['train'].class_to_idx,
              'state_dict' : model.state_dict()}
checkpoint_dir = args.dir + '/checkpoint.pth'
torch.save(checkpoint, checkpoint_dir)
