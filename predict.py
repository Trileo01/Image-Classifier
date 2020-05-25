import torch
import argparse
import json
from torch import nn
from torchvision import models,transforms
from PIL import Image

parser = argparse.ArgumentParser(description= "Predict.py")
parser.add_argument('img_dir', nargs='*', action='store')
parser.add_argument('--checkpoint_dir', nargs='*', action='store')
parser.add_argument('--topK', nargs='*', action='store', default=1, type=int, dest='topK')
parser.add_argument('--category_name', nargs='*', action='store', default='./cat_to_name.json', dest='category_names')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu')
args = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

checkpoint = torch.load(args.checkpoint_dir[0])

arch = getattr(models,checkpoint['model'])
model = arch(pretrained=True)

classifier = model.classifier
input_features = classifier.in_features
classifier = nn.Sequential(nn.Linear(input_features, checkpoint['hidden_layer']),
                           nn.ReLU(),
                           nn.Linear(checkpoint['hidden_layer'], 512),
                           nn.ReLU(),
                           nn.Linear(512, 102),
                           nn.LogSoftmax(dim=1)
                           )
model.classifier = classifier
model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])

image = Image.open(args.img_dir[0])
transformation = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
img = transformation(image)
idx_to_class = {value : key for key,value in model.class_to_idx.items()}
img = img.unsqueeze_(0)

if args.gpu:
    model.to('cuda')
    img = img.to('cuda')

output = model(img)
predict = torch.exp(output)
top_values, top_class = predict.topk(args.topK , dim = 1)
top_values, top_class = top_values.cpu(), top_class.cpu()
top_class = top_class.detach().numpy().tolist()[0]
top_class = [cat_to_name[idx_to_class[species]] for species in top_class]
top_values = top_values.detach().numpy().tolist()[0]

for i in range(args.topK):
    print("Predicted Flower : {} with probabiltiy of {}".format(top_class[i],top_values[i]))


