# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import os
import time
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2, glob, numpy as np, pandas as pd
import torchvision
from torchvision import transforms as T
import albumentations as A
#import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageData(Dataset):
  def __init__(self, image_dir, mask_dir,transform=None):
    self.image_dir=image_dir
    self.mask_dir=mask_dir
    self.images=sorted(os.listdir(image_dir))
    self.masks=sorted(os.listdir(mask_dir))
    self.transform=transform


  def __len__(self):
      #return len(self.image_dir)
      return len(os.listdir(self.mask_dir))

  def __getitem__(self,index):

    img_path=os.path.join(self.image_dir,self.images[index])
    mask_path=os.path.join(self.mask_dir,self.masks[index])

    #image = (cv2.imread(img_path)[:,:,::-1])  ##  cv2 imread reads the image as BGR in default, so the code changes it to RGB
    image = cv2.imread(img_path,cv2.IMREAD_COLOR)

    image = cv2.resize(image, (256,256))
    image_T=image
    #image=image/255.0  ##(256,256,3)
    image=np.transpose(image,(2,0,1))  ## (3,256,256)
    image=image.astype(np.float32)
    image=image/255.0  ##(256,256,3)
    image=torch.from_numpy(image)

    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256,256))
    mask_T=mask
    mask=mask/255.0
    mask=np.expand_dims(mask,axis=0)
    mask=mask.astype(np.float32)
    mask=torch.from_numpy(mask)

    if self.transform is not None:
            transformed = self.transform(image=image_T, mask=mask_T)
            image = transformed["image"]
            mask = transformed["mask"]



            image=np.transpose(image,(2,0,1))
            image=image.astype(np.float32)
            image=image/255.0

            image=torch.from_numpy(image)

            mask=mask.astype(np.float32)
            mask=mask/255.0
            mask=torch.from_numpy(mask)

    return image,mask

trn_tfms=A.Compose(
    [
     A.Resize(width=256, height=256),
     A.RandomCrop(width=256, height=256),
     A.Rotate(limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),
     A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p=0.1),
     A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25),
     A.Blur(blur_limit=3,p=0.2),
     #ToTensorV2(),
     ]
    )





class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
    
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients,n_heads):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()



        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.Final_psi= nn.Sequential(
            nn.Conv2d(n_heads, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            #nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi1 = self.relu(g1 + x1)
        psi1 = self.psi(psi1) ## [Att5: 1,1,32,32]
        #out1 = skip_connection * psi1  ### (Att5: 1,512,32,32,   A)


        g2 = self.W_gate(gate)
        x2 = self.W_x(skip_connection)
        psi2 = self.relu(g2 + x2)
        psi2 = self.psi(psi2)

        g3 = self.W_gate(gate)
        x3 = self.W_x(skip_connection)
        psi3 = self.relu(g3 + x3)
        psi3 = self.psi(psi3)

        g4 = self.W_gate(gate)
        x4 = self.W_x(skip_connection)
        psi4 = self.relu(g4 + x4)
        psi4 = self.psi(psi4)

        g5 = self.W_gate(gate)
        x5 = self.W_x(skip_connection)
        psi5 = self.relu(g5 + x5)
        psi5 = self.psi(psi5)

        g6 = self.W_gate(gate)
        x6 = self.W_x(skip_connection)
        psi6 = self.relu(g6 + x6)
        psi6 = self.psi(psi6)


        #out2 = skip_connection * psi2

        PSI=torch.cat((psi1,psi2,psi3,psi4,psi5,psi6),1)


        PSI_final=self.Final_psi(PSI)
        Output=skip_connection*PSI_final
        return Output
    
class AttentionUNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256,n_heads=6)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128,n_heads=6)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64,n_heads=6)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32,n_heads=6)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.Activation_lastlayer=nn.Sigmoid()

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        outputs=self.Activation_lastlayer(out)

        return outputs
    
    
model = AttentionUNet()
model = model.to(device)    




summary(model,(3,256,256))


# input_test=torch.rand(1,3,256,256).to(device)
# output_test=model(input_test)
# print(output_test.shape)


Model_path='/home/aniwat/Desktop/Research/Deep_learning_project/MPI_image_segmentation/Model_chracterizaton/Multi_head_Unet_6blocks.pth'
model.load_state_dict(torch.load(Model_path))

Test_IMG_DIR='/home/aniwat/Desktop/Research/Deep_learning_project/MPI_image_segmentation/Testing_data/Images/'
Test_MASK_DIR='/home/aniwat/Desktop/Research/Deep_learning_project/MPI_image_segmentation/Testing_data/Masks/'

Testdata=ImageData(Test_IMG_DIR,Test_MASK_DIR)
print(len(Testdata))



import tifffile as tiff
#from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, jaccard_score

def highlight(row):
    df = lambda x: ['background: #CCCCFF' if x.name in row
                        else '' for i in x]
    return df

def precision_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 3)

def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 3)

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)


def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places

def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)

Testdata=ImageData(Test_IMG_DIR,Test_MASK_DIR)
print(len(Testdata))

from sklearn.metrics import f1_score

def F1_SCORE(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    F1_test=f1_score(y_true_f,y_pred_f)


    return F1_test



Precision_All=[]
Recall_All=[]
Acc_All=[]
Dice_All=[]
ioU_All=[]

Sum_IoU=0;
Sum_F1=0;

for i in range(len(Testdata)):
    #print(i)
    X,Y=Testdata[i]
    Y.size()
    inputs_img=X
    X.size()
    X=X.unsqueeze(0)
    X.to(device)
    X.size()
    X=X.cuda()
    preditction=model(X)
    preditected_output=preditction.cpu().detach().numpy()
    preditected_output=preditected_output.squeeze(1)
    preditected_output_final=(preditected_output>0.80)

    Ground_truth=Y.cpu().detach().numpy()
    Ground_truth.squeeze()
    Ground_truth=(Ground_truth>0.9)



    Ground_truth_F1=Ground_truth.astype(int)
    preditected_output_F1=preditected_output_final.astype(int)

    precision=precision_score_(Ground_truth_F1,preditected_output_F1)
    Precision_All.append(precision)


    recall=recall_score_(Ground_truth_F1,preditected_output_F1)
    Recall_All.append(recall)

    Acc=accuracy(Ground_truth_F1,preditected_output_F1)
    Acc_All.append(Acc)

    dice=dice_coef(Ground_truth_F1,preditected_output_F1)
    Dice_All.append(dice)


    ioU=iou(Ground_truth_F1,preditected_output_F1)
    ioU_All.append(ioU)


print("Accuracy",np.mean(Acc_All))
print(np.std(Acc_All))

print("Precision",np.mean(Precision_All))
print(np.std(Precision_All))

print("Recall",np.mean(Recall_All))
print(np.std(Recall_All))

print("Dice",np.mean(Dice_All))
print(np.std(Dice_All))


print("IoU",np.mean(ioU_All))
print(np.std(ioU_All))


X,Y=Testdata[4]

Y.size()
inputs_img=X
X.size()
X=X.unsqueeze(0)
X.to(device)
X.size()

X=X.cuda()

preditction=model(X)
preditected_output=preditction.cpu().detach().numpy()
preditected_output=preditected_output.squeeze()
preditected_output_final=(preditected_output>0.90)
plt.figure(1),plt.imshow(preditected_output_final),
plt.axis('off')

plt.figure(0),plt.imshow(inputs_img.permute(1,2,0).cpu()),
plt.axis('off')


plt.figure(2),plt.imshow(np.squeeze(Y.cpu(),0))
plt.axis('off')



model.eval()
model_children = list(model.children())
print([n for n, _ in model.named_children()])

model.Att5.Final_psi
target_layers = [model.Att5.Final_psi[-1]]  
  
  
# defines two global scope variables to store our gradients and activations
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Gradients size: {gradients[0].size()}')
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Activations size: {activations.size()}')
  
backward_hook = model.Att2.Final_psi[1].register_full_backward_hook(backward_hook)
forward_hook = model.Att2.Final_psi[1].register_forward_hook(forward_hook)



# backward_hook = model.Att4.register_full_backward_hook(backward_hook)
# forward_hook = model.Att4.register_forward_hook(forward_hook)


X,Y=Testdata[4]

Y.size()
inputs_img=X
X.size()
X=X.unsqueeze(0)
X.to(device)
X.size()
X=X.cuda()
preditction=model(X).mean().backward()



# pool the gradients across the channels
pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])



import torch.nn.functional as F
import matplotlib.pyplot as plt

# weight the channels by corresponding gradients
for i in range(activations.size()[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
heatmap = F.relu(heatmap)

# normalize the heatmap
heatmap /= torch.max(heatmap)
heatmap=heatmap.cpu().detach().numpy()
# draw the heatmap
plt.matshow(heatmap)
plt.axis('off')


from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL

# Create a figure and plot the first image
fig, ax = plt.subplots()
ax.axis('off') # removes the axis markers

# First plot the original image
ax.imshow(to_pil_image(inputs_img, mode='RGB'))

# Resize the heatmap to the same size as the input image and defines
# a resample algorithm for increasing image resolution
# we need heatmap.detach() because it can't be converted to numpy array while
# requiring gradients
overlay = to_pil_image(heatmap, mode='F').resize((256,256), resample=PIL.Image.BICUBIC)

# Apply any colormap you want
cmap = colormaps['jet']
overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

# Plot the heatmap on the same axes,
# but with alpha < 1 (this defines the transparency of the heatmap)
ax.imshow(overlay, alpha=0.6, interpolation='nearest')
plt.axis('off')
# Show the plot
plt.show()


































# from sklearn.metrics import f1_score

# def F1_SCORE(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()

#     F1_test=f1_score(y_true_f,y_pred_f)


#     return F1_test




# Sum_IoU=0;
# Sum_F1=0;

# for i in range(len(Testdata)):
#     print(i)
#     X,Y=Testdata[i]
#     Y.size()
#     inputs_img=X
#     X.size()
#     X=X.unsqueeze(0)
#     X.to(device)
#     X.size()
#     X=X.cuda()
#     preditction=model(X)
#     preditected_output=preditction.cpu().detach().numpy()
#     preditected_output=preditected_output.squeeze(1)
#     preditected_output_final=(preditected_output>0.80)

#     Ground_truth=Y.cpu().detach().numpy()
#     Ground_truth.squeeze()
#     Ground_truth=(Ground_truth>0.9)



#     Ground_truth_F1=Ground_truth.astype(int)
#     preditected_output_F1=preditected_output_final.astype(int)

#     F1_test=F1_SCORE(Ground_truth_F1,preditected_output_F1)
#     print("F1:",F1_test)
#     Sum_F1=Sum_F1+F1_test



#     intersection=np.logical_and(Ground_truth, preditected_output_final)
#     union = np.logical_or(Ground_truth, preditected_output_final)
#     iou_score = np.sum(intersection) / np.sum(union)
#     print("IoU:",iou_score)
#     Sum_IoU=Sum_IoU+iou_score;


# Avg_IoU=Sum_IoU/len(Testdata);
# Avg_F1=Sum_F1/len(Testdata);

# print("AVG_IoU ",Avg_IoU)
# print("AVG_F1 ",Avg_F1)


































