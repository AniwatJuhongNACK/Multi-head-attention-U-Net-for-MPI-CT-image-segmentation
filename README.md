#Code for the  Multi-head-attention-U-Net-for-MPI-CT-image-segmentation manuscript 

## Abstract
Magnetic particle imaging (MPI) is an emerging non-invasive molecular imaging modality with high sensitivity and specificity, exceptional linear quantitative ability, and potential for successful applications in clinical settings. Computed tomography (CT) is typically combined with the MPI image to obtain more anatomical information. Herein, we present a deep learning-based approach for MPI-CT image segmentation. The proposed deep learning model exploits the advantages of the multi-head attention mechanism and the U-Net model to perform segmentation on the MPI-CT images, showing superb results. In addition, we characterized the model with a different number of attention heads to explore the optimal number for our custom MPI-CT dataset


## Weights of the multi-head attention U-Net models
Pre-trained models can be downloaded by following the link below.
https://drive.google.com/drive/folders/1AwPwklWs525l_V6tcA0HeRdHKLOYX8hK?usp=drive_link



## Model characterization

![image](https://github.com/AniwatJuhongNACK/Multi-head-attention-U-Net-for-MPI-CT-image-segmentation/assets/113541987/0c31ac38-35be-4706-9082-beb756361797)
The performance of models with the different attention heads (Dice/IoU scores vs the number of attention heads plot). 






![Fig8](https://github.com/AniwatJuhongNACK/Multi-head-attention-U-Net-for-MPI-CT-image-segmentation/assets/113541987/afd9e6c9-b1cb-48be-bc3f-90386eb30a64)
Visualization semantic segmentation results of the proposed model (4-head attention model) compared to traditional U-net, Trans U-net and the Attention U-Net (1-head attention model). 
