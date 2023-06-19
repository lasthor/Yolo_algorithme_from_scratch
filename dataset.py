from torch.utils.data import Dataset
import cv2
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader




class Data_YOLO(Dataset):
    def __init__(self,label_dir,image_dir,S=7 , C=2 ,transform=None,target_transform=None):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.S = S
        self.C = C
    
    def __len__(self):
        return len(os.listdir(self.label_dir))
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir,os.listdir(self.image_dir)[idx])
        label_path = os.path.join(self.label_dir,os.listdir(self.label_dir)[idx])


        boxes = []
        #this if we have one box per image 
        with open(label_path,'r') as file:
            data = file.read()
            data = data.split()
            x = float(data[1])
            y = float(data[2])
            w = float(data[3])
            h = float(data[4])

            label = int(data[0])
        file.close()

        boxes.append([label,x,y,w,h])

        image = cv2.imread(image_path)
        boxes = torch.tensor(boxes)
        

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label_matrix = torch.zeros((self.S,self.S,self.C + 5))

        for box in boxes:
            label , x, y , w, h = box.tolist()
            label = int(label)


            #postion of cell row and columen
            i , j = int(self.S* y) , int(self.S*x)
            #
            x_cell , y_cell = self.S * x - j , self.S* y - i
            #
            w_cell , h_cell = (w * self.S,h * self.S)

            if label_matrix[i,j,2] == 0:
               
                box_coordes = torch.tensor(
                    [x_cell,y_cell,w_cell,h_cell]
                )
                label_matrix[i,j,2] = 1
                label_matrix[i,j,3:7] =  box_coordes
                label_matrix[i,j,label] = 1

        return image/255  , label_matrix