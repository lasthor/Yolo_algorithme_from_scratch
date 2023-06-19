import torch 
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
     
    def __init__(self, S=7, B=2, C=2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss()

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward( self, predictions, target):
        predictions = predictions.reshape(-1,self.S , self.S , self.B * 5 + self.C)

        iou_box1 = intersection_over_union(predictions[...,3:7],target[...,3:7])
        iou_box2 = intersection_over_union(predictions[...,8:12],target[...,3:7])

        ious = torch.cat([iou_box2.unsqueeze(0) ,iou_box1.unsqueeze(0)],dim=0)
        #best_box 0 if the iou_box1 is biger than iou_box2 else 1 , best_box helps to know wich box is resposnbel
        iou_max ,best_box = torch.max(ious,dim=0)

        #this for existing object in the cell of not
        exist_obj = target[...,2].unsqueeze(3)

        '''At training time we only want one bounding box predictor
        to be responsible for each object. We assign one predictor
        to be “responsible” for predicting an object based on which
        prediction has the highest current IOU with the ground
        truth. This leads to specialization between the bounding box
        predictors.'''


        '''Note that the loss function only penalizes classification
        error if an object is present in that grid cell (hence the conditional class probability discussed earlier). 
        It also only penalizes bounding box coordinate error if that predictor is
        “responsible” for the ground truth box (i.e. has the highest
        IOU of any predictor in that grid cell).'''


        #------------------ localistion loss ----------------------------#

        box_predictions = exist_obj * (
            best_box * predictions[...,8:12]
            + (1 - best_box) *  predictions[...,3:7]
        )

        box_target = exist_obj * target[...,3:7]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4]  + 1e-6))

        box_target = torch.sqrt(box_target)

        box_loss = self.mse(
            torch.flatten(box_predictions,end_dim=-2),
            torch.flatten(box_target , end_dim=-2)
        )

        #----------------confidence loss for object -------------------##
        conf_obj_pred = exist_obj * (
            best_box * predictions[...,7:8] + (1 - best_box) * predictions[...,2:3]
        )
        conf_obj_target = exist_obj * target[...,2:3]

        conf_obj_exist_loss = self.mse(
            torch.flatten(conf_obj_pred , end_dim=-2),
            torch.flatten(conf_obj_target , end_dim=-2)

        )

        #----------------confidence loss for no object -------------------##
        #(N , S ,S , 1) --->(N * S * S ,1)
        conf_noobj_pred = (1 - exist_obj) * (
            (1 - best_box) * predictions[...,7:8] + best_box  * predictions[...,2:3]
        )

        no_object_loss = self.mse(
            torch.flatten((1 - best_box) * predictions[..., 2:3], start_dim=1),
            torch.flatten((1 - best_box) * target[..., 2:3], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - best_box) * predictions[..., 7:8], start_dim=1),
            torch.flatten((1 - best_box) * target[..., 2:3], start_dim=1)
        )
        #----------------class loss -------------------##
        #(N , s ,s ,20)
        pred_class = exist_obj * predictions[...,0:2]
        target_class = exist_obj * target[...,0:2]

        class_loss  = self.mse(
            torch.flatten(pred_class , end_dim = -2),
            torch.flatten(target_class , end_dim = -2 )
        )

        final_loss = (
            self.lambda_coord * box_loss
            + conf_obj_exist_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return final_loss






    