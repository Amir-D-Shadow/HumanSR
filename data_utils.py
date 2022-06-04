import os 
import numpy as np
import dlib
import torch
from PIL import Image
from utils.align_all_parallel import align_face
import shutil
import torchvision.transforms as transforms
import random

def run_alignment(image_path):

  predictor = dlib.shape_predictor(f"{os.getcwd()}/pixel2style2pixel/dlib_model_weights/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor)
  #print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image

def set_data(source_dir=f"{os.getcwd()}/data/train/CelebA-HQ-img",target_dir=f"{os.getcwd()}/data/train/aligned_data"):
    
    os.makedirs(target_dir,exist_ok=True)
    
    img_list = os.listdir(source_dir)
    m = len(img_list)
    i = 0
    for img_name in img_list:
        """
        img = run_alignment(f"{source_dir}/{img_name}")
        
        if isinstance(img,str):
            #shutil.copyfile(f"{source_dir}/{img_name}",f"{target_dir}/{img_name}")
            img = Image.open(f"{source_dir}/{img_name}").resize((256,256))
            img.save(f"{target_dir}/{img_name}")
        else:
            img.save(f"{target_dir}/{img_name}")
        """    
        img = Image.open(f"{source_dir}/{img_name}").resize((256,256))
        crop_size  = min(img.size[0],img.size[1])
        tf = transforms.Compose([transforms.CenterCrop(crop_size)])
        img = tf(img)
        img = img.resize((256, 256))
        img.save(f"{target_dir}/{img_name}")
        if ((i/m)*100) %10 == 0:
            print(f"{(i/m)*100}%")
        i = i + 1
        
class DataLoader:
    
    def __init__(self,ref_dir=f"{os.getcwd()}/data/HD_ref/aligned_data",train_dir=f"{os.getcwd()}/data/train/aligned_data"):
        
        #get training data path
        self.training_img_path = [] #save training data path 
        img_list = os.listdir(train_dir)
        for img_name in img_list:
            self.training_img_path.append(f"{train_dir}/{img_name}")
        
        #get reference data path
        self.ref_data_path = [] #save ref data path
        img_list = os.listdir(ref_dir)
        for img_name in img_list:
            self.ref_data_path.append(f"{ref_dir}/{img_name}")
            
    def load_data(self,batch_size = 8,num_ref_img=32):
        
        for img_lr,img_ori in self.load_train_img(batch_size=batch_size):
            ref_img = self.load_ref_image(num_ref_img=num_ref_img) # (B2,C,H,W)
            yield img_lr,img_ori,ref_img  # (B1,C,H,W) , (B1,C,H,W) , (B2,C,H,W)
            
                
    def load_train_img(self,batch_size=8):
        
        #set up
        original_image = []
        LR_image = []
        random.shuffle(self.training_img_path)
        count = 0
        tf = transforms.Compose([transforms.ToTensor()])
        #get image
        for img_dir in self.training_img_path:
            
            #get original image
            img_ori  = Image.open(img_dir)
            #get low resolution image
            img_lr = img_ori.resize((img_ori.size[0]//8,img_ori.size[1]//8)) #down scale by factor of 8
            #img_lr = img_lr.resize((img_ori.size[0],img_ori.size[1])) 
            
            #transform and save img_ori and img_lr
            img_ori = tf(img_ori)
            original_image.append(img_ori)
            img_lr = tf(img_lr)
            LR_image.append(img_lr)
            
            #update count
            count = count + 1
            #return batch of images
            if count == batch_size:
                original_image = torch.stack(original_image,dim=0).float() # (B,C,H,W)
                LR_image = torch.stack(LR_image,dim=0).float() #(B,C,h,w)
                yield LR_image , original_image 
                #reset params
                count = 0
                original_image = []
                LR_image = []
                
        original_image = torch.stack(original_image,dim=0).float() # (B,C,H,W)
        LR_image = torch.stack(LR_image,dim=0).float() #(B,C,h,w)         
        yield LR_image , original_image 
        
    def load_ref_image(self,num_ref_img=16):
        
        random.shuffle(self.ref_data_path)
        ref_img = []
        tf = transforms.Compose([transforms.ToTensor()])
        for i in range(num_ref_img):
            
            img = Image.open(self.ref_data_path[i])
            #img = tf(img)
            img = tf(img.resize((img.size[0]//8,img.size[1]//8)))
            ref_img.append(img)
            
        ref_img = torch.stack(ref_img,dim=0).float()
        
        return ref_img
                
                    
#test code
if __name__ == "__main__":
    
    #set_data(source_dir=f"{os.getcwd()}/data/HD_ref/images1024x1024",target_dir=f"{os.getcwd()}/data/HD_ref/aligned_data")
    #set_data()
    data_loader = DataLoader()
    #test load ref train image
    """
    i = 1
    for img_lr,img_ori in data_loader.load_train_img(9):
        ref_img = data_loader.load_ref_image(32)
        print(f"ref image shape: {ref_img.shape} low resolution shape: {img_lr.shape}, original image shape:{img_ori.shape} {i}")
        i = i + 1
    """
    #test load data
    i = 1 
    for img_lr,img_ori,img_ref in data_loader.load_data(batch_size=27,num_ref_img=32):
        print(f"ref image shape: {img_ref.shape} low resolution shape: {img_lr.shape}, original image shape:{img_ori.shape} {i}")
        i = i + 1
    
    print("Completed")