import torch

class Config:
    PROJECT_NAME = "CycleGAN_Painter"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TRAIN_MONET = "/content/dataset/monet_jpg"
    TRAIN_PHOTO = "/content/dataset/photo_jpg"
    CHECKPOINT_DIR = "/content/drive/MyDrive/Painter_Assignment/checkpoints"
    
    BATCH_SIZE = 1      
    NUM_WORKERS = 2     
    IMG_SIZE = 256
    
    LR_G = 0.0002
    LR_D = 0.00005     
    
    B1 = 0.5            
    B2 = 0.999          
    
    LAMBDA_CYCLE = 10.0 # High weight to preserve structure
    LAMBDA_ID = 5.0     # Helps preserve color (usually 0.5 * cycle)
    
    # Instead of 1.0, we will use 0.9 for real labels
    REAL_LABEL_SMOOTH = 0.9

    LOAD_MODEL = True
    START_EPOCH = 5      
    EPOCHS = 10          
    SAVE_EPOCH_FREQ = 1  