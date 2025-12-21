import torch

class ConfigUNet:
    PROJECT_NAME = "CycleGAN_Painter_UNET"  
    CHECKPOINT_DIR = "/content/drive/MyDrive/Painter_Assignment/checkpoints_unet" 
    MODEL_TYPE = "unet" 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_MONET = "/content/dataset/monet_jpg"
    TRAIN_PHOTO = "/content/dataset/photo_jpg"
    
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    IMG_SIZE = 256
    LR = 0.0002
    B1 = 0.5
    B2 = 0.999
    EPOCHS = 30           
    
    LAMBDA_CYCLE = 10.0
    LAMBDA_ID = 5.0
    
    SAVE_EPOCH_FREQ = 5
    LOAD_MODEL = False    
    START_EPOCH = 0