import torch

class ConfigAug:
    # --- EXPERIMENT 3 SETTINGS ---
    PROJECT_NAME = "CycleGAN_Painter_AUG"  # ახალი პროექტი WandB-ზე
    CHECKPOINT_DIR = "/content/drive/MyDrive/Painter_Assignment/checkpoints_aug"
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths (შეცვალე თუ სხვაგან გაქვს მონაცემები)
    TRAIN_MONET = "/content/dataset/monet_jpg"
    TRAIN_PHOTO = "/content/dataset/photo_jpg"
    
    # Hyperparameters
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    IMG_SIZE = 256
    LR = 0.0002
    B1 = 0.5
    B2 = 0.999
    
    # Epochs
    EPOCHS = 30
    DECAY_START_EPOCH = 15 # აქედან დაიწყება LR-ის კლება
    
    # Loss Weights
    LAMBDA_CYCLE = 10.0
    LAMBDA_ID = 5.0
    
    # Saving
    SAVE_EPOCH_FREQ = 5   # მოდელის შენახვა ყოველ 5 ეპოქაში
    LOG_IMAGES_FREQ = 1   # სურათების ატვირთვა ყოველ 1 ეპოქაში (რაც ითხოვე)