"""config file --  
   ---default with learning rate annelaing
   ---create custom to turn annealer OFF"""


Total_epochs        =   60 #total epochs to run
Number_of_snapshots =   2 #number of restarts to train efficient (Kinda Regularizer)
Initial_LR          =   1e-4 #strating learning rate
RDB_DEPTH           =   8 #Block (causes memory error if <= 22 --GPU dependent)
upsample_dim        =   2 #upsample the output limit
Train_count         =   463 #number of training images
Val_count           =   51 #number of val images
scale               =   250 #scale the image by (subjected to training format)
TENSORBOARD_DIR     =   "./tensorboard_files" #directory to write logs


param_maps = {"epochs":Total_epochs,
              "snaps":Number_of_snapshots,
              "lr":Initial_LR,
              "Depth":RDB_DEPTH,
              "upsample_config":upsample_dim,
              "scale":scale,
              "train_count":Train_count,
              "val_count":Val_count,
              "tb_writes":TENSORBOARD_DIR}
