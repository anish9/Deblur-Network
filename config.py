"""config file --  
   ---default with learning rate annelaing
   ---create custom to turn annealer OFF"""


Total_epochs        =   60
Number_of_snapshots =   2 
Initial_LR          =   1e-4
RDB_DEPTH           =   8
upsample_dim        =   2
Train_count         =   463
Val_count           =   51
scale               =   350
TENSORBOARD_DIR     =   "./tensorboard_files"


param_maps = {"epochs":Total_epochs,
              "snaps":Number_of_snapshots,
              "lr":Initial_LR,
              "Depth":RDB_DEPTH,
              "upsample_config":upsample_dim,
              "scale":scale,
              "train_count":Train_count,
              "val_count":Val_count,
              "tb_writes":TENSORBOARD_DIR}
