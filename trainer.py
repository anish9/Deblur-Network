from network import RRDNSR,deblur
from datagen import flow
from config import param_maps
from helpers import SnapshotCallbackBuilder,TensorBoard,LOSSES,PSNRLoss

TRAIN = flow("blur_seg/","train_lr","train_hr",batch=1,size1=0,size2=0,augument=False)
VAL = flow("blur_seg/","val_lr","val_hr",batch=1,size1=0,size2=0,augument=False)



snap_build = SnapshotCallbackBuilder("fas_blur.h5",param_maps["epochs"],param_maps["snaps"],init_lr=param_maps["lr"]) 
calls = snap_build.get_callbacks()
tensorboard = TensorBoard(log_dir=param_maps["tb_writes"])
calls.append(tensorboard)



"""trainer"""

MODEL = deblur(upsample=param_maps["upsample_config"],rdb_depth=param_maps["Depth"])

print(".......Trainer_Initialized........")
print("config...")
print(param_maps)
print("________________________________")
MODEL.compile(optimizer="Adam",loss=LOSSES,metrics=[PSNRLoss])
MODEL.fit_generator(TRAIN,steps_per_epoch=param_maps["train_count"],validation_data=VAL,validation_steps=param_maps["val_count"],epochs=param_maps["epochs"],callbacks=calls)
