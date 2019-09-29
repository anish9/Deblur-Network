import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,RemoteMonitor,LearningRateScheduler,TensorBoard


class SnapshotCallbackBuilder:
    def __init__(self, save_name,nb_epochs, nb_snapshots, init_lr=0.1):
        self.name = save_name
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def PSNRLoss(y_true, y_pred,max_pixel=1.0):
        return 10* K.log(max_pixel**2 /(K.mean(K.square(y_pred - y_true))))
    
    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            ModelCheckpoint(self.name,monitor="PSNRLoss", 
                                   mode = 'max', save_weights_only = True,save_best_only=True, verbose=1),
            LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)



def perceptual_loss(y_true, y_pred):
    image_shape = (None,None,3)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def LOSSES(y_true,y_pred):
    l = tf.keras.losses.mse(y_true[:,:,:],y_pred[:,:,:])
    adds = perceptual_loss(y_true,y_pred)
    total_loss = l+adds
    return total_loss

def PSNRLoss(y_true, y_pred,max_pixel=1.0):
        return 10* K.log(max_pixel**2 /(K.mean(K.square(y_pred - y_true))))