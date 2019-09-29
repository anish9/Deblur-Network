from layers import *


def deblur(upsample=2,rdb_depth=None):
    
    inpu = Input(shape=(None,None,3))
    xin1 = conv_global(inpu,1)
    xin2 = conv_global(xin1,2,stride=2)
    
    global_list = [xin2]
    for e in range(1,rdb_depth+1):
        r1 = RDBlocks(xin2,"rdbs"+str(e))
        global_list.append(r1)


    concs = Concatenate(axis=-1)(global_list)    
    concs = Conv2D(64,1,padding="same",name="glist_ups_conv1")(concs)
    
    concs = PReLU(shared_axes=[1,2])(concs)
    concs = Conv2D(64,3,padding="same",name="glist_ups_conv3")(concs)
    
    concs = tf_subpixel_conv(concs,2,256)
    concs = add([concs,xin1])
    
    global_merge = concs
    upsample_seg = tf_subpixel_conv(global_merge,2,128)
    if upsample == 2: 
        s2 = upsample_seg
        s2 = PReLU(shared_axes=[1,2])(s2)
        fout = Conv2D(3,9,padding="same",activation="linear")(s2)
    if upsample == 4:
        upsample_seg = tf_subpixel_conv(upsample_seg,2,32)
        fout = Conv2D(3,9,padding="same",activation="linear")(upsample_seg)
    network = Model(inputs=inpu,outputs=fout,name="mod_RDB_"+str(upsample)+str(rdb_depth))
    return network
