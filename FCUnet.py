import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, BatchNormalization,Input,concatenate
from tensorflow.keras.models import Model


def create_new_model():
    input_vec = Input(shape=(86,))
    #encoder
    x0=Activation('relu')(BatchNormalization()(Dense(1024)(input_vec)))
    x1=Activation('relu')(BatchNormalization()(Dense(1024)(x0)))
    x2=Activation('relu')(BatchNormalization()(Dense(1024)(x1)))
    x3=Activation('relu')(BatchNormalization()(Dense(1024)(x2)))

    x4=Activation('relu')(BatchNormalization()(Dense(512)(x3)))
    x5=Activation('relu')(BatchNormalization()(Dense(512)(x4)))
    x6=Activation('relu')(BatchNormalization()(Dense(512)(x5)))

    x7=Activation('relu')(BatchNormalization()(Dense(256)(x6)))
    x8=Activation('relu')(BatchNormalization()(Dense(256)(x7)))
    x9=Activation('relu')(BatchNormalization()(Dense(256)(x8)))

    x10=Activation('relu')(BatchNormalization()(Dense(128)(x9)))
    x11=Activation('relu')(BatchNormalization()(Dense(128)(x10)))
    x12=Activation('relu')(BatchNormalization()(Dense(128)(x11)))

    x13=Activation('relu')(BatchNormalization()(Dense(64)(x12)))
    x14=Activation('relu')(BatchNormalization()(Dense(64)(x13)))
    x15=Activation('relu')(BatchNormalization()(Dense(64)(x14)))

    #decoder
    merge1=concatenate([x12, x15], axis=-1)
    x16=Activation('relu')(BatchNormalization()(Dense(128)(merge1)))
    x17=Activation('relu')(BatchNormalization()(Dense(128)(x16)))
    x18=Activation('relu')(BatchNormalization()(Dense(128)(x17)))

    merge2=concatenate([x9, x18], axis=-1)
    x19=Activation('relu')(BatchNormalization()(Dense(256)(merge2)))
    x20=Activation('relu')(BatchNormalization()(Dense(256)(x19)))
    x21=Activation('relu')(BatchNormalization()(Dense(256)(x20)))

    merge3=concatenate([x6, x21], axis=-1)
    x22=Activation('relu')(BatchNormalization()(Dense(256)(merge3)))
    x23=Activation('relu')(BatchNormalization()(Dense(256)(x22)))
    x24=Activation('relu')(BatchNormalization()(Dense(128)(x23)))

    merge4=concatenate([x3, x24], axis=-1)
    x25=Activation('relu')(BatchNormalization()(Dense(64)(merge4)))
    x26=Activation('relu')(BatchNormalization()(Dense(32)(x25)))
    x27=Activation('relu')(BatchNormalization()(Dense(16)(x26)))

    x28=Dense(1, activation='linear')(x27) #53

    model=Model(input_vec,x28)
    # model.summary()
    return model
