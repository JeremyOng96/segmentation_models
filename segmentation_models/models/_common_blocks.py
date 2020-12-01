from keras_applications import get_submodules_from_kwargs  
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import keras.backend as K

class SelfAttention(keras.layers.Layer):
    def __init__(self,
                 stage=None,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint
        
        self.convk_name = 'decoder_stage{}_k'.format(stage)
        self.convq_name = 'decoder_stage{}_q'.format(stage)
        self.convv_name = 'decoder_stage{}_v'.format(stage)
               
        self.softmax = layers.Activation('softmax')
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        
        self._shape = input_shape
        _, self.h, self.w, self.filters = input_shape
        
        self.conv_k = layers.Conv2D(self.filters//8, 1, use_bias=False, kernel_initializer='he_normal',name=self.convk_name)
        self.conv_q = layers.Conv2D(self.filters//8, 1, use_bias=False, kernel_initializer='he_normal',name=self.convq_name)
        self.conv_v = layers.Conv2D(self.filters, 1, use_bias=False, kernel_initializer='he_normal',name=self.convv_name)
        
        self.built = True

    def call(self, input_tensor):

        self.k = self.conv_k(input_tensor)
        self.q = self.conv_q(input_tensor)
        self.v = self.conv_v(input_tensor)
        self.k = K.reshape(self.k,(-1,self.h*self.w,self.filters//8)) # [B,HW,f]
        self.q = tf.transpose(K.reshape(self.q, (-1, self.h*self.w, self.filters // 8)), (0, 2, 1))
        self.logits = K.batch_dot(self.k, self.q)
        self.xd = self.softmax(self.logits)
        self.v = K.reshape(self.v, (-1, self.h*self.w, self.filters))
        self.attn = K.batch_dot(self.xd, self.v) # [B,Hw,f]
        self.attn = K.reshape(self.attn, (-1, self.h, self.w, self.filters))

        self.out = self.gamma*self.attn + input_tensor
        return self.out
    
    def get_config(self):
        config = {
            "gamma" : self.gamma,
            "k" : self.k,
            "q" : self.q,
            "v" : self.v,
            "attn" : self.attn,
            "w": self.xd,
            "out" : self.out
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper

def Conv2dBn_P(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm and Parameterized ReLU"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + 'PReLU'

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.PReLU(name=act_name)(x)

        return x

    return wrapper
