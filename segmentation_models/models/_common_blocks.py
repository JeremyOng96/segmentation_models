from keras_applications import get_submodules_from_kwargs  
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


def GCN(out_c=21,k=7):
	pad_h = (int((k-1)/2),0)
    	pad_w = (0,int((k-1)/2))
	
	def layer(input_tensor):
		x_l = layers.ZeroPadding2D(padding=pad_h)(input_tensor)
		x_l = layers.Conv2D(out_c,(k,1))(x_l)
		x_l = layers.ZeroPadding2D(padding=pad_w)(x_l)
		x_l = layers.Conv2D(out_c,(1,k))(x_l)

		x_r = layers.ZeroPadding2D(padding=pad_w)(input_tensor)
		x_r = layers.Conv2D(out_c,(1,k))(x_r)
		x_r = layers.ZeroPadding2D(padding=pad_h)(x_r)
		x_r = layers.Conv2D(out_c,(k,1))(x_r)

		x = layers.Add()([x_l,x_r])
		
		return x
	
	return layer

def BR(out_c=21):
	def layer(input_tensor):
		
		residual = layers.Conv2D(out_c,3,padding='same')(input_tensor)
		residual = layers.Activation('relu')(residual)
		residual = layers.Conv2D(out_c,3,padding='same')(residual)

		x = layers.Add()([residual,input_tensor])

		return x
	
	return layer

def cbam_block(ratio=16, **kwargs):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	def layer(input_tensor):
		output = channel_attention(input_tensor, ratio)
		output = spatial_attention(output)
    
		return output
	return layer

def channel_attention(input_tensor, ratio=16):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_tensor.shape[channel_axis]
	
	shared_layer_one = layers.Dense(channel//ratio,activation='relu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')    
	shared_layer_two = layers.Dense(channel,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')	
    
    # Use average pooling layers
	avg_pool = layers.GlobalAveragePooling2D()(input_tensor)    
	avg_pool = layers.Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
    # Use max pooling layers
	max_pool = layers.GlobalMaxPooling2D()(input_tensor)
	max_pool = layers.Reshape((1,1,channel))(max_pool) # The reshaping is used for python broadcasting
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = layers.Add()([avg_pool,max_pool])
	cbam_feature = layers.Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)
	
	return layers.Multiply()([input_tensor,cbam_feature]) # Output of F'

def spatial_attention(input_tensor):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_tensor.shape[1]
		cbam_feature = layers.Permute((2,3,1))(input_tensor)
	else:
		channel = input_tensor.shape[-1]
		cbam_feature = input_tensor
	
	avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = layers.Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)
		
	return layers.Multiply()([input_tensor,cbam_feature]) # Output of F''



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

        x = layers.PReLU(shared_axes = [1,2], name=act_name)(x)

        return x

    return wrapper
