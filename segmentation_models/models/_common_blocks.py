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
    
class Scale(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.alpha = self.add_weight(name='alpha',shape=(1,),initializer=tf.keras.initializers.Zeros())
        self.beta = self.add_weight(name='beta',shape=(1,),initializer=tf.keras.initializers.Ones())


    def call(self, inputs):
        a, b = inputs
        ratio = self.alpha/(self.alpha+self.beta)
        return tf.add(ratio*a,(1-ratio)*b)
    
    def get_config(self):
        config = {
            "alpha" : self.alpha,
            "beta" : self.beta,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadAttention2D(keras.layers.Layer):
    def __init__(self, depth_k, depth_v, num_heads, relative, **kwargs):
        """
        Applies attention augmentation on a convolutional layer output.
        Args:
        depth_k : Depth of k (int)
        depth_v : Depth of v (int)
        num_heads : Num attention heads (int)
        relative: Whether to use relative embeddings (bool)
        Returns:
        Output of tensor shape
        [Batch, Height, Width, Depth_v]
        """
        super(MultiHeadAttention2D, self).__init__(**kwargs)

        # Performs checking for MHA assumptions
        if depth_k % num_heads != 0:
            raise ValueError('`depth_k` (%d) is not divisible by `num_heads` (%d)' % (depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError('`depth_v` (%d) is not divisible by `num_heads` (%d)' % (depth_v, num_heads))

        if depth_k // num_heads < 1.:
            raise ValueError('depth_k / num_heads cannot be less than 1 ! '
                              'Given depth_k = %d, num_heads = %d' % (
                              depth_k, num_heads))

        if depth_v // num_heads < 1.:
            raise ValueError('depth_v / num_heads cannot be less than 1 ! '
                              'Given depth_v = %d, num_heads = %d' % (
                                  depth_v, num_heads))


        # Initialize necessary variables
        self.dk = depth_k
        self.dv = depth_v
        self.nh = num_heads
        self.relative = relative
        self.dkh = self.dk // self.nh
        self.dvh = self.dv // self.nh

        # Initialize the necessary layers

    def build(self, input_shape):
        self._shape = input_shape
        self.B, self.H, self.W, self.d = input_shape

        if self.relative:
            self.rel_embeddings_w = self.add_weight('rel_embeddings_w',shape=(2 * self.W - 1, self.dkh),initializer=tf.keras.initializers.RandomNormal(stddev=self.dkh ** -0.5),trainable = True)
            self.rel_embeddings_h = self.add_weight('rel_embeddings_h',shape=(2 * self.H - 1, self.dkh),initializer=tf.keras.initializers.RandomNormal(stddev=self.dkh ** -0.5),trainable = True)


    def call(self,inputs,**kwargs):
        # Input is the KQV matrix
        # dk = 24, dv = 24
        flatten_hw = lambda x,d: tf.reshape(x, [-1, self.nh, self.H*self.W,d])
        
        # Compute q, k, v matrix 
        k, q, v = tf.split(inputs,[self.dk,self.dk,self.dv],axis = -1) # [1,16,16,24] for k q and v
        # Rescale the value of q
        q *= (self.dkh ** -0.5)

        # Splits a tensor with shape [batch, num_heads, height, width, channels] 
        # to a tensor with shape [batch,num_heads,height,width,channels/num_heads]

        q = self.split_heads_2d(q,self.nh)
        k = self.split_heads_2d(k,self.nh)
        v = self.split_heads_2d(v,self.nh)
        # [B,Nh,HW,HW]
        logits = tf.matmul(flatten_hw(q,self.dkh),flatten_hw(k,self.dkh),transpose_b= True)

        if self.relative:
            rel_logits_h, rel_logits_w = self.relative_logits(q,self.H,self.W,self.nh)

            logits += rel_logits_h
            logits += rel_logits_w

        weights = tf.nn.softmax(logits)
        attn_out = tf.matmul(weights, flatten_hw(v,self.dvh))
        attn_out = tf.reshape(attn_out,[-1,self.nh,self.H,self.W,self.dvh])
        attn_out = self.combine_heads_2d(attn_out) # Output shape = [B,H, W, dv]

        return attn_out

    def shape_list(self,x):
        """
        Returns a list of dimensions
        Arguments:
        x : A keras tensor    
        """

        static = x.get_shape().as_list()
        shape = tf.shape(x)
        ret = []
        for i, static_dim in enumerate(static):
            dim = static_dim or shape[i]
            ret.append(dim)

        return ret


    def split_heads_2d(self,inputs,Nh):
        """ Split channels into multiple heads """
        B, H, W, d = self.shape_list(inputs)
        ret_shape = [B,H,W,Nh,d//Nh]
        split = tf.reshape(inputs, ret_shape)
        return tf.transpose(split, [0,3,1,2,4])

    def combine_heads_2d(self, inputs):
        """ Combine heads (inverse of split_heads_2d)."""
        transposed = tf.transpose(inputs,[0,2,3,1,4])
        Nh, channels = self.shape_list(transposed)[-2:]
        ret_shape = self.shape_list(transposed)[:-2] + [Nh * channels]
        return tf.reshape(transposed,ret_shape)

    def rel_to_abs(self,x):
        """ Converts tensor from relative to absolute indexing. """
        # [B, Nh, L, 2L-1]
        B, Nh, L, _ = self.shape_list(x)
        # Pad to shift from relative to absolute indexing
        col_pad = tf.zeros((B,Nh,L,1))
        x = tf.concat([x,col_pad],axis = 3)
        flat_x = tf.reshape(x, [B,Nh,L*2*L])
        flat_pad = tf.zeros((B,Nh,L-1))
        flat_x_padded = tf.concat([flat_x,flat_pad],axis = 2)
        # Reshape and slice out the padded elements
        final_x = tf.reshape(flat_x_padded, [B,Nh,L+1,2*L-1])
        final_x = final_x[:,:,:L,L-1:]
        return final_x

    def relative_logits_1d(self,q,rel_k,H,W,Nh,transpose_mask):
        """ Compute relative logits along H or W """

        rel_logits = tf.einsum("bhxyd,md->bhxym",q,rel_k)
        # Collapse height and heads
        rel_logits = tf.reshape(rel_logits, [-1,Nh*H,W,2 * W-1])
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it and tile height times
        rel_logits = tf.reshape(rel_logits, [-1, Nh,H,W,W])
        rel_logits = tf.expand_dims(rel_logits, axis = 3)
        rel_logits = tf.tile(rel_logits,[1,1,1,H,1,1])
        # Reshape for adding to the logits
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        rel_logits = tf.reshape(rel_logits, [-1,Nh,H*W,H*W])
        return rel_logits

    def relative_logits(self,q,H,W,Nh):
        """ Compute relative logits """

        rel_logits_w = self.relative_logits_1d(q,self.rel_embeddings_w,H,W,Nh,[0,1,2,4,3,5])
        rel_logits_h = self.relative_logits_1d(q,self.rel_embeddings_h,W,H,Nh,[0,1,4,2,5,3])

        # [B, Nh, HW, HW]
        return rel_logits_h, rel_logits_w

    def get_config(self):
        config = {
            "dk" : self.dk,
            "dv" : self.dv,
            "nh" : self.nh,
            "relative" : self.relative,
            "dkh" : self.dkh,
            "dvh" : self.dvh
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def MultiHeadAttention( f_out,
                        stage = None,
                        Rk=1,
                        Rv=1,
                        Nh=8,
                        relative=False):
    convkqv_name = 'decoder_stage{}_kqv'.format(stage)
    convprojection_name = 'decoder_stage{}_projection'.format(stage)

    def layer(input_tensor): 
        ei = lambda x : int(np.ceil(x/Nh)*Nh)
        dk = ei(f_out*Rk)
        dv = ei(f_out*Rv)
        
        # Form the MHA matrix
        kqv = layers.Conv2D(filters = 2*dk + dv,kernel_size = 1,padding = "same",kernel_initializer="he_normal",name=convkqv_name)(input_tensor)
        attn_out = MultiHeadAttention2D(dk,dv,Nh,relative)(kqv)
        # Projection of MHA
        attn_out = layers.Conv2D(f_out,1,name=convprojection_name )(attn_out)
        return attn_out
    
    return layer

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
