#!/usr/bin/env python
# coding: utf-8

from keras import backend as K

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


#================================================================================
# without mask version
#================================================================================

def dot_attention(self, tensor_left, tensor_right):
    """
    Compute the attention between elements of two sentences with the dot
    product.
    """
    attn_weights = K.batch_dot(x=tensor_left,
                               y=K.permute_dimensions(tensor_right,
                                                      pattern=(0, 2, 1)))
    return K.permute_dimensions(attn_weights, (0, 2, 1))


def fc_attention(self, tensor_left, tensor_rights):
    """
    Compute the attention between elements of two sentences with the fully-connected network.
    """
    tensor_left = K.expand_dims(tensor_left, axis=2)
    tensor_right = K.expand_dims(tensor_right, axis=1)
    tensor_left = K.repeat_elements(tensor_left, tensor_right.shape[2], 2)
    tensor_right = K.repeat_elements(tensor_right, tensor_left.shape[1], 1)
    tensor_merged = K.concatenate([tensor_left, tensor_right], axis=-1)
    middle_output = Dense(128, activation='tanh')(tensor_merged)
    attn_weights = Dense(128)(tensor_merged)
    attn_weights = K.squeeze(attn_weights)

    return attn_weights


def soft_alignment(self, attn_weights, tensor_to_align):
    """
    Compute the soft alignment.
    """
    # Subtract the max. from the attention weights to avoid overflows.
    exp = K.exp(attn_weights - K.max(attn_weights, axis=-1, keepdims=True))
    exp_sum = K.sum(exp, axis=-1, keepdims=True)
    softmax = exp / exp_sum

    return K.batch_dot(softmax, tensor_to_align)

