#!/usr/bin/env python
# coding: utf-8


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


def dual_attention_alignment(tensor_left, tensor_right):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def dot_attention(self, tensor_left, tensor_right):
    """
    Compute the attention between elements of two sentences with the dot
    product.
    """
    attn_weights = K.batch_dot(x=tensor_left,
                               y=K.permute_dimensions(tensor_right,
                                                      pattern=(0, 2, 1)))
    return K.permute_dimensions(attn_weights, (0, 2, 1))


def soft_alignment(self, attention_weight, tensor_to_align):
    """
    Compute the soft alignment.
    """
    # Subtract the max. from the attention weights to avoid overflows.
    exp = K.exp(attention_weight - K.max(attention_weight, axis=-1, keepdims=True))
    exp_sum = K.sum(exp, axis=-1, keepdims=True)
    softmax = exp / exp_sum

    return K.batch_dot(softmax, tensor_to_align)
