import tensorflow as tf
import numpy as np

def Sample_from_Discretized_Mix_Logistic(y, log_scale_min= None):
    '''
    y: [Batch, Time, Dim]
    '''
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    if y.get_shape()[-1] % 3 != 0:
        raise ValueError('The dimension of \'y\' must be a multiple of 3.')
    nr_mix = y.get_shape()[-1] // 3

    logit_probs, means, log_scales = tf.split(y, num_or_size_splits=3, axis= -1)    # [Batch, Time, Dim // 3]
    
    temp = tf.random.uniform(
        shape= tf.shape(logit_probs),
        minval= 1e-5,
        maxval= 1.0 - 1e-5,
        dtype= logit_probs.dtype
        )   # [Batch, Time, Dim // 3]
    temp = logit_probs - tf.math.log(-tf.math.log(temp))    # [Batch, Time, Dim // 3]
    argmax = tf.math.argmax(temp, axis= -1, output_type=tf.int32)   # [Batch, Time]

    one_hot = tf.one_hot(indices= argmax, depth= nr_mix, dtype= y.dtype)    # [Batch, Time, Dim // 3]
    means = tf.reduce_sum(means * one_hot, axis= -1)    # [Batch, Time]
    log_scales = tf.maximum(tf.reduce_sum(log_scales * one_hot, axis= -1), log_scale_min)   # [Batch, Time]

    u = tf.random.uniform(
        shape= tf.shape(means),
        minval= 1e-5,
        maxval= 1.0 - 1e-5,
        dtype= means.dtype
        )   # [Batch, Time]
    x = means + tf.exp(log_scales) * (tf.math.log(u) - tf.math.log(1.0 - u))
    x = tf.clip_by_value(x, clip_value_min= -1.0, clip_value_max= 1.0)

    return x

def Discretized_Mix_Logistic_Loss(
    labels,
    logits,
    classes= 65536,
    log_scale_min= None
    ):
    '''
    labels: [Batch, Time]
    logits: [Batch, Time, Dim]
    '''
    classes = tf.cast(classes, dtype= logits.dtype)

    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    if logits.get_shape()[-1] % 3 != 0:
        raise ValueError('The dimension of \'y\' must be a multiple of 3.')
    nr_mix = logits.get_shape()[-1] // 3

    logit_probs, means, log_scales = tf.split(logits, num_or_size_splits=3, axis= -1)    # [Batch, Time, Dim // 3]
    log_scales = tf.maximum(log_scales, log_scale_min)   # [Batch, Time]

    labels = tf.cast(tf.tile(
        tf.expand_dims(labels, axis= -1),
        [1, 1, means.get_shape()[-1]]
        ), dtype= logits.dtype)
    cetnered_labels = labels - means
    inv_stdv = tf.exp(-log_scales)

    plus_in = inv_stdv * (cetnered_labels + 1 / (classes - 1))
    cdf_plus = tf.math.sigmoid(plus_in)
    min_in = inv_stdv * (cetnered_labels - 1 / (classes - 1))
    cdf_min = tf.math.sigmoid(min_in)

    log_cdf_plus = plus_in - tf.math.softplus(plus_in)
    log_one_minus_cdf_min = -tf.math.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * cetnered_labels
    log_pdf_mid = mid_in - log_scales - 2.0 * tf.math.softplus(mid_in)

    inner_inner_cond = tf.cast(tf.greater(cdf_delta, 1e-5), dtype= logits.dtype)    
    inner_inner_out = \
        inner_inner_cond * tf.math.log(tf.maximum(cdf_delta, 1e-12)) + \
        (1.0 - inner_inner_cond) * (log_pdf_mid - tf.math.log((classes - 1) /2))
    inner_cond = tf.cast(tf.greater(labels, 0.999), dtype= logits.dtype)
    inner_out = \
        inner_cond * log_one_minus_cdf_min + \
        (1.0 - inner_cond) * inner_inner_out
    cond = tf.cast(tf.less(labels, -0.999), dtype= logits.dtype)
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = log_probs + tf.math.log_softmax(logit_probs, -1)
    
    return -tf.reduce_mean(tf.math.reduce_logsumexp(log_probs, axis= -1, keepdims=True))

if __name__ == "__main__":
    labels = tf.convert_to_tensor(np.random.randint(0, 65535, size=(2, 653)).astype(np.float32))
    logits = tf.convert_to_tensor(np.random.rand(2,653,30).astype(np.float32))

    q = Discretized_Mix_Logistic_Loss(labels, logits)
    print(q)