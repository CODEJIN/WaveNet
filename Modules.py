# Refer:
# https://https://github.com/fatchord/WaveRNN

import tensorflow as tf
import numpy as np
import json
from MoL import Sample_from_Discretized_Mix_Logistic, Discretized_Mix_Logistic_Loss

from ProgressBar import progress

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)


class WaveNet(tf.keras.Model):
    def build(self, input_shapes):
        if hp_Dict['WaveNet']['Initial_Filters'] % 2 != 0:
            raise ValueError('Initial filter must be a even.')

        self.layer_Dict = {}

        self.layer_Dict['First'] = tf.keras.Sequential()
        self.layer_Dict['First'].add(tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis= -1)
            ))
        # self.layer_Dict['First'].add(tf.keras.layers.Conv1D(
        #     filters= hp_Dict['WaveNet']['Initial_Filters'],
        #     kernel_size= 1,
        #     strides= 1,
        #     padding= 'causal',
        #     use_bias= True,
        #     kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
        #     bias_initializer= tf.keras.initializers.zeros
        #     ))
        self.layer_Dict['First'].add(Incremental_Conv1D_Causal_WN(
            filters= hp_Dict['WaveNet']['Initial_Filters'],
            kernel_size= 1,
            strides= 1,
            use_bias= True,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            bias_initializer= tf.keras.initializers.zeros
            ))
        for block_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Blocks']):
            for stack_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Stacks_in_Block']):
                self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)] = ResConvGLU(
                    filters= hp_Dict['WaveNet']['Initial_Filters'] // 2,
                    kernel_size= hp_Dict['WaveNet']['ResConvGLU']['Kernel_Size'],
                    skip_out_filters= hp_Dict['WaveNet']['ResConvGLU']['Skip_Out_Filters'],
                    dropout_rate= hp_Dict['WaveNet']['ResConvGLU']['Dropout_Rate'],
                    dilation_rate= 2 ** stack_Index,
                    use_bias= True
                    )

        self.layer_Dict['Last'] = tf.keras.Sequential()
        self.layer_Dict['Last'].add(tf.keras.layers.ReLU())
        self.layer_Dict['Last'].add(tf.keras.layers.Conv1D(
            filters= hp_Dict['WaveNet']['ResConvGLU']['Skip_Out_Filters'],
            kernel_size= 1,
            strides= 1,
            padding= 'same',
            use_bias= True,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            bias_initializer= tf.keras.initializers.zeros
            ))
        self.layer_Dict['Last'].add(tf.keras.layers.ReLU())
        self.layer_Dict['Last'].add(tf.keras.layers.Conv1D(
            filters= hp_Dict['WaveNet']['MoL_Sizes'],
            kernel_size= 1,
            strides= 1,
            padding= 'same',
            use_bias= True,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            bias_initializer= tf.keras.initializers.zeros
            ))
        
        self.layer_Dict['Local_Condition_Upsample'] = UpsampleNet(
            upsample_scales= hp_Dict['WaveNet']['Upsample']['Scales'],
            pad= hp_Dict['WaveNet']['Upsample']['Pad']
            )

        self.layer_Dict['Global_Condition_Embedding'] = tf.keras.Sequential()
        self.layer_Dict['Global_Condition_Embedding'].add(tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis= -1)
            ))
        self.layer_Dict['Global_Condition_Embedding'].add(tf.keras.layers.Embedding(
            input_dim= hp_Dict['WaveNet']['Speaker_Count'],
            output_dim= hp_Dict['WaveNet']['Initial_Filters'],
            ))

    def call(self, inputs, training):
        '''
        inputs: x, local_Condition, global_Condition
        x: [Batch, Time]
        local_Condition: [Batch, Time, Dim]
        global_Condition: [Batch,]
        '''
        x, local_Conditions, global_Conditions = inputs
        # return tf.cond(
        #     pred= tf.convert_to_tensor(training),
        #     true_fn= lambda: self.train(x, local_Conditions, global_Conditions),
        #     false_fn= lambda: self.inference(local_Conditions, global_Conditions)
        #     ) 
        if training:
            return self.train(x, local_Conditions, global_Conditions)
        else:
            return self.inference(local_Conditions, global_Conditions)
            

    def train(self, x, local_Conditions, global_Conditions):
        local_Conditions = self.layer_Dict['Local_Condition_Upsample'](local_Conditions)
        global_Conditions = self.layer_Dict['Global_Condition_Embedding'](global_Conditions)        
        x = self.layer_Dict['First'](inputs= x, training= tf.convert_to_tensor(True))
        skips = 0
        for block_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Blocks']):
            for stack_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Stacks_in_Block']):
                x, new_Skips = self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)](
                    inputs= [x, local_Conditions, global_Conditions]
                    )
                skips += new_Skips
        skips *= np.sqrt(1.0 / (hp_Dict['WaveNet']['ResConvGLU']['Blocks'] * hp_Dict['WaveNet']['ResConvGLU']['Stacks_in_Block']))

        logits = self.layer_Dict['Last'](skips)
        
        return logits, tf.zeros(shape=(tf.shape(logits)[0], 1), dtype= logits.dtype)

    def inference(self, local_Conditions, global_Conditions):
        batch_Size = tf.shape(local_Conditions)[0]
        local_Conditions = self.layer_Dict['Local_Condition_Upsample'](local_Conditions)
        global_Conditions = self.layer_Dict['Global_Condition_Embedding'](global_Conditions)

        # Inference step by step
        initial_Samples = tf.zeros(shape=(batch_Size, 1))        
        def body(step, samples):            
            current_Local_Condition = tf.expand_dims(local_Conditions[:, step, :], axis= 1)
            # global_Condition is always same. Thus, there is no step slicing.
            
            # Initialize
            self.layer_Dict['First'].layers[-1].inputs_initialize()
            for block_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Blocks']):
                for stack_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Stacks_in_Block']):
                    self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)].inputs_initialize()

            x = self.layer_Dict['First'](
                inputs= tf.expand_dims(samples[:, -1], axis= 1),
                training= tf.convert_to_tensor(False)
                )
            skips = 0
            for block_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Blocks']):
                for stack_Index in range(hp_Dict['WaveNet']['ResConvGLU']['Stacks_in_Block']):
                    x, new_Skips = self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)](
                        inputs= [x, current_Local_Condition, global_Conditions]
                        )
                    skips += new_Skips
            skips *= np.sqrt(1.0 / (hp_Dict['WaveNet']['ResConvGLU']['Blocks'] * hp_Dict['WaveNet']['ResConvGLU']['Stacks_in_Block']))
            
            logit = self.layer_Dict['Last'](skips)                        
            samples = tf.concat([initial_Samples, Sample_from_Discretized_Mix_Logistic(logit)], axis= -1)
            
            try: progress(step + 1, local_Conditions.shape[1], status='({}/{})'.format(step + 1, local_Conditions.get_shape()[1])) #initial time it will be ignored.
            except: pass

            return step + 1, samples
        print()
        
        _, samples = tf.while_loop(
            cond= lambda step, outputs: step < tf.shape(local_Conditions)[1],
            body= body,
            loop_vars= [
                0,
                initial_Samples
                ],
            shape_invariants= [
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                ]
            )
        return tf.zeros(shape=(tf.shape(samples)[0], 1, hp_Dict['WaveNet']['MoL_Sizes']), dtype= samples.dtype), samples

class UpsampleNet(tf.keras.Model):
    def __init__(
        self,
        upsample_scales,
        pad
        ):
        super(UpsampleNet, self).__init__()
        self.upsample_scales = upsample_scales
        self.pad = pad

        self.total_scale = np.cumproduct(self.upsample_scales)[-1]
        self.indent = self.pad * self.total_scale

    def build(self, input_shapes):
        self.layer_Dict = {}

        self.layer = tf.keras.Sequential()
        self.layer.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis= -1)))
        for scale in self.upsample_scales:
            self.layer.add(tf.keras.layers.UpSampling2D(size= (scale, 1)))
            self.layer.add(Weight_Norm_Wrapper(tf.keras.layers.Conv2D(
                filters= 1,
                kernel_size= (scale * 2 + 1, 1),
                kernel_initializer= tf.constant_initializer(1 / (scale * 2 + 1)),
                padding= 'same',
                use_bias= False
                )))                
        self.layer.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis= -1)))
        self.built = True

    def call(self, inputs, training):
        new_Tensor = self.layer(inputs, training)
        new_Tensor = new_Tensor[:, self.indent:-self.indent, :]
        
        return new_Tensor

class ResConvGLU(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        skip_out_filters,
        dropout_rate= 0.05,
        dilation_rate= 1,
        use_bias= True
        ):
        super(ResConvGLU, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.skip_out_filters = skip_out_filters
        self.dropout_rate = dropout_rate
        self.dilation_rate = dilation_rate
        self.use_bias= use_bias

    def build(self, input_shapes):
        self.layer_Dict = {}
        self.layer_Dict['Dropout'] = tf.keras.layers.Dropout(
            rate= self.dropout_rate
            )
        # self.layer_Dict['Conv1D'] = tf.keras.layers.Conv1D(
        #     filters= self.filters,
        #     kernel_size= self.kernel_size,
        #     strides= 1,
        #     padding= 'causal',
        #     dilation_rate= self.dilation_rate,
        #     use_bias= self.use_bias,
        #     kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
        #     bias_initializer= tf.keras.initializers.zeros
        #     )
        # self.layer_Dict['Conv1D'] = Weight_Norm_Wrapper(
        #     self.layer_Dict['Conv1D']
        #     )
        self.layer_Dict['Conv1D'] = Incremental_Conv1D_Causal_WN(
            filters= self.filters,
            kernel_size= self.kernel_size,
            strides= 1,
            dilation_rate= self.dilation_rate,
            use_bias= self.use_bias,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            bias_initializer= tf.keras.initializers.zeros
            )

        self.layer_Dict['Out'] = tf.keras.layers.Conv1D(
            filters= input_shapes[0][-1],
            kernel_size= 1,
            strides= 1,
            padding= 'same',
            use_bias= self.use_bias,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            bias_initializer= tf.keras.initializers.zeros
            )
        self.layer_Dict['Out'] = Weight_Norm_Wrapper(
            self.layer_Dict['Out']
            )
        self.layer_Dict['Skip'] = tf.keras.layers.Conv1D(
            filters= self.skip_out_filters,
            kernel_size= 1,
            strides= 1,
            padding= 'same',
            use_bias= self.use_bias,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            bias_initializer= tf.keras.initializers.zeros
            )
        self.layer_Dict['Skip'] = Weight_Norm_Wrapper(
            self.layer_Dict['Skip']
            )

        self.layer_Dict['Local_Condition_Conv1D'] = tf.keras.layers.Conv1D(
            filters= self.filters,
            kernel_size= 1,
            strides= 1,
            padding= 'same',
            use_bias= False,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            )
        self.layer_Dict['Local_Condition_Conv1D'] = Weight_Norm_Wrapper(
            self.layer_Dict['Local_Condition_Conv1D']
            )

        self.layer_Dict['Global_Condition_Conv1D'] = tf.keras.layers.Conv1D(
            filters= self.filters,
            kernel_size= 1,
            strides= 1,
            padding= 'same',
            use_bias= False,
            kernel_initializer= tf.keras.initializers.he_normal(),    # kaiming_normal
            )
        self.layer_Dict['Global_Condition_Conv1D'] = Weight_Norm_Wrapper(
            self.layer_Dict['Global_Condition_Conv1D']
            )

    def call(self, inputs, training= False):
        '''
        inputs: x, local_Condition, global_Condition
        x: [Batch, Time, Sig_Dim]
        local_Condition: [Batch, Time, Sig_Dim]
        global_Condition: [Batch, Time, Sig_Dim]
        '''

        x, local_Condition, global_Condition = inputs

        new_Tensor = self.layer_Dict['Dropout'](x)
        new_Tensor = self.layer_Dict['Conv1D'](new_Tensor, training)
        h_Tensor, s_Tensor = tf.split(new_Tensor, num_or_size_splits= 2, axis= -1)

        local_Condition_Tensor = self.layer_Dict['Local_Condition_Conv1D'](local_Condition)
        local_H_Tensor, local_S_Tensor = tf.split(local_Condition_Tensor, num_or_size_splits= 2, axis= -1)            
        h_Tensor = h_Tensor + local_H_Tensor
        s_Tensor = s_Tensor + local_S_Tensor

        global_Condition_Tensor = self.layer_Dict['Global_Condition_Conv1D'](global_Condition)
        global_H_Tensor, global_S_Tensor = tf.split(global_Condition_Tensor, num_or_size_splits= 2, axis= -1)
        h_Tensor = h_Tensor + global_H_Tensor
        s_Tensor = s_Tensor + global_S_Tensor

        new_Tensor = tf.math.tanh(h_Tensor) * tf.math.sigmoid(s_Tensor)

        skip_Tenosr = self.layer_Dict['Skip'](new_Tensor)
        out_Tensor = self.layer_Dict['Out'](new_Tensor)

        out_Tensor = (out_Tensor + x) * np.sqrt(0.5)
        
        return out_Tensor, skip_Tenosr

    def inputs_initialize(self):
        self.layer_Dict['Conv1D'].inputs_initialize()

class Weight_Norm_Wrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, g_initializer= None, g_regularizer= None):
        super(Weight_Norm_Wrapper, self).__init__(layer= layer)
        self.g_initializer = g_initializer
        self.g_regularizer = g_regularizer

    def build(self, input_shapes):
        super(Weight_Norm_Wrapper, self).build(input_shapes)
        self.g = self.add_weight(            
            name= 'weight_norm_g',
            shape= self.layer.kernel.get_shape()[-1:],
            initializer= self.g_initializer,
            regularizer= self.g_regularizer,
            trainable= True,
            )
        
    @property
    def kernel(self):
        norm_kernel= tf.math.l2_normalize(
            self.layer.kernel,
            axis= list(range(len(self.layer.kernel.get_shape()) - 1)),
            # epsilon=1e-4
            )
        return self.g * norm_kernel

    def call(self, inputs):
        _kernel = self.layer.kernel
        self.layer.kernel = self.kernel
        new_tensor = self.layer(inputs)
        self.layer.kernel = _kernel

        return new_tensor

class Incremental_Conv1D_Causal_WN(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides= 1,
        dilation_rate= 1,
        use_bias= True,
        kernel_initializer= None,    # Basic is kaiming_normal
        bias_initializer= None,
        g_initializer= None,
        ):
        super(Incremental_Conv1D_Causal_WN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer or tf.keras.initializers.he_normal
        self.bias_initializer = bias_initializer or tf.keras.initializers.zeros
        self.g_initializer = g_initializer or tf.keras.initializers.glorot_uniform

    def build(self, input_shapes):
        self.kernel = self.add_weight(            
            name= 'kernel',
            shape= [self.kernel_size, input_shapes[-1], self.filters],
            initializer= self.kernel_initializer,
            trainable= True,
            )
        self.g = self.add_weight(            
            name= 'weight_norm_g',
            shape= [self.filters,],
            initializer= self.g_initializer,
            trainable= True,
            )
        
        if self.use_bias:
            self.bias = self.add_weight(            
                name= 'bias',
                shape= [self.filters,],
                initializer= self.bias_initializer,
                trainable= True,
                )

        self.inputs_initialize()

        self.built = True

    def call(self, inputs, training):
        return tf.cond(
            pred= tf.convert_to_tensor(training),
            true_fn= lambda: self.usual(inputs),
            false_fn= lambda: self.incremental(inputs)
            )

    def usual(self, inputs):
        left_size = self.dilation_rate * (self.kernel_size - 1)
        padding = tf.zeros(
            shape=(
                tf.shape(inputs)[0],
                left_size,
                inputs.get_shape()[-1]
                ),
            dtype= inputs.dtype
            )            
        inputs = tf.concat([padding, inputs], axis= 1)
        norm_kernel = self.g * tf.math.l2_normalize(
            self.kernel,
            axis= [0, 1],
            )

        return tf.nn.conv1d(
            input= inputs,
            filters= norm_kernel,
            stride= self.strides,
            padding= 'VALID',
            dilations= self.dilation_rate
            )

    def incremental(self, inputs):
        self.previous_inputs.append(inputs)

        norm_kernel = self.g * tf.math.l2_normalize(
            self.kernel,
            axis= [0, 1],
            )

        x = tf.nn.conv1d(
            input= self.get_padded_incremental_inputs(),
            filters= norm_kernel,
            stride= self.strides,
            padding= 'VALID',
            dilations= self.dilation_rate
            )

        return x

    def get_padded_incremental_inputs(self):
        left_size = self.dilation_rate * (self.kernel_size - 1)

        stacked_inputs = tf.concat(self.previous_inputs, axis= 1)
        padding = tf.zeros(
            shape=(
                tf.shape(stacked_inputs)[0],
                tf.maximum(left_size - tf.shape(stacked_inputs)[1], 0),
                stacked_inputs.get_shape()[-1]
                ),
            dtype= stacked_inputs.dtype
            )

        return tf.concat([padding, stacked_inputs], axis= 1)[:, -(left_size + 1):]
        
    def inputs_initialize(self):
        self.previous_inputs = []

        


class Loss(tf.keras.layers.Layer):
    def call(self, inputs):        
        labels, logits = inputs
        return Discretized_Mix_Logistic_Loss(labels= labels, logits= logits)


class ExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        min_learning_rate= None,
        staircase=False,
        name=None
        ):    
        super(ExponentialDecay, self).__init__(
            initial_learning_rate= initial_learning_rate,
            decay_steps= decay_steps,
            decay_rate= decay_rate,
            staircase= staircase,
            name= name
            )

        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        learning_rate = super(ExponentialDecay, self).__call__(step)
        if self.min_learning_rate is None:
            return learning_rate

        return tf.maximum(learning_rate, self.min_learning_rate)

    def get_config(self):
        config_dict = super(ExponentialDecay, self).get_config()
        config_dict['min_learning_rate'] = self.min_learning_rate

        return config_dict


if __name__ == "__main__":
    new_WaveNet = WaveNet()

    x = np.random.rand(3, 768).astype(np.float32)
    locals = np.random.rand(3, 7, 80).astype(np.float32)
    globals = np.array([1,]).astype(np.int32)
    x = new_WaveNet(inputs=[x, locals, globals], training= True)

    print(x)