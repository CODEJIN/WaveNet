# Refer:
# https://https://github.com/fatchord/WaveRNN

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import json, os, time, argparse
from threading import Thread
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from datetime import datetime

from Feeder import Feeder
from Audio import inv_spectrogram
import Modules
from scipy.io import wavfile

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)


if not hp_Dict['Device'] is None:
    os.environ["CUDA_VISIBLE_DEVICES"]= hp_Dict['Device']

if hp_Dict['Use_Mixed_Precision']:    
    policy = mixed_precision.Policy('mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

class WaveNet:
    def __init__(self, is_Training= False):
        self.feeder = Feeder(is_Training= is_Training)
        self.Model_Generate()

    def Model_Generate(self):
        input_Dict = {}
        layer_Dict = {}
        tensor_Dict = {}

        input_Dict['Audio'] = tf.keras.layers.Input(
            shape= [None,],
            dtype= tf.as_dtype(policy.compute_dtype)            
            )        
        input_Dict['Mel'] = tf.keras.layers.Input(
            shape= [None, hp_Dict['Sound']['Mel_Dim']],
            dtype= tf.as_dtype(policy.compute_dtype)
            )
        input_Dict['Speaker'] = tf.keras.layers.Input(
            shape= [],
            dtype= tf.int32
            )        

        layer_Dict['WaveRNN'] = Modules.WaveNet()        
        layer_Dict['Loss'] = Modules.Loss()
        
        tensor_Dict['Logits'], _ = layer_Dict['WaveRNN'](
            inputs= [input_Dict['Audio'][:, :-1], input_Dict['Mel'], input_Dict['Speaker']],
            training= True
            ) #Using audio is [:, :-1].

        _, tensor_Dict['Samples'] = layer_Dict['WaveRNN'](
            inputs= [input_Dict['Audio'], input_Dict['Mel'], input_Dict['Speaker']],
            training= False
            ) #Using audio is [:, :-1].
        tensor_Dict['Loss'] = layer_Dict['Loss'](
            inputs=[input_Dict['Audio'][:, 1:], tensor_Dict['Logits']]
            )    #Using audio is [:, 1:time + 1]

        self.model_Dict = {}
        self.model_Dict['Train'] = tf.keras.Model(
            inputs= [input_Dict['Audio'], input_Dict['Mel'], input_Dict['Speaker']],
            outputs= tensor_Dict['Loss']
            )

        self.model_Dict['Inference'] = tf.keras.Model(
            inputs= [input_Dict['Audio'], input_Dict['Mel'], input_Dict['Speaker']],
            outputs= tensor_Dict['Samples']
            )

        learning_Rate = Modules.ExponentialDecay(
            initial_learning_rate= hp_Dict['Train']['Learning_Rate']['Initial'],
            decay_steps= hp_Dict['Train']['Learning_Rate']['Decay_Step'],
            decay_rate= hp_Dict['Train']['Learning_Rate']['Decay_Rate'],
            min_learning_rate= hp_Dict['Train']['Learning_Rate']['Min'],
            staircase= False
            )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= learning_Rate,
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            clipnorm= 4.0
            )

        self.model_Dict['Train'].summary()
        self.model_Dict['Inference'].summary()

        self.checkpoint = tf.train.Checkpoint(optimizer= self.optimizer, model= self.model_Dict['Train'])

    # @tf.function(
    #     input_signature=[            
    #         tf.TensorSpec(shape=[None, None], dtype= tf.as_dtype(policy.compute_dtype)),
    #         tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype= tf.as_dtype(policy.compute_dtype)),
    #         tf.TensorSpec(shape=[None,], dtype= tf.int32),
    #         ],
    #     autograph= True,
    #     experimental_relax_shapes= True
    #     )
    def Train_Step(self, audios, mels, speakers):
        with tf.GradientTape() as tape:
            loss = self.model_Dict['Train'](
                inputs= [audios, mels, speakers],
                training= True
                )
        gradients = tape.gradient(loss, self.model_Dict['Train'].trainable_variables)

        self.optimizer.apply_gradients([
            (gradient, variable)
            for gradient, variable in zip(gradients, self.model_Dict['Train'].trainable_variables)
            if not gradient is None
            ])  #Avoid last outout

        return loss

    # @tf.function(
    #     input_signature=[            
    #         tf.TensorSpec(shape=[None, None], dtype= tf.as_dtype(policy.compute_dtype)),
    #         tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype= tf.as_dtype(policy.compute_dtype)),
    #         tf.TensorSpec(shape=[None,], dtype= tf.int32),
    #         ],
    #     autograph= False,
    #     experimental_relax_shapes= False
    #     )
    def Inference_Step(self, audios, mels, speakers):        
        sig = self.model_Dict['Inference'](
            inputs= [audios, mels, speakers],
            training= False
            )

        return sig

    def Restore(self, checkpoint_File_Path= None):
        if checkpoint_File_Path is None:
            checkpoint_File_Path = tf.train.latest_checkpoint(hp_Dict['Checkpoint_Path'])

        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.checkpoint.restore(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        if not os.path.exists(os.path.join(hp_Dict['Inference_Path'], 'Hyper_Parameters.json')):
            os.makedirs(hp_Dict['Inference_Path'], exist_ok= True)
            with open(os.path.join(hp_Dict['Inference_Path'], 'Hyper_Parameters.json').replace("\\", "/"), "w") as f:
                json.dump(hp_Dict, f, indent= 4)

        def Save_Checkpoint():
            os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)
            self.checkpoint.save(
                os.path.join(
                    hp_Dict['Checkpoint_Path'],
                    'S_{}.CHECKPOINT.H5'.format(self.optimizer.iterations.numpy())
                    ).replace('\\', '/')
                )
                
        def Run_Inference():            
            wav_List = []
            speaker_List = []
            with open('Inference_Wav_for_Training.txt', 'r') as f:
                for line in f.readlines():
                    path, speaker = line.strip().split('\t')
                    wav_List.append(path.strip())
                    speaker_List.append(int(speaker))

            self.Inference(wav_List= wav_List, wav_Speaker_List= speaker_List)

        # Save_Checkpoint()
        if hp_Dict['Train']['Initial_Inference']:
            Run_Inference()
        while True:
            start_Time = time.time()

            loss = self.Train_Step(**self.feeder.Get_Pattern())
            if np.isnan(loss) or np.isinf(np.abs(loss)):
                raise Exception('Because NaN/Inf loss is generated.')                
                
            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(self.optimizer.iterations.numpy()),
                'LR: {:0.7f}'.format(self.optimizer.lr(self.optimizer.iterations.numpy() - 1)),
                'Loss: {:0.5f}'.format(loss)
                ]
            print('\t\t'.join(display_List))
            with open(os.path.join(hp_Dict['Inference_Path'], 'log.txt'), 'a') as f:
                f.write('\t'.join([
                '{:0.3f}'.format(time.time() - start_Time),
                '{}'.format(self.optimizer.iterations.numpy()),
                '{:0.7f}'.format(self.optimizer.lr(self.optimizer.iterations.numpy() - 1)),
                '{:0.5f}'.format(loss)
                ]) + '\n')

            if self.optimizer.iterations.numpy() % hp_Dict['Train']['Checkpoint_Save_Timing'] == 0:
                Save_Checkpoint()
            
            if self.optimizer.iterations.numpy() % hp_Dict['Train']['Inference_Timing'] == 0:
                Run_Inference()

    def Inference(
        self,
        mel_List= None,
        mel_Speaker_List = None,
        wav_List= None,
        wav_Speaker_List = None,
        label= None,
        split_Mel_Window= 7,
        overlap_Window= 1,
        batch_Size= 16
        ):
        print('Inference running...')

        original_Sig_List, pattern_Dict_List, split_Mel_Index_List = self.feeder.Get_Inference_Pattern(
            mel_List= mel_List,
            mel_Speaker_List= mel_Speaker_List,
            wav_List= wav_List,
            wav_Speaker_List= wav_Speaker_List,
            split_Mel_Window= split_Mel_Window,
            overlap_Window= overlap_Window,
            batch_Size= batch_Size
            )
        if pattern_Dict_List is None:
            print('No data. Inference fail.')
            return
            
        print('Number of batch size: {}'.format(len(pattern_Dict_List)))
        split_Sigs = np.vstack([self.Inference_Step(**pattern_Dict).numpy() for pattern_Dict in pattern_Dict_List])
        split_Sigs = split_Sigs[:, overlap_Window*hp_Dict['Sound']['Frame_Shift']:] #Overlap cutting
        sig_List = []
        current_Index = 0
        split_Sig_List = []
        for index, split_Mel_Index in enumerate(split_Mel_Index_List):
            if split_Mel_Index > current_Index:
                sig_List.append(np.hstack(split_Sig_List))
                current_Index += 1
                split_Sig_List = []            
            split_Sig_List.append(split_Sigs[index])
        sig_List.append(np.hstack(split_Sig_List))
            
        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                sig_List,
                original_Sig_List,
                label or datetime.now().strftime("%Y%m%d.%H%M%S")
                ]
            )
        export_Inference_Thread.daemon = True
        export_Inference_Thread.start()
    
    def Export_Inference(self, sig_List, original_Sig_List= None, label= 'Result'):
        
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Plot').replace("\\", "/"), exist_ok= True)
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Wav').replace("\\", "/"), exist_ok= True)

        original_Sig_List = original_Sig_List or [None] * len(sig_List)
        for index, (sig, original_Sig) in enumerate(zip(sig_List, original_Sig_List)):
            if not original_Sig is None:
                new_Figure = plt.figure(figsize=(80, 10 * 2), dpi=100)
                plt.subplot(211)
                plt.plot(original_Sig)
                plt.title('Original wav flow    Index: {}'.format(index))
                plt.subplot(212)
            else:
                new_Figure = plt.figure(figsize=(80, 10), dpi=100)
            plt.plot(sig)
            plt.title('Inference flow    Index: {}'.format(index))
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Plot', '{}.IDX_{}.PNG'.format(label, index)).replace("\\", "/")
                )
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(hp_Dict['Inference_Path'], 'Wav', '{}.IDX_{}.WAV'.format(label, index)).replace("\\", "/"),
                data= (sig * 32768).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

if __name__ == '__main__':
    new_Model = WaveNet(is_Training= True)

    new_Model.Restore()
    new_Model.Train()