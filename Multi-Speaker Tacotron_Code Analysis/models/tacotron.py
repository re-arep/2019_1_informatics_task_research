# Code based on https://github.com/keithito/tacotron/blob/master/models/tacotron.py

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, BahdanauMonotonicAttention
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper

from utils.infolog import log
from text.symbols import symbols

from .modules import *
from .helpers import TacoTestHelper, TacoTrainingHelper
from .rnn_wrappers import AttentionWrapper, DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper


class Tacotron():
    def __init__(self, hparams):
        self._hparams = hparams


    def initialize(
            self, inputs, input_lengths, num_speakers, speaker_id,
            mel_targets=None, linear_targets=None, loss_coeff=None,
            rnn_decoder_test_mode=False, is_randomly_initialized=False,
        ):
        is_training = linear_targets is not None  # linear_targets가 초기값(None)이면 False
        self.is_randomly_initialized = is_randomly_initialized  # 초기값 False

        with tf.variable_scope('inference') as scope:  # 'inference'라는 이름으로 묶음
            hp = self._hparams
            batch_size = tf.shape(inputs)[0]  # 첫번째 차원은 샘플 수, 두번째 차원은 입력 특성 수 (여기선 샘플수)

            # Embeddings
            char_embed_table = tf.get_variable(
                    'embedding', [len(symbols), hp.embedding_size], dtype=tf.float32,  # list : variable이 소속될 collection에 대한 리스트 한글의 종류수와 임베딩 크기에 속해있다. , 'embedding이라는 이름의 공유 변수 생성
                    initializer=tf.truncated_normal_initializer(stddev=0.5))  # initializer : 초기화한 가중치 dtype : 리턴한 tensor의 타입
            # [N, T_in, embedding_size]
            char_embedded_inputs = \
                    tf.nn.embedding_lookup(char_embed_table, inputs)  # inputs의 인덱스에 따라 char_embed_table값 리턴

            self.num_speakers = num_speakers
            if self.num_speakers > 1:  # 다중화자일때
                if hp.speaker_embedding_size != 1:  # hparams의 speaker_embedding_size값이 1이 아닐때
                    speaker_embed_table = tf.get_variable(  # 공유변수 생성
                            'speaker_embedding',  # 'speaker_embedding'이라는 이름의
                            [self.num_speakers, hp.speaker_embedding_size], dtype=tf.float32,  # num_speakers와 speaker_embedding_size에 속해있는
                            initializer=tf.truncated_normal_initializer(stddev=0.5))  # 초기화값 가중치
                    # [N, T_in, speaker_embedding_size]
                    speaker_embed = tf.nn.embedding_lookup(speaker_embed_table, speaker_id)  # speaker의 인덱스에 따라 speaker_embed_table값 리턴 (Tensor)
############################################################## 추가설명 필요
                if hp.model_type == 'deepvoice':  # deepvoice일때
                    if hp.speaker_embedding_size == 1:  # hparams의 speaker_embedding_size값이 1일때
                        before_highway = get_embed(  # def get_embed(inputs, num_inputs, embed_size, name):
                                speaker_id, self.num_speakers,  # speaker_id의 인덱스에 따라 embed_table값 리턴
                                hp.enc_prenet_sizes[-1], "before_highway")
                        encoder_rnn_init_state = get_embed(
                                speaker_id, self.num_speakers, 
                                hp.enc_rnn_size * 2, "encoder_rnn_init_state")

                        attention_rnn_init_state = get_embed(
                                speaker_id, self.num_speakers, 
                                hp.attention_state_size, "attention_rnn_init_state")
                        decoder_rnn_init_states = [get_embed(
                                speaker_id, self.num_speakers, 
                                hp.dec_rnn_size, "decoder_rnn_init_states{}".format(idx + 1)) \
                                        for idx in range(hp.dec_layer_num)]
##############################################################
                    else:  # hparams의 speaker_embedding_size값이 1이 아닐때
                        deep_dense = lambda x, dim: \
                                tf.layers.dense(x, dim, activation=tf.nn.softsign)
                        # input:x, units:dim, 활성화함수로 softsign사용
                        # lambda함수 예제 (lambda x,y: x + y)(10, 20) =>> 30
                        # tf.layers.dense( inputs, units, activation)
                        # inputs는 앞의 레이어를 정의
                        # units는 이 레이어에 크기를 정의
                        # 마지막으로 activation은 sigmoid나,ReLu와 같은 Activation 함수
                        # dense는 히든레이어를 구현하는 함수이다.
                        # https://bcho.tistory.com/1196

                        before_highway = deep_dense(
                                speaker_embed, hp.enc_prenet_sizes[-1])  # 앞 레이어 : speaker_embed 레이어 수 : hp.enc_prenet_sizes[-1] (기본값 128)
                        encoder_rnn_init_state = deep_dense(
                                speaker_embed, hp.enc_rnn_size * 2)  # 앞 레이어 : speaker_embed 레이어 수 : hp.enc_rnn_size * 2 (기본값 128 * 2)

                        attention_rnn_init_state = deep_dense(
                                speaker_embed, hp.attention_state_size)  # 앞 레이어 : speaker_embed 레이어 수 : hp.attention_state_size (기본값 256)
                        decoder_rnn_init_states = [deep_dense(
                                speaker_embed, hp.dec_rnn_size) for _ in range(hp.dec_layer_num)]  # hp.dec_layer_num 수만큼 (기본값 2) 레이어 list

                    speaker_embed = None  # deepvoice does not use speaker_embed directly 딥보이스는 speaker_embed를 바로 사용하지 않는다.
                elif hp.model_type == 'simple':  # modeltype이 deepvoice가 아니라 simple일때
                    before_highway = None
                    encoder_rnn_init_state = None
                    attention_rnn_init_state = None
                    decoder_rnn_init_states = None  # 레이어 전부 x
                else:
                    raise Exception(" [!] Unkown multi-speaker model type: {}".format(hp.model_type))  # multi-speaker model type이 아니라고 에러메세지 출력
            else:  # 스피커의 수가 1명이면
                speaker_embed = None
                before_highway = None
                encoder_rnn_init_state = None
                attention_rnn_init_state = None
                decoder_rnn_init_states = None  # 레이어 전부 x
            ##############
            # Encoder (특수문자, 한글 자모음text를 숫자로)
            ##############

            # [N, T_in, enc_prenet_sizes[-1]]
            prenet_outputs = prenet(char_embedded_inputs, is_training,  # 다음 변수에 따른 조건으로 신경망 생성
                    hp.enc_prenet_sizes, hp.dropout_prob,
                    scope='prenet')

            encoder_outputs = cbhg(
                    prenet_outputs, input_lengths, is_training,
                    hp.enc_bank_size, hp.enc_bank_channel_size,
                    hp.enc_maxpool_width, hp.enc_highway_depth, hp.enc_rnn_size,
                    hp.enc_proj_sizes, hp.enc_proj_width,
                    scope="encoder_cbhg",
                    before_highway=before_highway,
                    encoder_rnn_init_state=encoder_rnn_init_state)


            ##############
            # Attention (중요!)
            ##############

            # For manaul control of attention
            self.is_manual_attention = tf.placeholder(
                    tf.bool, shape=(), name='is_manual_attention',
            )
            self.manual_alignments = tf.placeholder(
                    tf.float32, shape=[None, None, None], name="manual_alignments",
            )

            dec_prenet_outputs = DecoderPrenetWrapper(
                    GRUCell(hp.attention_state_size),
                    speaker_embed,
                    is_training, hp.dec_prenet_sizes, hp.dropout_prob)

            if hp.attention_type == 'bah_mon':
                attention_mechanism = BahdanauMonotonicAttention(
                        hp.attention_size, encoder_outputs)
            elif hp.attention_type == 'bah_norm':
                attention_mechanism = BahdanauAttention(
                        hp.attention_size, encoder_outputs, normalize=True)
            elif hp.attention_type == 'luong_scaled':
                attention_mechanism = LuongAttention(
                        hp.attention_size, encoder_outputs, scale=True)
            elif hp.attention_type == 'luong':
                attention_mechanism = LuongAttention(
                        hp.attention_size, encoder_outputs)
            elif hp.attention_type == 'bah':
                attention_mechanism = BahdanauAttention(
                        hp.attention_size, encoder_outputs)
            elif hp.attention_type.startswith('ntm2'):
                shift_width = int(hp.attention_type.split('-')[-1])
                attention_mechanism = NTMAttention2(
                        hp.attention_size, encoder_outputs, shift_width=shift_width)
            else:
                raise Exception(" [!] Unkown attention type: {}".format(hp.attention_type))

            attention_cell = AttentionWrapper(
                    dec_prenet_outputs,
                    attention_mechanism,
                    self.is_manual_attention,
                    self.manual_alignments,
                    initial_cell_state=attention_rnn_init_state,
                    alignment_history=True,
                    output_attention=False
            )

            # Concatenate attention context vector and RNN cell output into a 512D vector.
            # [N, T_in, attention_size+attention_state_size]
            concat_cell = ConcatOutputAndAttentionWrapper(
                    attention_cell, embed_to_concat=speaker_embed)
                        
            # Decoder (layers specified bottom to top):
            cells = [OutputProjectionWrapper(concat_cell, hp.dec_rnn_size)]
            for _ in range(hp.dec_layer_num):
                cells.append(ResidualWrapper(GRUCell(hp.dec_rnn_size)))

            # [N, T_in, 256]
            decoder_cell = MultiRNNCell(cells, state_is_tuple=True)

            # Project onto r mel spectrograms (predict r outputs at each RNN step):
            output_cell = OutputProjectionWrapper(
                    decoder_cell, hp.num_mels * hp.reduction_factor)
            decoder_init_state = output_cell.zero_state(
                    batch_size=batch_size, dtype=tf.float32)

            if hp.model_type == "deepvoice":
                # decoder_init_state[0] : AttentionWrapperState
                # = cell_state + attention + time + alignments + alignment_history
                # decoder_init_state[0][0] = attention_rnn_init_state (already applied)
                decoder_init_state = list(decoder_init_state)

                for idx, cell in enumerate(decoder_rnn_init_states):
                    shape1 = decoder_init_state[idx + 1].get_shape().as_list()
                    shape2 = cell.get_shape().as_list()
                    if shape1 != shape2:
                        raise Exception(" [!] Shape {} and {} should be equal". \
                                format(shape1, shape2))
                    decoder_init_state[idx + 1] = cell

                decoder_init_state = tuple(decoder_init_state)

            if is_training:
                helper = TacoTrainingHelper(
                        inputs, mel_targets, hp.num_mels, hp.reduction_factor,
                        rnn_decoder_test_mode)
            else:
                helper = TacoTestHelper(
                        batch_size, hp.num_mels, hp.reduction_factor)

            (decoder_outputs, _), final_decoder_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                            BasicDecoder(output_cell, helper, decoder_init_state),
                            maximum_iterations=hp.max_iters)

            # [N, T_out, M]
            mel_outputs = tf.reshape(
                    decoder_outputs, [batch_size, -1, hp.num_mels])

            # Add post-processing CBHG:
            # [N, T_out, 256]
            #post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training)
            post_outputs = cbhg(
                    mel_outputs, None, is_training,
                    hp.post_bank_size, hp.post_bank_channel_size,
                    hp.post_maxpool_width, hp.post_highway_depth, hp.post_rnn_size,
                    hp.post_proj_sizes, hp.post_proj_width,
                    scope='post_cbhg')

            if speaker_embed is not None and hp.model_type == 'simple':
                expanded_speaker_emb = tf.expand_dims(speaker_embed, [1])
                tiled_speaker_embedding = tf.tile(
                        expanded_speaker_emb, [1, tf.shape(post_outputs)[1], 1])

                # [N, T_out, 256 + alpha]
                post_outputs = \
                        tf.concat([tiled_speaker_embedding, post_outputs], axis=-1)

            linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)    # [N, T_out, F]

            # Grab alignments from the final decoder state:
            alignments = tf.transpose(
                    final_decoder_state[0].alignment_history.stack(), [1, 2, 0])


            self.inputs = inputs
            self.speaker_id = speaker_id
            self.input_lengths = input_lengths
            self.loss_coeff = loss_coeff
            self.mel_outputs = mel_outputs
            self.linear_outputs = linear_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            self.final_decoder_state = final_decoder_state

            log('='*40)
            log(' model_type: %s' % hp.model_type)
            log('='*40)

            log('Initialized Tacotron model. Dimensions: ')
            log('    embedding:                %d' % char_embedded_inputs.shape[-1])
            if speaker_embed is not None:
                log('    speaker embedding:        %d' % speaker_embed.shape[-1])
            else:
                log('    speaker embedding:        None')
            log('    prenet out:               %d' % prenet_outputs.shape[-1])
            log('    encoder out:              %d' % encoder_outputs.shape[-1])
            log('    attention out:            %d' % attention_cell.output_size)
            log('    concat attn & out:        %d' % concat_cell.output_size)
            log('    decoder cell out:         %d' % decoder_cell.output_size)
            log('    decoder out (%d frames):  %d' % (hp.reduction_factor, decoder_outputs.shape[-1]))
            log('    decoder out (1 frame):    %d' % mel_outputs.shape[-1])
            log('    postnet out:              %d' % post_outputs.shape[-1])
            log('    linear out:               %d' % linear_outputs.shape[-1])


    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            mel_loss = tf.abs(self.mel_targets - self.mel_outputs)

            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            expanded_loss_coeff = tf.expand_dims(
                    tf.expand_dims(self.loss_coeff, [-1]), [-1])

            if hp.prioritize_loss:
                # Prioritize loss for frequencies.
                upper_priority_freq = int(5000 / (hp.sample_rate * 0.5) * hp.num_freq)
                lower_priority_freq = int(165 / (hp.sample_rate * 0.5) * hp.num_freq)

                l1_priority= l1[:,:,lower_priority_freq:upper_priority_freq]

                self.loss = tf.reduce_mean(mel_loss * expanded_loss_coeff) + \
                        0.5 * tf.reduce_mean(l1 * expanded_loss_coeff) + \
                        0.5 * tf.reduce_mean(l1_priority * expanded_loss_coeff)
                self.linear_loss = tf.reduce_mean(
                        0.5 * (tf.reduce_mean(l1) + tf.reduce_mean(l1_priority)))
            else:
                self.loss = tf.reduce_mean(mel_loss * expanded_loss_coeff) + \
                        tf.reduce_mean(l1 * expanded_loss_coeff)
                self.linear_loss = tf.reduce_mean(l1)

            self.mel_loss = tf.reduce_mean(mel_loss)
            self.loss_without_coeff = self.mel_loss + self.linear_loss


    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams

            step = tf.cast(global_step + 1, dtype=tf.float32)

            if hp.decay_learning_rate_mode == 0:
                if self.is_randomly_initialized:
                    warmup_steps = 4000.0
                else:
                    warmup_steps = 40000.0
                self.learning_rate = hp.initial_learning_rate * warmup_steps**0.5 * \
                        tf.minimum(step * warmup_steps**-1.5, step**-0.5)
            elif hp.decay_learning_rate_mode == 1:
                self.learning_rate = hp.initial_learning_rate * \
                        tf.train.exponential_decay(1., step, 3000, 0.95)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                    global_step=global_step)

    def get_dummy_feed_dict(self):
        feed_dict = {
                self.is_manual_attention: False,
                self.manual_alignments: np.zeros([1, 1, 1]),
        }
        return feed_dict
