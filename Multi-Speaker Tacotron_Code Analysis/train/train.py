import os
import time
import math
import argparse
'''
argparse
명령행 파싱 모듈
program --help 에서 --help 처럼 프로그램 실행시 지정한 명령을 실행할 수 있는 모듈이다
main 함수에서 이용된다
'''
import traceback
import subprocess
import numpy as np
from jamo import h2j
import tensorflow as tf
from datetime import datetime
from functools import partial

from hparams import hparams, hparams_debug_string
from models import create_model, get_most_recent_checkpoint
#hparams(학습 조건 있는 file) 불러오기

from utils import ValueWindow, prepare_dirs
from utils import infolog, warning, plot, load_hparams
from utils import get_git_revision_hash, get_git_diff, str2bool, parallel_run

from audio import save_audio, inv_spectrogram
from text import sequence_to_text, text_to_sequence
from datasets.datafeeder import DataFeeder, _prepare_inputs

log = infolog.log


def create_batch_inputs_from_texts(texts):  # create_batch_inputs_from_texts 함수 define
    sequences = [text_to_sequence(text) for text in texts]  # 받은 값을 전부 text_to_sequence함수 위치 : text/__init__.py

    inputs = _prepare_inputs(sequences)
    input_lengths = np.asarray([len(x) for x in inputs], dtype=np.int32) # input_length는 inputs의 원소의 갯수

    for idx, (seq, text) in enumerate(zip(inputs, texts)):
        recovered_text = sequence_to_text(seq, skip_eos_and_pad=True)
        if recovered_text != h2j(text):
            log(" [{}] {}".format(idx, text))
            log(" [{}] {}".format(idx, recovered_text))
            log("="*30)

    return inputs, input_lengths


def get_git_commit():
    subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])     # Verify client is clean
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
    log('Git commit: %s' % commit)
    return commit


def add_stats(model, model2=None, scope_name='train'):
    with tf.variable_scope(scope_name) as scope:
        summaries = [
                tf.summary.scalar('loss_mel', model.mel_loss),
                tf.summary.scalar('loss_linear', model.linear_loss),
                tf.summary.scalar('loss', model.loss_without_coeff),
        ]

        if scope_name == 'train':
            gradient_norms = [tf.norm(grad) for grad in model.gradients if grad is not None]

            summaries.extend([
                    tf.summary.scalar('learning_rate', model.learning_rate),
                    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)),
            ])

    if model2 is not None:
        with tf.variable_scope('gap_test-train') as scope:
            summaries.extend([
                    tf.summary.scalar('loss_mel',
                            model.mel_loss - model2.mel_loss),
                    tf.summary.scalar('loss_linear', 
                            model.linear_loss - model2.linear_loss),
                    tf.summary.scalar('loss',
                            model.loss_without_coeff - model2.loss_without_coeff),
            ])

    return tf.summary.merge(summaries)


def save_and_plot_fn(args, log_dir, step, loss, prefix):
    idx, (seq, spec, align) = args

    audio_path = os.path.join(
            log_dir, '{}-step-{:09d}-audio{:03d}.wav'.format(prefix, step, idx))
    align_path = os.path.join(
            log_dir, '{}-step-{:09d}-align{:03d}.png'.format(prefix, step, idx))

    waveform = inv_spectrogram(spec.T)
    save_audio(waveform, audio_path)

    info_text = 'step={:d}, loss={:.5f}'.format(step, loss)
    if 'korean_cleaners' in [x.strip() for x in hparams.cleaners.split(',')]:
        log('Training korean : Use jamo')
        plot.plot_alignment(
                align, align_path, info=info_text,
                text=sequence_to_text(seq,
                        skip_eos_and_pad=True, combine_jamo=True), isKorean=True)
    else:
        log('Training non-korean : X use jamo')
        plot.plot_alignment(
                align, align_path, info=info_text,
                text=sequence_to_text(seq,
                        skip_eos_and_pad=True, combine_jamo=False), isKorean=False) 

def save_and_plot(sequences, spectrograms,
        alignments, log_dir, step, loss, prefix):

    fn = partial(save_and_plot_fn,
        log_dir=log_dir, step=step, loss=loss, prefix=prefix)
    items = list(enumerate(zip(sequences, spectrograms, alignments)))

    parallel_run(fn, items, parallel=False)
    log('Test finished for step {}.'.format(step))


def train(log_dir, config):
    config.data_paths = config.data_paths  # 파싱된 명령행 인자값 중 데이터 경로 : default='datasets/kr_example'

    data_dirs = [os.path.join(data_path, "data") \
            for data_path in config.data_paths]
    num_speakers = len(data_dirs) # 학습하는 화자 수 측정 : 단일화자 모델-1, 다중화자 모델-2
    config.num_test = config.num_test_per_speaker * num_speakers

    if num_speakers > 1 and hparams.model_type not in ["deepvoice", "simple"]:  # 다중화자 모델 학습일 때 모델 타입이 "deepvoice"나 "simple"이 아니라면
        raise Exception("[!] Unkown model_type for multi-speaker: {}".format(config.model_type))  # hparams.modle_type을 config.model_type으로 오타남.

    commit = get_git_commit() if config.git else 'None'  # git 관련된거여서 무시
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')  # checkpoint_path 경로 지정-model.skpt 파일 경로

    log(' [*] git recv-parse HEAD:\n%s' % get_git_revision_hash())  # git log
    log('='*50)  # 줄 구분용 =====
    #log(' [*] dit diff:\n%s' % get_git_diff())
    log('='*50)  # 줄 구분용 =====
    log(' [*] Checkpoint path: %s' % checkpoint_path)  # check_point 경로 출력
    log(' [*] Loading training data from: %s' % data_dirs)
    log(' [*] Using model: %s' % config.model_dir)
    log(hparams_debug_string())

    # Set up DataFeeder:
    coord = tf.train.Coordinator()  # 쓰레드 사용 선언
    with tf.variable_scope('datafeeder') as scope:
        train_feeder = DataFeeder(
                coord, data_dirs, hparams, config, 32,
                data_type='train', batch_size=hparams.batch_size)
        test_feeder = DataFeeder(
                coord, data_dirs, hparams, config, 8,
                data_type='test', batch_size=config.num_test)

    # Set up model:
    is_randomly_initialized = config.initialize_path is None
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('model') as scope:
        model = create_model(hparams)
        model.initialize(
                train_feeder.inputs, train_feeder.input_lengths,
                num_speakers,  train_feeder.speaker_id,
                train_feeder.mel_targets, train_feeder.linear_targets,
                train_feeder.loss_coeff,
                is_randomly_initialized=is_randomly_initialized)

        model.add_loss()
        model.add_optimizer(global_step)
        train_stats = add_stats(model, scope_name='stats') # legacy

    with tf.variable_scope('model', reuse=True) as scope:
        test_model = create_model(hparams)
        test_model.initialize(
                test_feeder.inputs, test_feeder.input_lengths,
                num_speakers, test_feeder.speaker_id,
                test_feeder.mel_targets, test_feeder.linear_targets,
                test_feeder.loss_coeff, rnn_decoder_test_mode=True,
                is_randomly_initialized=is_randomly_initialized)
        test_model.add_loss()

    test_stats = add_stats(test_model, model, scope_name='test')
    test_stats = tf.summary.merge([test_stats, train_stats])

    # Bookkeeping:
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=2)

    sess_config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True

    # Train!
    #with tf.Session(config=sess_config) as sess:
    with tf.Session() as sess:  # with문 내의 모든 명령들은 CPU 혹은 GPU 사용 선언
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # summary 오퍼레이션이 평가된 결과 및 텐서보드 그래프를 파라미터 형식으로 log_dir 에 저장
            sess.run(tf.global_variables_initializer())  # 데이터셋이 로드되고 그래프가 모두 정의되면 변수를 초기화하여 훈련 시작

            if config.load_path:  # log의 설정 값들 경로를 지정하였다면
                # Restore from a checkpoint if the user requested it.
                restore_path = get_most_recent_checkpoint(config.model_dir)  # 가장 마지막에 저장된 파일경로 저장
                saver.restore(sess, restore_path)  # restore_path 값 가져오기
                log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)  # git과 slack을 이용한 log 출력
            elif config.initialize_path:  # log의 설정 값들로 초기화하여 사용하기로 지정하였다면
                restore_path = get_most_recent_checkpoint(config.initialize_path)  # 지정된 경로에서 가장 마지막에 저장된 파일경로 저장
                saver.restore(sess, restore_path)  # restore_path 값 가져오기
                log('Initialized from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)  # git과 slack을 이용한 log 출력

                zero_step_assign = tf.assign(global_step, 0)  # global_step의 텐서 객체 참조 변수 값을 0으로 바꿔주는 명령어 지정
                sess.run(zero_step_assign)  # 변수들을 모두 0으로 바꾸는 명령어 실행

                start_step = sess.run(global_step)  # global_step 값 부분을 시작지점으로 하여 연산 시작
                log('='*50)
                log(' [*] Global step is reset to {}'. \
                        format(start_step))  # 즉, 연산 시작 부분이 0으로 초기화 되었다고 알려줌.
                log('='*50)
            else:
                log('Starting new training run at commit: %s' % commit, slack=True)  # 과거의 데이터를 사용하지 않을 경우 새로운 학습이라고 log 출력

            start_step = sess.run(global_step)  # 연산 시작지점 가져오기

            train_feeder.start_in_session(sess, start_step)
            test_feeder.start_in_session(sess, start_step)

            while not coord.should_stop():  # 쓰레드가 멈춰야하는 상황이 아니라면
                start_time = time.time()  # 시작시간 지정(1970년 1월 1일 이후 경과된 시간을 UTC 기준으로 초로 반환)
                step, loss, opt = sess.run(
                        [global_step, model.loss_without_coeff, model.optimize],
                        feed_dict=model.get_dummy_feed_dict())  # step 값은 global_step 값으로 지정, loss 값은

                time_window.append(time.time() - start_time)
                loss_window.append(loss)

                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
                        step, time_window.average, loss, loss_window.average)
                log(message, slack=(step % config.checkpoint_interval == 0))

                if loss > 100 or math.isnan(loss):
                    log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
                    raise Exception('Loss Exploded')

                if step % config.summary_interval == 0:
                    log('Writing summary at step: %d' % step)

                    feed_dict = {
                            **model.get_dummy_feed_dict(),
                            **test_model.get_dummy_feed_dict()
                    }
                    summary_writer.add_summary(sess.run(
                            test_stats, feed_dict=feed_dict), step)

                if step % config.checkpoint_interval == 0:
                    log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % config.test_interval == 0:
                    log('Saving audio and alignment...')
                    num_test = config.num_test

                    fetches = [
                            model.inputs[:num_test],
                            model.linear_outputs[:num_test],
                            model.alignments[:num_test],
                            test_model.inputs[:num_test],
                            test_model.linear_outputs[:num_test],
                            test_model.alignments[:num_test],
                    ]
                    feed_dict = {
                            **model.get_dummy_feed_dict(),
                            **test_model.get_dummy_feed_dict()
                    }

                    sequences, spectrograms, alignments, \
                            test_sequences, test_spectrograms, test_alignments = \
                                    sess.run(fetches, feed_dict=feed_dict)

                    save_and_plot(sequences[:1], spectrograms[:1], alignments[:1],
                            log_dir, step, loss, "train")
                    save_and_plot(test_sequences, test_spectrograms, test_alignments,
                            log_dir, step, loss, "test")

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()  # argparse 라이브러리를 사용
    '''
    add_argument를 통해 명령행 옵션을 추가한다
    
    위치인자, 옵션인자
    위치인자 : '명령행 옵션'에 -, -- 가 안붙는 형태, 선언 순서가 명령행 입력 순서에 영향을 끼친다
    옵션인자 : '명령행 옵션'에 -, -- 가 붙는 형태
    옵션인자는 명령행에 없어도 되지만 위치인자는 없으면 오류를 낸다
    
    add_argument('명령행 옵션', type=x, default=y, action='store_true', help='hello, python')
    '명령행 옵션' : 사용하고 싶은 명령어 옵션 명명
    type=x : 명령행 옵션을 통해 받는 변수의 type 지정
    default=y : 해당 옵션을 사용하지 않아도 dafault=y 값이 사용됨. 평상시에는 None의 값을 가짐
    action='store_true' : 명령행 옵션을 변수명으로 이용시 반환값을 True로 지정. default 값은 False
    help='hello, python' : -h, --help 를 사용할때 나타나는 명령행 옵션 옆에서 명령행 옵션에 대한 설명을 서술하는 용도
    '''
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--data_paths', default='datasets/kr_example')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--initialize_path', default=None)

    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=500)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--skip_path_filter', type=str2bool, default=False, help='Use only for debugging')
    # utils 폴더의 __init__.py 내부의 str2bool 함수 사용

    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')

    config = parser.parse_args()  # 명령행 인자값 파싱
    config.data_paths = config.data_paths.split(",")  # (주의!)data_paths(default:datasets/kr_example)를 split.
    setattr(hparams, "num_speakers", len(config.data_paths))  # hparams의 "num_speakers에 값 len(config.data_paths) 설정"

    prepare_dirs(config, hparams)

    log_path = os.path.join(config.model_dir, 'train.log')  # config.model_dir, 'train.log' 경로 합침
    infolog.init(log_path, config.model_dir, config.slack_url)

    tf.set_random_seed(config.random_seed)
    print(config.data_paths)

    if any("krbook" not in data_path for data_path in config.data_paths) and \
            hparams.sample_rate != 20000:
        warning("Detect non-krbook dataset. May need to set sampling rate from {} to 20000".\
                format(hparams.sample_rate))
        
    if any('LJ' in data_path for data_path in config.data_paths) and \
           hparams.sample_rate != 22050:
        warning("Detect LJ Speech dataset. Set sampling rate from {} to 22050".\
                format(hparams.sample_rate))

    if config.load_path is not None and config.initialize_path is not None:
        raise Exception(" [!] Only one of load_path and initialize_path should be set")

    train(config.model_dir, config)


if __name__ == '__main__':
    main() #main 함수 실행
