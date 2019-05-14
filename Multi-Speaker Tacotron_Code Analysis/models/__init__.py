import os
from glob import glob
from .tacotron import Tacotron


def create_model(hparams):
  return Tacotron(hparams)


def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]  # checkpoint_dir의 하위 파일들 중에서 "*.ckpt-*.data-*"꼴의 파일 경로들을 전부 가져와서 checkpoint_path에 리스트로 저장
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]  # checkpoint_path의 파일결로들에서 파일명만 추출 및 "-", "."을 기준으로 split 후 1000, 2000과 같은 값들을 정수로 저장

    max_idx = max(idxes)  # idxes 중 최대값 저장 => 즉, 가장 마지막에 저장된 파일의 숫자값 저장
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))  # 마지막에 저장된 모델 파일 경로 저장

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint
