# Code based on https://github.com/keithito/tacotron/blob/master/text/cleaners.py
'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).

클리너는 교육 및 평가시 입력 텍스트를 통해 실행되는 변환입니다.
클리너는 쉼표로 구분 된 클리너 이름 목록을 "클리너"로 전달하여 선택할 수 있습니다.
하이퍼 파라미터. 일부 클리너는 영어에 따라 다릅니다. 일반적으로 다음을 사용하려고합니다.
    1. 영어 텍스트의 경우 "english_cleaners"
    2. ASCII를 사용하여 ASCII로 음역 될 수있는 영어가 아닌 텍스트의 경우 "transliteration_cleaners"
         Unidecode 라이브러리 (https://pypi.python.org/pypi/Unidecode)
    3. 음역을 사용하지 않으려면 "basic_cleaners"를 입력하십시오 (이 경우,
         귀하의 데이터와 일치하는 symbols.py의 기호)
'''

import re
from .korean import tokenize as ko_tokenize

# Added to support LJ_speech
from unidecode import unidecode
from .en_numbers import normalize_numbers as en_normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def korean_cleaners(text):
    '''Pipeline for Korean text, including number and abbreviation expansion.'''
    text = ko_tokenize(text)
    return text


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return en_normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    '''Converts to ascii, existed in keithito but deleted in carpedm20'''
    return unidecode(text)
    

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


