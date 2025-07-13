from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import pathlib
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"
LANG_MODEL_PATH = DATA_PATH / "lid.176.bin"
NSFW_MODEL_PATH = DATA_PATH / "jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_SPEECH_MODEL_PATH = DATA_PATH / "jigsaw_fasttext_bigrams_hatespeech_final.bin"

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    Extract text from HTML bytes.

    :param html_bytes: HTML bytes
    :type html_bytes: bytes
    :return: Extracted text
    :rtype: str | None
    """
    encoding = detect_encoding(html_bytes)
    if encoding is None:
        return None
    try:
        html_str = html_bytes.decode(encoding)
    except UnicodeDecodeError:
        return None
    text = extract_plain_text(html_str)
    return text

def indentify_language(text: str):
    import numpy as np
    model = fasttext.load_model(str(LANG_MODEL_PATH))
    text = text.replace("\n", ' ')
    predict = model.predict(text, k = 1)
    language_label = predict[0][0].replace("__label__", "")
    confidence = predict[1][0]
    return (language_label, confidence)

def mask_email(text: str) -> str:
    pattern = r'[a-zA-Z0-9_.+-]+@[A-Za-z0-9.-]+\.[a-zA-Z0-9-.]+'
    masked, count =  re.subn(pattern, '|||EMAIL_ADDRESS|||', text)
    return (masked, count)

def mask_phone_numbers(text: str) -> str:
    pattern = re.compile(
        r'''
        (?:(?:\+?1[\s\-.]*)?)         # 可选国家码
        (?:\(?\d{3}\)?)[\s\-\.]*      # 区号，可以有括号，后可跟空格、横线、点
        \d{3}[\s\-\.]*\d{4}|1[0-9]{10}           # 主体7位数字（3位+4位），中间分隔符可有
        ''', re.VERBOSE
    )
    masked, count =  pattern.subn( '|||PHONE_NUMBER|||', text)
    return (masked, count)

def mask_ip_address(text: str) -> str:
    octet = r'(?:25[0-5]|2[0-4][0-9]|1?\d\d?)'
    ip_pattern = rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'
    masked, count = re.subn(ip_pattern, '|||IP_ADDRESS|||', text)
    return masked, count

def identify_nsfw(text: str):
    model = fasttext.load_model(str(NSFW_MODEL_PATH))
    predict = model.predict(text, k = 1)
    label = predict[0][0].replace("__label__", "")
    confidence = predict[1][0]
    return (label, confidence)

def identify_toxic(text: str):
    model = fasttext.load_model(str(TOXIC_SPEECH_MODEL_PATH))
    predict = model.predict(text, k = 1)
    label = predict[0][0].replace("__label__", "")
    confidence = predict[1][0]
    return (label, confidence)


def gopher_test(text: str):
    tokens = word_tokenize(text)
    # contain less than 50 or more than 100,000 words.
    if len(tokens) < 50 or len(tokens) > 1_000_000:
        return False
    # have a mean word length outside the range of 3 to 10 characters.
    avg_len = sum((len(token) for token in tokens)) / len(tokens)
    if not 3 <= avg_len <= 10:
        return False
    # Have more than 30% of lines ending with an ellipsis (“...”)
    lines = text.splitlines()
    ellipsis_lines = sum(1 for line in lines if line.strip().endswith('...'))
    if ellipsis_lines / len(lines) > 0.3:
        return False
    # contain less than 80% of words with at least one alphabetic character.
    alpha_words = sum(1 for token in tokens if token.isalpha())
    if alpha_words / len(tokens) < 0.8:
        return False
    return True


    
    