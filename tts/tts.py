from typing import TYPE_CHECKING
from pathlib import Path
from io import BytesIO
import wave
from re import findall, sub
from pymorphy2 import MorphAnalyzer
from transliterate import translit
from num2words import num2words


import torch
from torch.package import PackageImporter
import numpy as np

from tts.exceptions import *

if TYPE_CHECKING:
    from .typing.package import TTSModelMultiAcc_v3

NUMBERS = """0,ноль,нулевой
1,один,первый
2,два,второй
3,три,третий
4,четыре,четвертый
5,пять,пятый
6,шесть,шестой
7,семь,седьмой
8,восемь,восьмой
9,девять,девятый
10,десять,десятый
11,одинадцать,одинадцатый
12,двенадцать,двенадцатый
13,тринадцать,тринадцатый
14,четырнадцать,четырнадцатый
15,пятнадцать,пятнадцатый
16,шестнадцать,шестнадцатый
17,семнадцать,семнадцатый
18,восемнадцать,восемнадцатый
19,девятнадцать,девятнадцатый
20,двадцать,двадцатый
30,тридцать,тридцатый
40,сорок,сороковой
50,пятьдесят,пятидесятый
60,шестьдесят,шестидесятый
70,семьдесят,семидесятый
80,восемьдесят,восьмидесятый
90,девяносто,девяностый
100,сто,сотый
200,двести,двухсотый
300,триста,трехсотый
400,четыреста,четырехсотый
500,пятьсот,пятисотый
600,шестьсот,шестисотый
700,семьсот,семисотый
800,восемьсот,восьмисотый
900,девятьсот,девятисотый
1000,тысяча,тысячный
1000000,миллион,миллионный"""

# fixes import package error on Mac
# https://github.com/snakers4/silero-models/discussions/104
#torch.backends.quantized.engine = "qnnpack"

MAX_INT16 = 32767

print(f"Using {torch.get_num_threads()} threads. To change, set environment variable MKL_NUM_THREADS")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.cuda.device(0)
print('Using device:', device)


class TTS:
    VALID_SAMPLE_RATES = (8000, 24000, 48000)

    def __init__(self):
        self.models: dict[str, "TTSModelMultiAcc_v3"] = {}
        self.speakers: dict[str, list[str]] = {}
        self.model_by_speaker: dict[str, "TTSModelMultiAcc_v3"] = {}

        for model_path in Path("models").glob("*.pt"):
            self._load_model(model_path)

    def generate(
        self, text: str, speaker: str, sample_rate: int, pitch: int, rate: int
    ) -> bytes:
        model = self.model_by_speaker.get(speaker)
        if not model:
            raise NotFoundModelException(speaker)
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise InvalidSampleRateException(sample_rate)

        if not 0 <= pitch <= 100:
            raise InvalidPitchException(pitch)
        if not 0 <= rate <= 100:
            raise InvalidRateException(rate)

        pitch = self._interpolate_pitch(pitch)
        rate = self._interpolate_rate(rate)

        text = self._delete_dashes(text)
        text = self._delete_html_brackets(text)
        text = self.normalize(text)
        tensor = self._generate_audio(model, text, speaker, sample_rate, pitch, rate)
        return self._convert_to_wav(tensor, sample_rate)

    def _load_model(self, model_path: Path):
        package = PackageImporter(model_path)
        model: "TTSModelMultiAcc_v3" = package.load_pickle("tts_models", "model")
        if model.device != device:
            model.to(device)

        language = model_path.stem[3:]  # remove prefix "v3_" or "v4_"
        self.models[language] = model

        self._load_speakers(model, language)

    def _load_speakers(self, model: "TTSModelMultiAcc_v3", language: str):
        if "random" in model.speakers:
            model.speakers.remove("random")

        self.speakers[language] = model.speakers
        for speaker in model.speakers:
            self.model_by_speaker[speaker] = model

    def _delete_dashes(self, text: str) -> str:
        # This fixes the problem:
        # https://github.com/twirapp/silero-tts-api-server/issues/8
        return text.replace("-", "").replace("‑", "")

    def _delete_html_brackets(self, text: str) -> str:
        # Safeguarding against pitch and rate modifications with HTML tags in text.
        # And also prevents raising the error of generation of audio `ValueError`, if there is html tags.
        return text.replace("<", "").replace(">", "")

    def _interpolate_pitch(self, pitch: int) -> int:
        # One interesting feature of the models is that when a pitch of -100 is input,
        # it transforms to `1.0 + (-100 / 100) = 0`, making the sound equivalent to generating `1.0 + (0 / 100) = 1`.
        # This makes the voice the same for 0 and 1
        if pitch == 0:
            return -101

        SCALE_FACTOR = 2
        OFFSET = -100
        return pitch * SCALE_FACTOR + OFFSET

    def _interpolate_rate(self, rate: int) -> int:
        OFFSET = 50
        return rate + OFFSET

    def _generate_audio(
        self,
        model: "TTSModelMultiAcc_v3",
        text: str,
        speaker: str,
        sample_rate: int,
        pitch: int,
        rate: int,
    ) -> torch.Tensor.cuda:
        ssml_text = f"<speak><prosody pitch='+{pitch}%' rate='{rate}%'>{text}</prosody></speak>"
        try:
            return model.apply_tts(
                ssml_text=ssml_text, speaker=speaker, sample_rate=sample_rate
            )
        except ValueError:
            raise NotCorrectTextException(text)
        except Exception as error:
            if str(error) == "Model couldn't generate your text, probably it's too long":
                raise TextTooLongException(text)
            raise

    def _convert_to_wav(self, tensor: torch.Tensor, sample_rate: int) -> bytes:
        audio = self._normalize_audio(tensor)
        with BytesIO() as buffer, wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)  # mono
            wav.setsampwidth(2)  # quality is 16 bit. Do not change
            wav.setframerate(sample_rate)
            wav.writeframes(audio)

            buffer.seek(0)
            return buffer.read()

    def _normalize_audio(self, tensor: torch.Tensor):
        audio: np.ndarray = tensor.numpy() * MAX_INT16
        return audio.astype(np.int16)

    morph = MorphAnalyzer(lang='ru')

    def normalize_date(text: str) -> str:
        return text

    def normalize_time(text: str) -> str:
        return text

    def normalize_number(text: str) -> str:
        number_strings = findall(
            r'(?<![a-zA-Z\d])\d+(?:\.\d+)?(?:(?:\s|\w)*?<d>.*?</d>)*(?!(?:[a-zA-Z\d\"\']|\s)*\'?/?>)',
            text)

        for number_string in number_strings:
            number_data = number_string.split(' ')

            number = num2words(number_data[0], lang='ru')
            number_gender = None

            inflected_words = []

            for i in range(1, len(number_data)):
                if '<d>' not in number_data[i]:
                    inflected_words.append(number_data[i])
                    continue

                word_to_declension = morph.parse(number_data[i][3:-4])[0]

                if not number_gender:
                    number_gender = word_to_declension.tag.gender

                inflected_word = word_to_declension.make_agree_with_number(float(number_data[0]))

                if inflected_word:
                    word_to_declension = inflected_word

                inflected_words.append(word_to_declension.word)

            last_number_word = morph.parse(number.split(' ')[-1])[0]

            if number_gender:
                inclined_number = last_number_word.inflect({number_gender})

                if inclined_number:
                    numbers = number.split(' ')
                    numbers.pop()
                    numbers.append(inclined_number.word)
                    number = ' '.join(numbers)

            inflected_words.insert(0, number)
            text = text.replace(number_string, ' '.join(inflected_words))

        return text

    def translit_text(text: str) -> str:
        tag_empty_text = sub('<[^>]*>', '', text)
        english_words = findall(r'[a-zA-Z]+', tag_empty_text)

        for word in english_words:
            result = translit(word, "ru")
            text = text.replace(word, result)

        return text

    def normalize(text: str) -> str:
        text = " ".join(text.split())
        text = normalize_number(text)
        text = translit_text(text)

        return text

tts = TTS()
