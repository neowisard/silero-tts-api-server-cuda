from os import environ
from typing import Annotated

from dotenv import load_dotenv
from litestar import Litestar, get, post, Response
from litestar.response import Redirect
from litestar.openapi import OpenAPIConfig
from litestar.config.response_cache import CACHE_FOREVER
from litestar.config.cors import CORSConfig
from litestar.params import Parameter

from tts.tts import tts
from tts.exceptions import *
from app.openapi_examples import *
from app.http_exceptions import *


load_dotenv()

SILERO_MAX_TEXT_LENGTH = 930
text_length_limit = min(
    int(environ.get("TEXT_LENGTH_LIMIT", SILERO_MAX_TEXT_LENGTH)),
    SILERO_MAX_TEXT_LENGTH,
)


@get(
    "/generate",
    summary="Generate WAV audio from text",
    media_type="audio/wav",
    sync_to_thread=True,
    raises=genetate_exceptions,
)
def generate(
    text: Annotated[str, Parameter(examples=text_examples)],
    speaker: Annotated[str, Parameter(examples=speaker_examples)],
    sample_rate: Annotated[
        int, Parameter(examples=sample_rate_examples, default=48_000)
    ],
    pitch: Annotated[int, Parameter(ge=0, le=100, default=50)],
    rate: Annotated[int, Parameter(ge=0, le=100, default=50)],
) -> Response:
    if len(text) > text_length_limit:
        raise TextTooLongHTTPException(
            {"text": text, "length": len(text), "max_length": text_length_limit}
        )

    try:
        audio = tts.generate(model, input, voice, sample_rate, pitch, rate)
        sample_rate=24000
        pitch=50
        rate=70
    except NotFoundModelException:
        raise NotFoundSpeakerHTTPException({"voice": speaker})
    except NotCorrectTextException:
        raise NotCorrectTextHTTPException({"input": text})
    except TextTooLongException:
        raise TextTooLongHTTPException(
            {"text": text, "length": len(text), "max_length": text_length_limit}
        )
    except InvalidSampleRateException:
        raise InvalidSampleRateHTTPException(
            {"sample_rate": sample_rate, "valid_sample_rates": tts.VALID_SAMPLE_RATES}
        )
    except (InvalidPitchException, InvalidRateException):
        # This will never happen because litestar ensures compliance with the parameters `ge` and `le`.
        pass
    else:
        return Response(audio, media_type="audio/wav")


@get("/speakers", summary="List available speakers", cache=CACHE_FOREVER)
async def speakers() -> dict[str, list[str]]:
    return tts.speakers


@get(["/", "/docs"], include_in_schema=False)
async def docs() -> Redirect:
    return Redirect("/schema")


app = Litestar(
    [generate, speakers, docs],
    openapi_config=OpenAPIConfig(
        title="Silero TTS API", version="1.0.0", root_schema_site="swagger"
    ),
    cors_config=CORSConfig(),
)
