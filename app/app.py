from os import environ
from typing import Annotated

from dotenv import load_dotenv
from litestar import Litestar, get, post, Response
from litestar.response import Redirect
from litestar.openapi import OpenAPIConfig
from litestar.config.response_cache import CACHE_FOREVER
from litestar.config.cors import CORSConfig
from litestar.params import Body

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


@post(
    "/generate",
    summary="Generate WAV audio from text",
    media_type="audio/wav",
    sync_to_thread=True,
    raises=genetate_exceptions,
)
def generate(
    data: Annotated[
        dict,
        Body(
            examples=text_examples,
            description="Data containing text, speaker, sample_rate, pitch, and rate",
        ),
    ],
) -> Response:
    text = data.get("text", "")
    speaker = data.get("speaker", "")
    sample_rate = data.get("sample_rate", 48_000)
    pitch = data.get("pitch", 50)
    rate = data.get("rate", 50)

    if len(text) > text_length_limit:
        raise TextTooLongHTTPException(
            {"text": text, "length": len(text), "max_length": text_length_limit}
        )

    try:
        audio = tts.generate(text, speaker, sample_rate, pitch, rate)
    except NotFoundModelException:
        raise NotFoundSpeakerHTTPException({"speaker": speaker})
    except NotCorrectTextException:
        raise NotCorrectTextHTTPException({"text": text})
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