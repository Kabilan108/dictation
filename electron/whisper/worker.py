#!//mnt/arrakis/sietch/projects/dictator/electron/whisper/.venv/bin/python
# worker.py
#
# A program that reads from STDIN and executes commands.

from contextlib import contextmanager
from pathlib import Path
from typing import Any
from enum import Enum
import sys

from rich.progress import Progress, SpinnerColumn, TextColumn
from pydantic import BaseModel
from torch import cuda
import typer

app = typer.Typer()


class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


MODELS = {
    ModelSize.SMALL: "distil-whisper/distil-small.en",
    ModelSize.MEDIUM: "distil-whisper/distil-medium.en",
    ModelSize.LARGE: "distil-whisper/distil-large-v3",
}


class Status(Enum):
    SUCCESS = "success"
    FAIL = "fail"


class Msg(Enum):
    EXIT = "[exit]"
    ERROR = "[error]"
    READY = "[ready]"
    CH_MODEL = "[ch_model]"
    TRANSCRIBE = "[transcribe]"
    TRANSCRIPT = "[transcript]"


class Response(BaseModel):
    status: Status
    message: str
    data: Any | None = None


def printf(msg: Msg, *args) -> None:
    print(msg.value, *args)
    sys.stdout.flush()


def check_cuda() -> Response:
    """Check if CUDA is available."""

    try:
        return Response(
            status=Status.SUCCESS,
            message="CUDA check success",
            data=cuda.is_available(),
        )
    except Exception as e:
        return Response(
            status=Status.FAIL,
            message=f"CUDA check fail: {e}",
        )


@contextmanager
def progress(description: str):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as pg:
        pg.add_task(description=description, total=None)
        yield pg


def load_model(model_str: str) -> Response:
    """Load model onto GPU."""

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch

    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)

    # TODO: rn this only handles short (<30s) clips efficiently
    # TODO: implement the chunking strategy in distil-whisper README

    try:
        device = "cuda:0" if check_cuda().data else "cpu"
        torch_dtype = torch.float16 if check_cuda().data else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_str,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_str)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
        return Response(
            status=Status.SUCCESS,
            message="Model loaded.",
            data=pipe,
        )
    except Exception as e:
        return Response(
            status=Status.FAIL,
            message=f"Model loading error: {e}",
        )


def transcribe(pipe, audiofile: Path) -> Response:
    """Transcribe audio file."""

    if not audiofile.exists():
        return Response(
            status=Status.FAIL,
            message="File does not exist.",
        )

    try:
        result = pipe(str(audiofile))
        cuda.empty_cache()

        return Response(
            status=Status.SUCCESS,
            message="Transcription success.",
            data=result["text"],
        )
    except Exception as e:
        return Response(
            status=Status.FAIL,
            message=f"Transcription failed: {e}",
        )


@app.command()
def repl(model: ModelSize = typer.Option("medium")) -> None:
    """Start transcription REPL."""

    res = load_model(MODELS[model])
    if res.status == Status.FAIL:
        printf(Msg.ERROR, res.message)
        typer.Exit(code=1)

    pipe = res.data
    del res
    printf(Msg.READY)

    try:
        for line in sys.stdin:
            line = line.strip()

            if line == Msg.EXIT.value:
                return

            if line.startswith(Msg.TRANSCRIBE.value):
                try:
                    audiofile = Path(line.split()[1])
                except IndexError:
                    printf(Msg.ERROR, "no audio file provided")
                    continue

                if audiofile.exists():
                    res = transcribe(pipe, audiofile)
                    printf(Msg.TRANSCRIPT, res.data)
                else:
                    printf(Msg.ERROR, "audio file does not exist")
            elif line.startswith(Msg.CH_MODEL.value):
                try:
                    model = ModelSize(line.split()[1])
                except IndexError:
                    printf(Msg.ERROR, "no model size provided")
                    continue
                except ValueError:
                    printf(Msg.ERROR, "invalid model size")
                    continue

                del pipe
                cuda.empty_cache()

                res = load_model(MODELS[model])
                if res.status == Status.FAIL:
                    printf(Msg.ERROR, res.message)
                    continue

                pipe = res.data
                del res
                printf(Msg.READY)
            else:
                printf(Msg.ERROR, "unknown command")
    except KeyboardInterrupt:
        return
    except Exception as e:
        printf(Msg.ERROR, f"unexpected error: {e}")
        return


@app.command("transcribe")
def main(audiofile: Path, model: ModelSize = typer.Option("medium")) -> None:
    """Transcribe audio file."""

    if not audiofile.exists():
        typer.echo(f"{audiofile} does not exist")
        return

    with progress("Loading model..."):
        res = load_model(MODELS[model])

    if res.status == Status.FAIL:
        typer.echo(res.message)
        return

    pipe = res.data
    del res

    with progress("Transcribing..."):
        res = transcribe(pipe, audiofile)

    if res.status == Status.FAIL:
        typer.echo(res.message)
        return

    typer.echo(res.data)


if __name__ == "__main__":
    app()
