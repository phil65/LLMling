"""Speech-to-text transcription API call implementation."""

from pydantic import BaseModel, Field, HttpUrl
import litellm
import requests
from pathlib import Path

from jinjarope.llm.calls.base import APICall


class TranscriptionParameters(BaseModel):
    """Parameters for speech-to-text transcription."""

    model: str = "whisper-1"
    """Model to use for transcription"""

    language: str | None = None
    """Optional language code (e.g., 'en', 'es', 'fr')"""

    prompt: str | None = None
    """Optional prompt to guide the transcription"""

    response_format: str = "json"
    """Response format (json, text, srt, verbose_json, or vtt)"""

    temperature: float = Field(default=0, ge=0, le=1)
    """Sampling temperature for non-deterministic results"""


class SpeechToTextCall(APICall):
    """API call for speech-to-text transcription."""

    audio_url: HttpUrl | None = None
    """URL of the audio file to transcribe"""

    audio_path: str | None = None
    """Local path to audio file to transcribe"""

    parameters: TranscriptionParameters = Field(default_factory=TranscriptionParameters)
    """Transcription-specific parameters"""

    def execute(self) -> dict:
        """Execute the transcription API call."""
        if not self.audio_url and not self.audio_path:
            msg = "Either audio_url or audio_path must be provided"
            raise ValueError(msg)

        # Handle audio file
        if self.audio_url:
            # Download file from URL
            response = requests.get(str(self.audio_url))
            temp_file = Path("temp_audio")
            temp_file.write_bytes(response.content)
            audio_file = temp_file
        else:
            audio_file = Path(self.audio_path)

        try:
            # Perform transcription using litellm
            with open(audio_file, "rb") as audio:
                transcription = litellm.transcription(
                    model=self.parameters.model,
                    file=audio,
                    language=self.parameters.language,
                    prompt=self.parameters.prompt,
                    response_format=self.parameters.response_format,
                    temperature=self.parameters.temperature,
                )

            audio_source = {
                "type": "url" if self.audio_url else "file",
                "source": str(self.audio_url) if self.audio_url else self.audio_path,
            }

            return {
                "call_id": str(self.call_id),
                "name": self.name,
                "description": self.description,
                "audio": audio_source,
                "parameters": self.parameters.model_dump(),
                "context_sources": [
                    source.model_dump() for source in self.context_sources
                ],
                "transcription": transcription,
            }
        finally:
            # Cleanup temporary file if we downloaded from URL
            if self.audio_url and temp_file.exists():
                temp_file.unlink()


if __name__ == "__main__":
    # Example usage of SpeechToTextCall
    transcription_call = SpeechToTextCall(
        name="Sample Transcription",
        description="Example of transcribing an audio file",
        audio_url="https://example.com/audio.mp3",
        parameters=TranscriptionParameters(
            model="whisper-1", language="en", temperature=0.3
        ),
    )

    result = transcription_call.execute()
    print("Transcription call configuration:")
    print(result)
