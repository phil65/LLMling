class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], **params) -> dict:
        pass
    
    @abstractmethod
    def transcribe(self, audio_file: Path, **params) -> dict:
        pass
    
    @abstractmethod
    def generate_image(self, prompt: str, **params) -> list[str]:
        pass
