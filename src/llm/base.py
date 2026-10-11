"""Provider-agnostic LLM interface.

Every concrete provider (Gemini, OpenAI, etc.) must implement this interface.
Pipeline stages (meta_profiler.py, translator.py, renderer.py) depend only on
`BaseLLM`, never on a specific provider SDK.

Image input is a separate interface, `VisionLLM`, because it is a separate
model: a provider's text model generally does not take images, and a stage
that needs both asks the factory for each one.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLM provider adapters."""

    @abstractmethod
    def generate(self, system_prompt: str, user_text: str, temperature: float = 0.7) -> str:
        """Generate text from the LLM.

        Args:
            system_prompt: The system/instruction prompt.
            user_text: The user-provided content to process.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """
        raise NotImplementedError


class VisionLLM(ABC):
    """Abstract base class for providers that accept image input.

    Same error contract as `BaseLLM` implementations: transient failures raise
    `RuntimeError` (worth retrying), permanent ones `ValueError`.
    """

    @abstractmethod
    def generate_with_images(
        self,
        system_prompt: str,
        user_text: str,
        images: list[tuple[bytes, str]],
        temperature: float = 0.0,
    ) -> str:
        """Generate text from instructions, text and images.

        Args:
            system_prompt: The instructions. Text only -- providers reject an
                image in a system message.
            user_text: Text sent alongside the images, in the user message.
            images: ``(data, mime_type)`` pairs, e.g. ``(b"...", "image/png")``.
                Use `image_mime_type` to detect the type from the bytes.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """
        raise NotImplementedError


# Magic numbers of the formats vision APIs accept. The type is read from the
# bytes, not the file extension: MinerU names everything `.jpg`.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def image_mime_type(data: bytes) -> str | None:
    """The MIME type of ``data`` if it is JPEG, PNG, GIF or WebP, else None."""
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
