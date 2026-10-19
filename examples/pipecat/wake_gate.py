"""Wake-word gate for a Pipecat pipeline: the bot hears nothing until someone says «Гектор».

Put it right after transport.input(). Input audio must be 16 kHz mono int16
(PipelineParams(audio_in_sample_rate=16000)). Simplified from the robot's own gate.
"""
import asyncio
from collections import deque

from huggingface_hub import snapshot_download
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame,
                                   InputAudioRawFrame, TranscriptionFrame, UserStartedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def load_hektor(repo: str = "MartynU/hektor-micro-wakeword-esp32", revision: str = "v1.0") -> MicroWakeWord:
    path = snapshot_download(repo, revision=revision)
    return MicroWakeWord.from_config(f"{path}/manifest.json")


class WakeGate(FrameProcessor):
    """Passes microphone audio downstream only after the wake word.

    Closes after idle_s seconds with no speech from the user or the bot. Keeps preroll_ms of audio
    from before the detection, so "Гектор, what time is it" keeps its first words.
    """

    def __init__(self, model: MicroWakeWord, idle_s: float = 20.0, preroll_ms: int = 300, frame_ms: int = 20):
        super().__init__()
        self._model = model
        self._features = MicroWakeWordFeatures()
        self._idle_s = idle_s
        self._preroll = deque(maxlen=max(1, preroll_ms // frame_ms))
        self._open = False
        self._bot_speaking = False
        self._timer: asyncio.TimerHandle | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, InputAudioRawFrame):
            if isinstance(frame, (UserStartedSpeakingFrame, TranscriptionFrame, BotStartedSpeakingFrame)):
                self._cancel_timer()                  # a turn is in progress
            if isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                if self._open:
                    self._arm_timer()
            await self.push_frame(frame, direction)
            return

        if not self._open and self._heard_wake_word(frame.audio):
            self._open = True
            self._arm_timer()
            for old in self._preroll:
                await self.push_frame(old, direction)
            self._preroll.clear()

        if not self._open:
            self._preroll.append(frame)
        elif not self._bot_speaking:                  # don't send the bot's own voice to STT
            await self.push_frame(frame, direction)

    def _heard_wake_word(self, audio: bytes) -> bool:
        heard = False
        for features in self._features.process_streaming(audio):
            if self._model.process_streaming(features):
                heard = True
        return heard

    def _arm_timer(self):
        self._cancel_timer()
        self._timer = asyncio.get_running_loop().call_later(self._idle_s, self._close)

    def _cancel_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _close(self):
        self._timer = None
        self._open = False
        self._model.reset()
        self._features = MicroWakeWordFeatures()
