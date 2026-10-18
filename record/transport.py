# «транспорт до плати: SmallWebRTC, звук в обидва боки».

from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True
    )
}

async def make_transport(runner_args: RunnerArguments) -> BaseTransport:
    return await create_transport(runner_args, transport_params)
