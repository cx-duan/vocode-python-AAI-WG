import asyncio
import logging
import signal
from dotenv import load_dotenv
from vocode.streaming.synthesizer.coqui_tts_synthesizer import CoquiTTSSynthesizer

from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.synthesizer.gtts_synthesizer import GTTSSynthesizer
from vocode.streaming.synthesizer.play_ht_synthesizer import PlayHtSynthesizer
from vocode.streaming.synthesizer.rime_synthesizer import RimeSynthesizer
from vocode.streaming.transcriber.google_transcriber import GoogleTranscriber
from vocode.streaming.transcriber.whisper_cpp_transcriber import WhisperCPPTranscriber

load_dotenv()

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_microphone_input_and_speaker_output
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    GoogleTranscriberConfig,
    WhisperCPPTranscriberConfig,
)
from vocode.streaming.models.agent import (
    ChatGPTAgentConfig,
    CutOffResponse,
    FillerAudioConfig,
    RESTfulUserImplementedAgentConfig,
    WebSocketUserImplementedAgentConfig,
    EchoAgentConfig,
    LLMAgentConfig,
    ChatGPTAgentConfig,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    AzureSynthesizerConfig,
    CoquiTTSSynthesizerConfig,
    ElevenLabsSynthesizerConfig,
    GTTSSynthesizerConfig,
    GoogleSynthesizerConfig,
    PlayHtSynthesizerConfig,
    RimeSynthesizerConfig,
)
import vocode
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriber
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriberConfig


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def main():
    microphone_input, speaker_output = create_microphone_input_and_speaker_output(
        streaming=True, use_default_devices=False
    )
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=AssemblyAITranscriber(
            AssemblyAITranscriberConfig.from_input_device(
                microphone_input, endpointing_config=PunctuationEndpointingConfig()
            )
    # conversation = StreamingConversation(
    #     output_device=speaker_output,
    #     transcriber=DeepgramTranscriber(
    #         DeepgramTranscriberConfig.from_input_device(
    #             microphone_input, endpointing_config=PunctuationEndpointingConfig()
    #         )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello AssemblyAI"),
                prompt_preamble="""You are a helpful AI assistant""",
                generate_responses=True,
                cut_off_response=CutOffResponse(),
            )
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(
                speaker_output, api_key=vocode.getenv("AZURE_SPEECH_KEY")
                )
        ),
        logger=logger,
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: conversation.terminate())
    while conversation.is_active():
        chunk = microphone_input.get_audio()
        if chunk:
            conversation.receive_audio(chunk)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
