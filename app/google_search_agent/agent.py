"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```
"""

import os
import asyncio
import io
import logging
import traceback
import time

import cv2
import pyaudio
import PIL.Image

import argparse
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.types import Type

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Logging setup (file only, non-blocking) ──────────────────────────
LOG_FILE = "live_api_debug.log"
logger = logging.getLogger("liveapi")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s",
                                    datefmt="%H:%M:%S"))
logger.addHandler(_fh)
logger.propagate = False  # don't spam console

# ── Knowledge base (RAG) logging ─────────────────────────────────────
RAG_LOG_FILE = "knowledge_base.log"
rag_logger = logging.getLogger("knowledge_base")
rag_logger.setLevel(logging.INFO)
_rag_fh = logging.FileHandler(RAG_LOG_FILE, mode="w", encoding="utf-8")
_rag_fh.setFormatter(logging.Formatter("%(asctime)s  [RAG] %(message)s",
                                       datefmt="%H:%M:%S"))
rag_logger.addHandler(_rag_fh)
rag_logger.propagate = False

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
RAG_MODEL = "gemini-2.5-flash"  # model used for RAG queries (non-live)

DEFAULT_MODE = "camera"
CAPTURED_FRAMES_DIR = "captured_frames"


def _save_frame_sync(folder: str, frame_number: int, frame_bytes: bytes) -> None:
    """Save a JPEG frame to disk (blocking; run via asyncio.to_thread)."""
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"frame_{frame_number:06d}.jpg")
        with open(path, "wb") as f:
            f.write(frame_bytes)
    except Exception as e:
        logger.warning("Failed to save frame %d: %s", frame_number, e)


# ── Clients ───────────────────────────────────────────────────────────
# Live API client (v1beta for the realtime session)
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Standard API client for RAG / file-search calls (stable v1)
rag_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# ── File Search Store (RAG) setup ─────────────────────────────────────
STORE_DISPLAY_NAME = "zap-tap-wrap-guide"
_file_search_store_name: str | None = None  # set at startup


def _find_existing_store(display_name: str):
    """Return the first store whose display_name matches, or None."""
    for store in rag_client.file_search_stores.list():
        if store.display_name == display_name:
            rag_logger.info("Found existing File Search Store: %s", store.name)
            return store
    return None


def _wait_for_operation(operation):
    """Poll until a long-running operation completes."""
    poll_count = 0
    while not operation.done:
        time.sleep(3)
        poll_count += 1
        operation = rag_client.operations.get(operation)
        rag_logger.info("  ... indexing in progress (poll %d)", poll_count)
    return operation


def initialize_file_search_store(knowledge_folder: str | None):
    """Create (or reuse) a File Search Store and index all files in *knowledge_folder*."""
    global _file_search_store_name

    if not knowledge_folder:
        rag_logger.info("No --knowledge-folder supplied — RAG tool disabled")
        return

    if not os.path.isdir(knowledge_folder):
        rag_logger.error("Knowledge folder does not exist: %s", knowledge_folder)
        return

    rag_logger.info("Initializing knowledge base from: %s", os.path.abspath(knowledge_folder))

    # 1. Find or create the store
    store = _find_existing_store(STORE_DISPLAY_NAME)
    if store is None:
        rag_logger.info("Creating new File Search Store...")
        store = rag_client.file_search_stores.create(
            config={"display_name": STORE_DISPLAY_NAME}
        )
        rag_logger.info("Created File Search Store: %s", store.name)

    _file_search_store_name = store.name

    # 2. Gather files already in the store so we skip duplicates
    existing_names = set()
    try:
        for doc in rag_client.file_search_stores.documents.list(parent=store.name):
            if doc.display_name:
                existing_names.add(doc.display_name)
        if existing_names:
            rag_logger.info("Already indexed (%d files): %s", len(existing_names), sorted(existing_names))
    except Exception as e:
        rag_logger.warning("Could not list existing documents (store may be empty): %s", e)

    # 3. Upload every file in the folder
    files = [
        os.path.join(knowledge_folder, f)
        for f in os.listdir(knowledge_folder)
        if os.path.isfile(os.path.join(knowledge_folder, f))
    ]

    if not files:
        rag_logger.warning("Knowledge folder is empty: %s", knowledge_folder)
        return

    to_index = [f for f in files if os.path.basename(f) not in existing_names]
    rag_logger.info("Files to index: %d (skipping %d already indexed)", len(to_index), len(files) - len(to_index))

    for filepath in to_index:
        fname = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        rag_logger.info("Uploading: %s (%d bytes)", fname, file_size)
        op = rag_client.file_search_stores.upload_to_file_search_store(
            file=filepath,
            file_search_store_name=store.name,
            config={"display_name": fname},
        )
        _wait_for_operation(op)
        rag_logger.info("Indexed: %s", fname)

    rag_logger.info("Knowledge store ready — %d file(s) from %s", len(files), knowledge_folder)
    print(f"[RAG] Knowledge store ready — {len(files)} file(s) from {knowledge_folder}")


# ── Tool implementations ──────────────────────────────────────────────
# Add your tool functions here. Each function receives **kwargs matching
# the schema you declared, and returns a dict that becomes the response.

def getWeather(city: str = "unknown") -> dict:
    """Stub — replace with a real API call."""
    logger.info("TOOL getWeather called with city=%s", city)
    return {"temperature": "22°C", "condition": "Sunny", "city": city}


def searchKnowledgeBase(query: str = "") -> dict:
    """Query the indexed knowledge base using Gemini File Search (RAG)."""
    rag_logger.info("Query: %s", query[:200] + ("..." if len(query) > 200 else ""))

    if _file_search_store_name is None:
        rag_logger.warning("Knowledge base not initialized")
        return {"error": "Knowledge base not initialized. Start with --knowledge-folder."}

    try:
        response = rag_client.models.generate_content(
            model=RAG_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[_file_search_store_name]
                    )
                )]
            ),
        )
        answer = response.text or "No relevant information found."
        rag_logger.info("Answer (%d chars): %s", len(answer), answer[:150] + ("..." if len(answer) > 150 else ""))
        return {"answer": answer}
    except Exception as e:
        rag_logger.error("Search failed: %s", e)
        return {"error": str(e)}


# Registry: maps function name → Python callable
TOOL_IMPLEMENTATIONS: dict[str, callable] = {
    "getWeather": getWeather,
    "searchKnowledgeBase": searchKnowledgeBase,
}


# ── Tool declarations (sent to the model) ────────────────────────────
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="getWeather",
                description="Get current weather for a city",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "city": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The city name to get weather for",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="searchKnowledgeBase",
                description=(
                    "Search the game rules and guide in the knowledge base. "
                    "Use when the user wants to play, learn the rules, or needs "
                    "guidance. You can see the user via camera — use the rules "
                    "to guide them based on what you observe."
                ),
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "query": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The search query to look up in the knowledge base",
                        ),
                    },
                ),
            ),
        ]
    ),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_LOW",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    tools=tools,
    realtime_input_config=types.RealtimeInputConfig(
        activity_handling=types.ActivityHandling.NO_INTERRUPTION,
    ),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, capture_frames: bool = True):
        self.video_mode = video_mode
        self.capture_frames = capture_frames

        self.audio_in_queue = None

        self.session = None
        self.captured_frames_dir: str | None = None

        self.audio_stream = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            if self.session is not None:
                await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        """Capture one frame, resize, encode as JPEG, return raw bytes."""
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (768, 768))
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    async def get_frames(self):
        """Capture camera frames and send them directly to the session (1 FPS)."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        logger.info("Camera opened")
        video_sent = 0

        while True:
            frame_bytes = await asyncio.to_thread(self._get_frame, cap)
            if frame_bytes is None:
                logger.warning("Camera returned None frame, stopping")
                break

            # Send immediately — no queue, no delay
            if self.session is not None:
                await self.session.send_realtime_input(
                    media=types.Blob(data=frame_bytes, mime_type="image/jpeg")
                )
                video_sent += 1
                logger.debug("SEND video frame #%d", video_sent)
            await asyncio.sleep(1)




        cap.release()

    def _get_screen(self):
        """Grab screen, encode as JPEG, return raw bytes."""
        try:
            import mss  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
        except ImportError as e:
            raise ImportError("Please install mss package using 'pip install mss'") from e
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)

        img = PIL.Image.frombytes("RGB", i.size, i.rgb)
        img.thumbnail([768, 768])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        return image_io.read()

    async def get_screen(self):
        """Capture screen frames and send them directly to the session (1 FPS)."""
        video_sent = 0

        while True:
            frame_bytes = await asyncio.to_thread(self._get_screen)
            if frame_bytes is None:
                break

            # Send immediately — no queue, no delay
            if self.session is not None:
                await self.session.send_realtime_input(
                    media=types.Blob(data=frame_bytes, mime_type="image/jpeg")
                )
                video_sent += 1
                logger.debug("SEND screen frame #%d", video_sent)


            await asyncio.sleep(1.0)

    async def listen_audio(self):
        """Capture mic audio and send directly to the session."""
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        mic_chunks = 0
        muted_chunks = 0
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

            # Software echo gate: don't send mic audio while the model's
            # audio is still being played through the speakers.
            if self.audio_in_queue is not None and self.audio_in_queue.qsize() > 0:
                muted_chunks += 1
                if muted_chunks % 50 == 0:
                    logger.debug("MIC MUTED (echo gate) chunk #%d  (playback_queue=%d)",
                                 muted_chunks, self.audio_in_queue.qsize())
                continue  # drop this mic chunk — speaker is active

            # Send immediately — no queue
            if self.session is not None:
                await self.session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type="audio/pcm")
                )
                mic_chunks += 1
                if mic_chunks % 50 == 0:
                    logger.debug("MIC sent chunk #%d", mic_chunks)

    async def _handle_tool_call(self, tool_call):
        """Execute tool calls from the model and send responses back."""
        for fc in tool_call.function_calls:
            name = fc.name
            args = fc.args or {}
            call_id = fc.id
            logger.info("TOOL_CALL: %s(%s)  id=%s", name, args, call_id)

            impl = TOOL_IMPLEMENTATIONS.get(name)
            if impl is None:
                logger.warning("TOOL_CALL: no implementation for %s — returning error", name)
                result = {"error": f"Unknown function: {name}"}
            else:
                try:
                    result = await asyncio.to_thread(impl, **args)
                except Exception as e:
                    logger.error("TOOL_CALL: %s raised %s", name, e)
                    result = {"error": str(e)}

            logger.info("TOOL_RESPONSE: %s → %s", name, result)
            await self.session.send_tool_response(
                function_responses=types.FunctionResponse(
                    name=name,
                    response=result,
                    id=call_id,
                )
            )

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        turn_num = 0
        while True:
            if self.session is not None:
                turn = self.session.receive()
                turn_num += 1
                audio_chunks_this_turn = 0
                logger.info("──── TURN %d started ────", turn_num)

                async for response in turn:
                    sc = response.server_content

                    # ── Handle tool calls from the model ──
                    if response.tool_call is not None:
                        await self._handle_tool_call(response.tool_call)
                        continue

                    # ── Handle tool call cancellations ──
                    if response.tool_call_cancellation is not None:
                        logger.warning("TOOL_CALL_CANCELLED: ids=%s",
                                       response.tool_call_cancellation.ids)
                        continue

                    # Log non-audio metadata from server_content
                    if sc is not None:
                        # Thought (chain-of-thought text)
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.thought and part.text:
                                    logger.info("THOUGHT: %s", part.text.strip()[:200])
                                elif part.inline_data:
                                    audio_chunks_this_turn += 1
                                elif part.text:
                                    logger.info("TEXT: %s", part.text.strip()[:200])

                        # Turn lifecycle events
                        if sc.interrupted:
                            logger.warning("INTERRUPTED!")
                        if sc.generation_complete:
                            logger.info("GENERATION_COMPLETE  (audio chunks=%d)", audio_chunks_this_turn)
                        if sc.turn_complete:
                            logger.info("TURN_COMPLETE  (audio chunks=%d, playback_queue=%d)",
                                        audio_chunks_this_turn, self.audio_in_queue.qsize())

                        # Transcriptions (if the API sends them)
                        if sc.input_transcription:
                            logger.info("INPUT_TRANSCRIPTION: %s", sc.input_transcription)
                        if sc.output_transcription:
                            logger.info("OUTPUT_TRANSCRIPTION: %s", sc.output_transcription)

                    # Queue audio for playback
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="")

                logger.info("──── TURN %d ended  (playback_queue=%d) ────",
                            turn_num, self.audio_in_queue.qsize())
                # NOTE: With NO_INTERRUPTION mode, do NOT drain audio queue.

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        played = 0
        while True:
            if self.audio_in_queue is not None:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
                played += 1
                if played % 50 == 0:
                    logger.debug("PLAY chunk #%d  (playback_queue=%d)", played, self.audio_in_queue.qsize())

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                logger.info("Session connected to %s", MODEL)

                self.audio_in_queue = asyncio.Queue()

                if self.capture_frames and self.video_mode != "none":
                    self.captured_frames_dir = os.path.join(
                        CAPTURED_FRAMES_DIR,
                        time.strftime("%Y-%m-%d_%H-%M-%S"),
                    )
                    os.makedirs(self.captured_frames_dir, exist_ok=True)
                    logger.info("Capturing frames to %s", self.captured_frames_dir)
                logger.info("Session ready — direct send (no queue)")

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream is not None:
                self.audio_stream.close()
                traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--knowledge-folder",
        type=str,
        default=None,
        help="Path to a folder of documents to index for RAG (PDF, TXT, DOCX, etc.)",
    )
    parser.add_argument(
        "--no-capture-frames",
        action="store_true",
        help="Disable saving sent video frames to captured_frames/",
    )
    args = parser.parse_args()

    # Index knowledge base before starting the live session
    initialize_file_search_store(args.knowledge_folder)

    main = AudioLoop(video_mode=args.mode, capture_frames=not args.no_capture_frames)
    asyncio.run(main.run())
