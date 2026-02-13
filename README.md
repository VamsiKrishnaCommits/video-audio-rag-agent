# Video + Audio RAG Agent with Gemini Live API

A real-time video and audio RAG (Retrieval Augmented Generation) agent using Google's Gemini Live API. This agent can process live video from your camera or screen, audio from your microphone, and answer questions using a knowledge base of documents.

## Features

- 🎥 **Live Video Processing**: Stream video from your camera or screen capture
- 🎤 **Real-time Audio**: Voice interaction with the AI agent
- 📚 **RAG Support**: Query a knowledge base of documents using Gemini File Search
- 🔧 **Custom Tools**: Extensible tool system for custom functionality
- 🎯 **Low Latency**: Direct streaming with minimal buffering

## Project Structure

```
aitrain/
├── app/
│   ├── .env.example          # Environment variable template
│   └── google_search_agent/
│       ├── __init__.py
│       └── agent.py          # Main agent implementation
├── knowledge_docs/           # Place your RAG documents here (PDF, TXT, DOCX, etc.)
│   └── game_guide.txt        # Example knowledge document
├── requirements.txt           # Python dependencies
├── verify_setup.py           # Setup verification script
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))
- A microphone (for audio input)
- A webcam (optional, for camera mode) or screen (for screen capture mode)

## Quick Start

### 1. Clone or Download This Repository

```bash
git clone <your-repo-url>
cd aitrain
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Verify installation** (optional):
```bash
python verify_setup.py
```

**Note for macOS users**: If `pyaudio` installation fails, you may need to install PortAudio first:
```bash
brew install portaudio
pip install pyaudio
```

**Note for Linux users**: You may need to install system dependencies:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

### 4. Set Up Environment Variables

Copy the example environment file:
```bash
cp app/.env.example app/.env
```

Edit `app/.env` and replace `PASTE_YOUR_ACTUAL_API_KEY_HERE` with your actual Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

### 5. Set SSL Certificate (Required for Audio/Video)

```bash
export SSL_CERT_FILE=$(python -m certifi)
```

**For permanent setup**, add this to your `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export SSL_CERT_FILE=$(python -m certifi)' >> ~/.bashrc
source ~/.bashrc
```

### 6. Prepare Your Knowledge Base (Optional)

Place your documents (PDF, TXT, DOCX, etc.) in the `knowledge_docs/` folder. The agent will automatically index them when you run with the `--knowledge-folder` flag.

Example:
```bash
mkdir -p knowledge_docs
# Copy your documents here
cp your_documents/*.pdf knowledge_docs/
```

### 7. Run the Agent

**Basic usage (camera mode, no RAG):**
```bash
cd app
python google_search_agent/agent.py
```

**With RAG knowledge base:**
```bash
cd app
python google_search_agent/agent.py --knowledge-folder ../knowledge_docs
```

**Screen capture mode:**
```bash
cd app
python google_search_agent/agent.py --mode screen --knowledge-folder ../knowledge_docs
```

**Audio only (no video):**
```bash
cd app
python google_search_agent/agent.py --mode none --knowledge-folder ../knowledge_docs
```

## Command Line Options

- `--mode`: Video input mode
  - `camera` (default): Use webcam
  - `screen`: Capture screen
  - `none`: Audio only, no video
  
- `--knowledge-folder`: Path to folder containing documents for RAG
  - Example: `--knowledge-folder ../knowledge_docs`
  - Supports: PDF, TXT, DOCX, and other formats supported by Gemini File Search
  
- `--no-capture-frames`: Disable saving captured video frames to disk

## Usage

Once running, the agent will:

1. **Initialize the knowledge base** (if `--knowledge-folder` is provided)
   - Creates or reuses a Gemini File Search Store
   - Uploads and indexes all documents in the folder
   - Logs progress to `knowledge_base.log`

2. **Start the live session**
   - Connects to Gemini Live API
   - Begins streaming audio from your microphone
   - Streams video frames (if enabled)
   - Waits for your input

3. **Interact with the agent**
   - Speak into your microphone (the agent listens continuously)
   - Or type `q` and press Enter to send text messages
   - Type `q` alone to quit

### Example Interactions

- **Voice**: "What do you see?" (if camera/screen mode is enabled)
- **Voice**: "Search the knowledge base for game rules"
- **Voice**: "What is the weather in New York?" (uses the getWeather tool)
- **Text**: Type `q` then enter your question

## How RAG Works

The agent uses Gemini's File Search API to implement RAG:

1. **Indexing**: When you start with `--knowledge-folder`, all documents are uploaded to a Gemini File Search Store
2. **Querying**: When you ask questions, the agent uses the `searchKnowledgeBase` tool
3. **Retrieval**: Gemini searches your indexed documents and returns relevant information
4. **Response**: The agent uses the retrieved context to answer your question

The knowledge base persists across sessions - you only need to re-index when you add new documents.

## Customization

### Adding Custom Tools

Edit `app/google_search_agent/agent.py`:

1. **Implement your tool function**:
```python
def myCustomTool(param: str = "") -> dict:
    """Your tool description."""
    # Your implementation
    return {"result": "data"}
```

2. **Add to the registry**:
```python
TOOL_IMPLEMENTATIONS: dict[str, callable] = {
    "getWeather": getWeather,
    "searchKnowledgeBase": searchKnowledgeBase,
    "myCustomTool": myCustomTool,  # Add here
}
```

3. **Declare the tool schema**:
```python
tools = [
    types.Tool(
        function_declarations=[
            # ... existing tools ...
            types.FunctionDeclaration(
                name="myCustomTool",
                description="What your tool does",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "param": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="Parameter description",
                        ),
                    },
                ),
            ),
        ]
    ),
]
```

### Changing the Model

Edit `MODEL` and `RAG_MODEL` constants in `agent.py`:
```python
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"  # Live API model
RAG_MODEL = "gemini-2.5-flash"  # RAG query model
```

See [Gemini API Models](https://ai.google.dev/gemini-api/docs/models) for available models.

### Changing Voice

Edit the `CONFIG` in `agent.py`:
```python
speech_config=types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
    )
)
```

Available voices: `Zephyr`, `Charon`, `Kore`, `Fenrir`, `Aoede`, `Kore-HD`, `Fenrir-HD`, `Aoede-HD`

## Logging

The agent creates two log files:

- `live_api_debug.log`: Detailed debug logs for the Live API session
- `knowledge_base.log`: RAG indexing and query logs

These are useful for debugging and understanding what the agent is doing.

## Troubleshooting

### Audio Issues

- **No audio input**: Check microphone permissions and ensure `pyaudio` is installed correctly
- **No audio output**: Check speaker/headphone connection and system volume
- **Echo/feedback**: The agent includes echo cancellation, but ensure speakers aren't too close to the microphone

### Video Issues

- **Camera not found**: Ensure your webcam is connected and not in use by another application
- **Screen capture not working**: On Linux, you may need additional permissions or X11 setup

### API Issues

- **Authentication error**: Verify your `GEMINI_API_KEY` is correct in `app/.env`
- **Rate limiting**: You may hit API rate limits with frequent requests
- **File Search Store errors**: Check that your documents are in supported formats

### SSL Certificate Issues

If you see SSL errors, ensure the certificate is set:
```bash
export SSL_CERT_FILE=$(python -m certifi)
```

## Dependencies

- `google-genai`: Google Gemini API client
- `opencv-python`: Video capture and processing
- `pyaudio`: Audio input/output
- `pillow`: Image processing
- `mss`: Screen capture (for screen mode)
- `python-dotenv`: Environment variable management
- `certifi`: SSL certificates

See `requirements.txt` for specific versions.

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

## Support

For issues related to:
- **Gemini API**: See [Google AI Studio Documentation](https://ai.google.dev/)
- **This project**: [Open an issue on GitHub](your-repo-url/issues)
