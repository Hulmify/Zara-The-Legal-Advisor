# FastAPI Legal Assistant API

## Overview

This FastAPI application serves as a backend for a legal assistant chatbot named **Zara**, powered by a Llama-based language model. Zara is designed to handle legal queries, maintain conversation history, and provide responses in Markdown format.

## Features

- **Stream Responses**: Stream responses to user queries in real time.
- **Session Management**: Maintain session-specific conversation history.
- **Static File Serving**: Serve static files such as an `index.html`.
- **Health Check**: Verify the application and model health.
- **CORS Support**: Enable Cross-Origin Resource Sharing (CORS).

## Requirements

- Python 3.8+
- GPU or MPS support (optional but recommended for better performance)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the Llama model:
   Ensure you have access to the `meta-llama/Llama-3.2-3B` model via [Hugging Face](https://huggingface.co/). Download the model and tokenizer.

4. Set up the `public/` directory:
   Place your `index.html` file and other static resources in the `public/` folder.

## Usage

1. Run the application:

   ```bash
   python app.py
   ```

2. Access the API:
   - Health Check: [http://localhost:8000/health](http://localhost:8000/health)
   - Stream Responses: [http://localhost:8000/ask_stream/{session_id}](http://localhost:8000/ask_stream/{session_id})
   - Clear Session: [http://localhost:8000/clear/{session_id}](http://localhost:8000/clear/{session_id})
   - Clear All Sessions: [http://localhost:8000/clear_all](http://localhost:8000/clear_all)

## Endpoints

### `/ask_stream/{session_id}`

**Method**: `GET`

**Description**: Streams the assistant’s response to a user query.

**Query Parameters**:

- `question` (string): The user’s question.

### `/health`

**Method**: `GET`

**Description**: Checks if the API and model are running properly.

### `/clear/{session_id}`

**Method**: `GET`

**Description**: Clears the session data for the specified session ID.

### `/clear_all`

**Method**: `GET`

**Description**: Clears all session data.

### `/`

**Method**: `GET`

**Description**: Serves the `index.html` file.

## Configuration

### Model Configuration

Modify the `MODEL_NAME` variable in the script to change the model being used.

### Device Configuration

The application automatically detects if a GPU or MPS device is available. Modify the `device` variable if needed.

## Example Usage

1. **Start a session:**

   ```bash
   curl -X GET "http://localhost:8000/ask_stream/{session_id}?question=What+is+contract+law"
   ```

2. **Clear a session:**

   ```bash
   curl -X GET "http://localhost:8000/clear/{session_id}"
   ```

3. **Health check:**
   ```bash
   curl -X GET "http://localhost:8000/health"
   ```

## Notes

- **Session Management**: The application uses an in-memory store for session data, which will be lost if the server restarts. Implement a database-backed store for persistence.
- **Streaming Support**: The `/ask_stream/{session_id}` endpoint streams partial responses to improve responsiveness.

## License

This project is licensed under the [GNU General Public License](LICENSE).

## Author

Built by Zoeb Chhatriwala and powered by **Hulmify**.
