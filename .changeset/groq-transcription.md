---
'@tanstack/ai-groq': minor
---

Add Groq transcription (speech-to-text) adapter. Exports `GroqTranscriptionAdapter`, `createGroqTranscription`, `groqTranscription`, and the `whisper-large-v3` / `whisper-large-v3-turbo` model identifiers via `GROQ_TRANSCRIPTION_MODELS`. Talks to Groq's OpenAI-compatible `/openai/v1/audio/transcriptions` endpoint.
