/**
 * Provider-specific options for Groq Transcription.
 *
 * Groq's `/openai/v1/audio/transcriptions` endpoint accepts a subset of
 * OpenAI's parameters. See https://console.groq.com/docs/speech-to-text
 * for the full list.
 */
export interface GroqTranscriptionProviderOptions {
  /**
   * The sampling temperature, between 0 and 1. Higher values like 0.8 make
   * the output more random; lower values like 0.2 make it more focused and
   * deterministic. Groq recommends the default of 0.
   */
  temperature?: number
  /**
   * Timestamp granularities to populate for this transcription.
   * `response_format` must be set to `verbose_json` to use this option.
   * Either or both of `word` and `segment` are supported.
   */
  timestamp_granularities?: Array<'word' | 'segment'>
}
