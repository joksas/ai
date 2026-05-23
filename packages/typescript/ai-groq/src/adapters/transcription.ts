import OpenAI from "openai";
import { BaseTranscriptionAdapter } from "@tanstack/ai/adapters";
import { toRunErrorPayload } from "@tanstack/ai/adapter-internals";
import { base64ToArrayBuffer, generateId } from "@tanstack/ai-utils";
import { getGroqApiKeyFromEnv, withGroqDefaults } from "../utils/client";
import type {
	TranscriptionOptions,
	TranscriptionResult,
	TranscriptionSegment,
} from "@tanstack/ai";
import type OpenAI_SDK from "openai";
import type { GroqTranscriptionProviderOptions } from "../audio/transcription-provider-options";
import type { GroqTranscriptionModel } from "../model-meta";
import type { GroqClientConfig } from "../utils/client";

/**
 * Configuration for the Groq Transcription adapter.
 */
export interface GroqTranscriptionConfig extends GroqClientConfig {}

/**
 * Groq Transcription (Speech-to-Text) Adapter
 *
 * Tree-shakeable adapter for Groq audio transcription functionality.
 * Supports whisper-large-v3 and whisper-large-v3-turbo models.
 *
 * @see https://console.groq.com/docs/speech-to-text
 */
export class GroqTranscriptionAdapter<
	TModel extends GroqTranscriptionModel,
> extends BaseTranscriptionAdapter<TModel, GroqTranscriptionProviderOptions> {
	readonly name = "groq" as const;

	protected client: OpenAI;

	constructor(config: GroqTranscriptionConfig, model: TModel) {
		super(model, {});
		this.client = new OpenAI(withGroqDefaults(config));
	}

	async transcribe(
		options: TranscriptionOptions<GroqTranscriptionProviderOptions>,
	): Promise<TranscriptionResult> {
		const { model, audio, language, prompt, responseFormat, modelOptions } =
			options;

		const file = this.prepareAudioFile(audio);

		const responseFormatValue = this.mapResponseFormat(responseFormat);
		const request: OpenAI_SDK.Audio.TranscriptionCreateParams = {
			model,
			file,
			...(modelOptions ?? {}),
		};
		if (language !== undefined) {
			request.language = language;
		}
		if (prompt !== undefined) {
			request.prompt = prompt;
		}
		if (responseFormatValue !== undefined) {
			request.response_format = responseFormatValue;
		}

		const useVerbose = responseFormat === "verbose_json";

		try {
			options.logger.request(
				`activity=transcription provider=${this.name} model=${model} verbose=${useVerbose}`,
				{ provider: this.name, model },
			);
			if (useVerbose) {
				const response = (await this.client.audio.transcriptions.create({
					...request,
					response_format: "verbose_json",
				})) as OpenAI_SDK.Audio.Transcriptions.TranscriptionVerbose;

				const segments = response.segments?.map(
					(seg): TranscriptionSegment => ({
						id: seg.id,
						start: seg.start,
						end: seg.end,
						text: seg.text,
						confidence: Math.exp(seg.avg_logprob),
					}),
				);
				const words = response.words?.map((w) => ({
					word: w.word,
					start: w.start,
					end: w.end,
				}));
				return {
					id: generateId(this.name),
					model,
					text: response.text,
					language: response.language,
					duration: response.duration,
					...(segments !== undefined && { segments }),
					...(words !== undefined && { words }),
				};
			} else {
				const response = await this.client.audio.transcriptions.create(request);

				return {
					id: generateId(this.name),
					model,
					text: typeof response === "string" ? response : response.text,
					...(language !== undefined && { language }),
				};
			}
		} catch (error: unknown) {
			options.logger.errors(`${this.name}.transcribe fatal`, {
				error: toRunErrorPayload(error, `${this.name}.transcribe failed`),
				source: `${this.name}.transcribe`,
			});
			throw error;
		}
	}

	protected prepareAudioFile(audio: string | File | Blob | ArrayBuffer): File {
		if (typeof File !== "undefined" && audio instanceof File) {
			return audio;
		}
		if (typeof Blob !== "undefined" && audio instanceof Blob) {
			this.ensureFileSupport();
			return new File([audio], "audio.mp3", {
				type: audio.type || "audio/mpeg",
			});
		}
		if (typeof ArrayBuffer !== "undefined" && audio instanceof ArrayBuffer) {
			this.ensureFileSupport();
			return new File([audio], "audio.mp3", { type: "audio/mpeg" });
		}
		if (typeof audio === "string") {
			this.ensureFileSupport();

			if (audio.startsWith("data:")) {
				const parts = audio.split(",");
				const header = parts[0];
				const base64Data = parts[1] || "";
				const mimeMatch = header?.match(/data:([^;]+)/);
				const mimeType = mimeMatch?.[1] || "audio/mpeg";
				const bytes = base64ToArrayBuffer(base64Data);
				const extension = mimeType.split("/")[1] || "mp3";
				return new File([bytes], `audio.${extension}`, { type: mimeType });
			}

			const bytes = base64ToArrayBuffer(audio);
			return new File([bytes], "audio.mp3", { type: "audio/mpeg" });
		}

		throw new Error("Invalid audio input type");
	}

	// Throws on Node < 20 where the global `File` constructor isn't available.
	private ensureFileSupport(): void {
		if (typeof File === "undefined") {
			throw new Error(
				"`File` is not available in this environment. " +
					"Use Node.js 20 or newer, or pass a File object directly.",
			);
		}
	}

	// Groq only documents `json`, `text`, and `verbose_json`. `srt` and `vtt`
	// are forwarded as-is - the server will respond with HTTP 400 if it
	// doesn't accept them.
	protected mapResponseFormat(
		format?: "json" | "text" | "srt" | "verbose_json" | "vtt",
	): OpenAI_SDK.Audio.TranscriptionCreateParams["response_format"] {
		if (!format) return "json";
		return format;
	}
}

/**
 * Creates a Groq transcription adapter with explicit API key.
 * Type resolution happens here at the call site.
 *
 * @param model - The model name (e.g., 'whisper-1')
 * @param apiKey - Your Groq API key
 * @param config - Optional additional configuration
 * @returns Configured Groq transcription adapter instance with resolved types
 *
 * @example
 * ```typescript
 * const adapter = createGroqTranscription('whisper-large-v3', "gsk_...");
 *
 * const result = await generateTranscription({
 *   adapter,
 *   audio: audioFile,
 *   language: 'en'
 * });
 * ```
 */
export function createGroqTranscription<TModel extends GroqTranscriptionModel>(
	model: TModel,
	apiKey: string,
	config?: Omit<GroqTranscriptionConfig, "apiKey">,
): GroqTranscriptionAdapter<TModel> {
	return new GroqTranscriptionAdapter({ apiKey, ...config }, model);
}

/**
 * Creates a Groq transcription adapter with automatic API key detection from environment variables.
 * Type resolution happens here at the call site.
 *
 * Looks for `GROQ_API_KEY` in:
 * - `process.env` (Node.js)
 * - `window.env` (Browser with injected env)
 *
 * @param model - The model name (e.g., 'whisper-large-v3')
 * @param config - Optional configuration (excluding apiKey which is auto-detected)
 * @returns Configured Groq transcription adapter instance with resolved types
 * @throws Error if GROQ_API_KEY is not found in environment
 *
 * @example
 * ```typescript
 * // Automatically uses GROQ_API_KEY from environment
 * const adapter = groqTranscription('whisper-large-v3');
 *
 * const result = await generateTranscription({
 *   adapter,
 *   audio: audioFile
 * });
 *
 * console.log(result.text)
 * ```
 */
export function groqTranscription<TModel extends GroqTranscriptionModel>(
	model: TModel,
	config?: Omit<GroqTranscriptionConfig, "apiKey">,
): GroqTranscriptionAdapter<TModel> {
	const apiKey = getGroqApiKeyFromEnv();
	return createGroqTranscription(model, apiKey, config);
}
