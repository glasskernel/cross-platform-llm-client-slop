package com.write4me.llama_flutter_android

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import java.util.concurrent.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "NpuInferenceEngine"

class NpuInferenceEngine(private val context: Context) {
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var useNpu: Boolean = false

    /**
     * Load a .task model file.
     * @param modelPath absolute path to the .task file
     * @param useNpu true = Backend.NPU, false = Backend.GPU
     * @param maxTokens max tokens to generate
     */
    fun load(modelPath: String, useNpu: Boolean, maxTokens: Int) {
        this.useNpu = useNpu
        val backend: Backend = if (useNpu) {
            Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        } else {
            Backend.GPU()
        }
        Log.d(TAG, "Loading model: $modelPath, backend: $backend")
        val config = EngineConfig(
            modelPath = modelPath,
            backend = backend,
            maxNumTokens = maxTokens,
        )
        engine = Engine(config).also { it.initialize() }
        conversation = engine!!.createConversation(
            ConversationConfig(
                samplerConfig = if (useNpu) null else SamplerConfig(
                    topK = 40,
                    topP = 0.9,
                    temperature = 0.7,
                ),
            )
        )
        Log.d(TAG, "Model loaded successfully")
    }

    /**
     * Generate a response asynchronously, calling [onToken] for each partial result.
     */
    fun generate(
        prompt: String,
        onToken: (String) -> Unit,
        onDone: () -> Unit,
        onError: (String) -> Unit,
    ) {
        val conv = conversation ?: run {
            onError("NPU model not loaded")
            return
        }
        conv.sendMessageAsync(
            Contents.of(listOf(Content.Text(prompt))),
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    onToken(message.toString())
                }
                override fun onDone() {
                    onDone()
                }
                override fun onError(throwable: Throwable) {
                    if (throwable is CancellationException) {
                        onDone()
                    } else {
                        Log.e(TAG, "Generation error", throwable)
                        onError(throwable.message ?: "NPU generation failed")
                    }
                }
            },
            emptyMap(),
        )
    }

    fun stop() {
        conversation?.cancelProcess()
    }

    fun resetConversation() {
        val eng = engine ?: return
        conversation?.close()
        conversation = eng.createConversation(
            ConversationConfig(
                samplerConfig = if (useNpu) null else SamplerConfig(
                    topK = 40,
                    topP = 0.9,
                    temperature = 0.7,
                ),
            )
        )
    }

    fun dispose() {
        try { conversation?.close() } catch (e: Exception) { Log.e(TAG, "close conversation", e) }
        try { engine?.close() } catch (e: Exception) { Log.e(TAG, "close engine", e) }
        conversation = null
        engine = null
    }

    fun isLoaded(): Boolean = engine != null && conversation != null
}
