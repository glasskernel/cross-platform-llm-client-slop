import 'dart:async';
import 'package:get/get.dart';
import 'hive_service.dart';
import '../core/constants.dart';
import 'device_info_service.dart';
import 'npu_inference_android.dart' if (dart.library.html) 'npu_inference_stub.dart';

// Conditionally import llama_flutter_android — only on Android
import 'inference_android.dart' if (dart.library.html) 'inference_stub.dart' as platform;

/// Cross-platform inference service.
/// - Android / iOS: uses llama_flutter_android for local GGUF models
/// - Web: cloud-only mode (local inference coming soon)
class InferenceService extends GetxService {
  final HiveService _hive = Get.find<HiveService>();

  // ── Observable State ──
  final isModelLoaded = false.obs;
  final isGenerating = false.obs;
  final isLoadingModel = false.obs;
  final loadedModelName = ''.obs;
  final tokenCount = 0.obs;
  final modelLoadProgress = 0.0.obs;
  final generationSource = ''.obs;
  final streamingText = ''.obs;
  final gpuName = ''.obs;
  final gpuLayersUsed = 0.obs;
  final isGpuAccelerated = false.obs;

  /// Whether the current platform supports local inference.
  bool get supportsLocalInference => platform.supportsLocalInference;

  // Platform-specific engine
  platform.InferenceEngine? _engine;
  NpuInferenceService? _npuEngine;

  Future<String> loadModel(String modelPath, {String? modelName}) async {
    if (!supportsLocalInference) {
      return 'ERROR: Local inference is not available on this platform. Use Cloud mode.';
    }
    if (isLoadingModel.value) return 'ERROR: Model is already loading.';

    try {
      await unloadModel();
      isLoadingModel.value = true;
      modelLoadProgress.value = 0.0;

      _engine = platform.InferenceEngine();

      final contextSize = _hive.getSetting<int>(
            AppConstants.keyContextSize,
            defaultValue: AppConstants.defaultContextSize,
          ) ?? AppConstants.defaultContextSize;

      final deviceTier = _getDeviceTier();

      final result = await _engine!.loadModel(
        modelPath: modelPath,
        contextSize: contextSize,
        deviceTier: deviceTier,
        onProgress: (p) => modelLoadProgress.value = p,
      );

      isModelLoaded.value = result.success;
      isLoadingModel.value = false;
      modelLoadProgress.value = 1.0;
      loadedModelName.value = modelName ?? modelPath.split('/').last;
      gpuName.value = result.gpuName;
      gpuLayersUsed.value = result.gpuLayers;
      isGpuAccelerated.value = result.gpuLayers > 0;

      await _hive.setSetting(AppConstants.keyLocalModelPath, modelPath);
      await _hive.setSetting(AppConstants.keyLocalModelName, loadedModelName.value);

      return result.message;
    } catch (e) {
      isModelLoaded.value = false;
      isLoadingModel.value = false;
      modelLoadProgress.value = 0.0;
      return 'ERROR: Failed to load model — $e';
    }
  }

  Future<String> loadNpuModel(String modelPath, {String? modelName, bool useNpu = true}) async {
    if (!supportsLocalInference) {
      return 'ERROR: NPU inference requires Android.';
    }
    if (isLoadingModel.value) return 'ERROR: Model is already loading.';

    try {
      await unloadModel();
      isLoadingModel.value = true;
      modelLoadProgress.value = 0.0;

      _npuEngine = NpuInferenceService();
      final maxTokens = _hive.getSetting<int>(
            AppConstants.keyMaxTokens,
            defaultValue: AppConstants.defaultMaxTokens,
          ) ?? AppConstants.defaultMaxTokens;

      await _npuEngine!.loadModel(
        modelPath: modelPath,
        useNpu: useNpu,
        maxTokens: maxTokens,
      );

      isModelLoaded.value = true;
      isLoadingModel.value = false;
      modelLoadProgress.value = 1.0;
      loadedModelName.value = modelName ?? modelPath.split('/').last;
      gpuName.value = useNpu ? 'NPU' : 'GPU (LiteRT)';
      gpuLayersUsed.value = 0;
      isGpuAccelerated.value = true;

      await _hive.setSetting(AppConstants.keyNpuModelPath, modelPath);
      await _hive.setSetting(AppConstants.keyNpuModelName, loadedModelName.value);

      return 'NPU model loaded (${useNpu ? "NPU" : "GPU"}).';
    } catch (e) {
      isModelLoaded.value = false;
      isLoadingModel.value = false;
      modelLoadProgress.value = 0.0;
      _npuEngine = null;
      return 'ERROR: Failed to load NPU model — $e';
    }
  }

  Future<void> unloadModel() async {
    await stopGeneration();
    await _engine?.dispose();
    _engine = null;
    await _npuEngine?.dispose();
    _npuEngine = null;
    isModelLoaded.value = false;
    loadedModelName.value = '';
    gpuLayersUsed.value = 0;
    isGpuAccelerated.value = false;
    gpuName.value = '';
  }

  Future<String> generate({
    required String prompt,
    String? systemPrompt,
    List<Map<String, String>>? conversationHistory,
    String source = 'chat',
    void Function(String token)? onToken,
  }) async {
    // NPU path
    if (_npuEngine != null && isModelLoaded.value) {
      return _generateNpu(prompt: prompt, systemPrompt: systemPrompt,
          conversationHistory: conversationHistory, source: source, onToken: onToken);
    }

    if (!supportsLocalInference || _engine == null || !isModelLoaded.value) {
      return 'ERROR: No model loaded. Go to Models tab to download and load one.';
    }

    if (isGenerating.value) {
      // Wait for previous generation
      for (int i = 0; i < 10; i++) {
        await Future.delayed(const Duration(milliseconds: 500));
        if (!isGenerating.value) break;
      }
      if (isGenerating.value) {
        await stopGeneration();
        await Future.delayed(const Duration(milliseconds: 300));
      }
    }

    isGenerating.value = true;
    tokenCount.value = 0;
    generationSource.value = source;
    streamingText.value = '';

    try {
      final temperature = _hive.getSetting<double>(
            AppConstants.keyTemperature,
            defaultValue: AppConstants.defaultTemperature,
          ) ?? AppConstants.defaultTemperature;

      final maxTokens = _hive.getSetting<int>(
            AppConstants.keyMaxTokens,
            defaultValue: AppConstants.defaultMaxTokens,
          ) ?? AppConstants.defaultMaxTokens;

      final result = await _engine!.generate(
        prompt: prompt,
        conversationHistory: conversationHistory,
        systemPrompt: systemPrompt ?? AppConstants.systemPrompt,
        modelName: loadedModelName.value,
        maxTokens: maxTokens,
        temperature: temperature,
        onToken: (token) {
          tokenCount.value++;
          streamingText.value += token;
          onToken?.call(token);
        },
      );

      isGenerating.value = false;
      generationSource.value = '';
      return result;
    } catch (e) {
      isGenerating.value = false;
      generationSource.value = '';
      streamingText.value = '';
      return 'ERROR: $e';
    }
  }

  Future<void> stopGeneration() async {
    await _engine?.stop();
    await _npuEngine?.stop();
    isGenerating.value = false;
    tokenCount.value = 0;
    generationSource.value = '';
    streamingText.value = '';
  }

  Future<String> _generateNpu({
    required String prompt,
    String? systemPrompt,
    List<Map<String, String>>? conversationHistory,
    String source = 'chat',
    void Function(String token)? onToken,
  }) async {
    if (isGenerating.value) {
      await stopGeneration();
      await Future.delayed(const Duration(milliseconds: 200));
    }

    isGenerating.value = true;
    tokenCount.value = 0;
    generationSource.value = source;
    streamingText.value = '';

    // Build a simple prompt string for LiteRT (no chat template needed — model handles it)
    final sys = systemPrompt ?? AppConstants.systemPrompt;
    final fullPrompt = conversationHistory != null && conversationHistory.isNotEmpty
        ? _buildNpuPrompt(prompt, conversationHistory, sys)
        : '$sys\n\nUser: $prompt\nAssistant:';

    try {
      final result = await _npuEngine!.generate(
        prompt: fullPrompt,
        onToken: (token) {
          tokenCount.value++;
          streamingText.value += token;
          onToken?.call(token);
        },
      );
      isGenerating.value = false;
      generationSource.value = '';
      return result;
    } catch (e) {
      isGenerating.value = false;
      generationSource.value = '';
      streamingText.value = '';
      return 'ERROR: $e';
    }
  }

  String _buildNpuPrompt(String prompt, List<Map<String, String>> history, String sys) {
    final buf = StringBuffer('$sys\n\n');
    final recent = history.length > 8 ? history.sublist(history.length - 8) : history;
    for (final m in recent) {
      final role = m['role'] == 'assistant' ? 'Assistant' : 'User';
      buf.write('$role: ${m['content']}\n');
    }
    buf.write('User: $prompt\nAssistant:');
    return buf.toString();
  }

  String _getDeviceTier() {
    try {
      final device = Get.find<DeviceInfoService>();
      return device.deviceTier.value;
    } catch (_) {
      return 'mid';
    }
  }
}
