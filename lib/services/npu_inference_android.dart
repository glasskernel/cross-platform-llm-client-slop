import 'dart:async';
import 'package:flutter/services.dart';

/// Dart interface to the native NPU inference engine (LiteRT LM / MediaPipe).
/// Uses .task model files. Requires Android with a compatible NPU (e.g. Pixel 8+).
class NpuInferenceService {
  static const _channel = MethodChannel('llama_flutter_android/npu');

  StreamController<String>? _tokenController;
  Completer<String>? _completer;
  final StringBuffer _buffer = StringBuffer();

  NpuInferenceService() {
    _channel.setMethodCallHandler(_handleNativeCall);
  }

  Future<dynamic> _handleNativeCall(MethodCall call) async {
    switch (call.method) {
      case 'onNpuToken':
        final token = call.arguments as String? ?? '';
        _buffer.write(token);
        _tokenController?.add(token);
        break;
      case 'onNpuDone':
        _tokenController?.close();
        _tokenController = null;
        if (_completer != null && !_completer!.isCompleted) {
          _completer!.complete(_buffer.toString());
        }
        break;
    }
  }

  /// Load a .task model file.
  /// [useNpu] = true uses Backend.NPU, false falls back to Backend.GPU.
  Future<void> loadModel({
    required String modelPath,
    bool useNpu = true,
    int maxTokens = 1024,
  }) async {
    await _channel.invokeMethod('loadNpuModel', {
      'modelPath': modelPath,
      'useNpu': useNpu,
      'maxTokens': maxTokens,
    });
  }

  /// Generate a response, streaming tokens via [onToken].
  Future<String> generate({
    required String prompt,
    void Function(String token)? onToken,
  }) async {
    _buffer.clear();
    _completer = Completer<String>();
    _tokenController = StreamController<String>.broadcast();

    if (onToken != null) {
      _tokenController!.stream.listen(onToken);
    }

    // Fire-and-forget the native call; result arrives via onNpuDone
    _channel.invokeMethod('generateNpu', {'prompt': prompt}).catchError((e) {
      if (!_completer!.isCompleted) {
        _completer!.completeError(e);
      }
    });

    return _completer!.future;
  }

  Future<void> stop() async {
    await _channel.invokeMethod('stopNpu');
    _tokenController?.close();
    _tokenController = null;
    if (_completer != null && !_completer!.isCompleted) {
      _completer!.complete(_buffer.toString());
    }
  }

  Future<void> resetConversation() async {
    await _channel.invokeMethod('resetNpuConversation');
  }

  Future<void> dispose() async {
    await stop();
    await _channel.invokeMethod('disposeNpu');
  }

  Future<bool> isLoaded() async {
    return await _channel.invokeMethod<bool>('isNpuLoaded') ?? false;
  }
}
