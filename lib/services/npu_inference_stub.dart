/// Web stub — NPU inference is not available on web.
class NpuInferenceService {
  Future<void> loadModel({required String modelPath, bool useNpu = true, int maxTokens = 1024}) async {
    throw UnsupportedError('NPU inference is not available on web.');
  }
  Future<String> generate({required String prompt, void Function(String)? onToken}) async {
    throw UnsupportedError('NPU inference is not available on web.');
  }
  Future<void> stop() async {}
  Future<void> resetConversation() async {}
  Future<void> dispose() async {}
  Future<bool> isLoaded() async => false;
}
