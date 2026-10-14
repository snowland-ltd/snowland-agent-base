# Copyright 2026 Snowland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""v0.2 模型接入层测试: 能力类型注册表 / 多能力门面 / AutoBackend / 结果数据类 /
厂商全套子类。全程不依赖网络 —— 真实厂商后端仅校验 ``build()`` 返回的接口类型,
不发起实际请求。
"""
import unittest

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from snowland_agent_base.messages import AnthropicMessage, OpenAIMessage
from snowland_agent_base.model import (
    AutoBackend,
    DeepSeekEmbeddingBackend,
    DeepSeekLLMBackend,
    DeepSeekRerankBackend,
    EmbeddingBackend,
    ImageBackend,
    ImageModel,
    LLMBackend,
    ModelType,
    OpenAICompatVideoBackend,
    OrcaRouterImageBackend,
    OrcaRouterLLMBackend,
    OrcaRouterRerankBackend,
    OrcaRouterTTSBackend,
    OrcaRouterVideoBackend,
    RerankBackend,
    RerankItem,
    RerankModel,
    RerankResult,
    TTSBackend,
    TTSModel,
    VideoBackend,
    VideoModel,
    ZhipuImageBackend,
    ZhipuLLMBackend,
    ZhipuRerankBackend,
    ZhipuVideoBackend,
    build_auto_backend,
    build_embedding,
    build_image,
    build_model,
    build_rerank,
    build_tts,
    build_video,
    create_model,
    get_llm_backend,
    get_message_class,
    list_llm_backends,
    register_llm_backend,
)
from snowland_agent_base.model.base import (
    AudioResult,
    ImageResult,
    VideoResult,
)


# ---------------------------------------------------------------------------
# 假后端: 覆盖全部六种能力类型, 用于在不联网前提下验证门面与注册表的分发逻辑
# ---------------------------------------------------------------------------
class FakeChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    def invoke(self, *args, **kwargs):
        return "ok"


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 4


class FakeImageModel(ImageModel):
    def generate(self, prompt, **kwargs):
        return ImageResult(urls=["https://example.com/x.png"])


class FakeVideoModel(VideoModel):
    def generate(self, prompt, **kwargs):
        return VideoResult(urls=["https://example.com/x.mp4"], task_id="t1")


class FakeTTSModel(TTSModel):
    def synthesize(self, text, **kwargs):
        return AudioResult(audio=b"data", format="mp3")


class FakeRerankModel(RerankModel):
    def rerank(self, query, documents, top_n=None, **kwargs):
        items = [RerankItem(index=i, document=d, relevance_score=float(len(d)))
                 for i, d in enumerate(documents)]
        return RerankResult(results=items)


class FakeLLMBackend(LLMBackend):
    def __init__(self, model=None, **kwargs):
        self.model = model

    def build(self):
        return FakeChatModel()


class FakeEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model=None, **kwargs):
        self.model = model

    def build(self):
        return FakeEmbeddings()


class FakeImageBackend(ImageBackend):
    def __init__(self, model=None, **kwargs):
        self.model = model

    def build(self):
        return FakeImageModel()


class FakeVideoBackend(VideoBackend):
    def __init__(self, model=None, **kwargs):
        self.model = model

    def build(self):
        return FakeVideoModel()


class FakeTTSBackend(TTSBackend):
    def __init__(self, model=None, **kwargs):
        self.model = model

    def build(self):
        return FakeTTSModel()


class FakeRerankBackend(RerankBackend):
    def __init__(self, model=None, **kwargs):
        self.model = model

    def build(self):
        return FakeRerankModel()


VENDOR = "fakemulti"


def _register_fakes():
    register_llm_backend(VENDOR, FakeLLMBackend)                               # REASONING (默认)
    register_llm_backend(VENDOR, FakeEmbeddingBackend, ModelType.EMBEDDING)
    register_llm_backend(VENDOR, FakeImageBackend, ModelType.IMAGE)
    register_llm_backend(VENDOR, FakeVideoBackend, ModelType.VIDEO)
    register_llm_backend(VENDOR, FakeTTSBackend, ModelType.TTS)
    register_llm_backend(VENDOR, FakeRerankBackend, ModelType.RERANK)


_register_fakes()


# ---------------------------------------------------------------------------
# 1. ModelType 与能力类型接口
# ---------------------------------------------------------------------------
class TestModelType(unittest.TestCase):
    def test_members(self):
        self.assertEqual(ModelType.REASONING, ModelType("reasoning"))
        self.assertEqual(str(ModelType.IMAGE), "image")
        self.assertEqual(str(ModelType.RERANK), "rerank")
        for t in ("reasoning", "embedding", "image", "video", "tts", "rerank"):
            self.assertTrue(hasattr(ModelType, t.upper()))

    def test_capability_interfaces_declare_type(self):
        self.assertIs(LLMBackend.model_type, ModelType.REASONING)
        self.assertIs(EmbeddingBackend.model_type, ModelType.EMBEDDING)
        self.assertIs(ImageBackend.model_type, ModelType.IMAGE)
        self.assertIs(VideoBackend.model_type, ModelType.VIDEO)
        self.assertIs(TTSBackend.model_type, ModelType.TTS)
        self.assertIs(RerankBackend.model_type, ModelType.RERANK)


# ---------------------------------------------------------------------------
# 2. 结果数据类
# ---------------------------------------------------------------------------
class TestResultDataclasses(unittest.TestCase):
    def test_image_result_defaults(self):
        r = ImageResult()
        self.assertEqual(r.urls, [])
        self.assertIsNone(r.raw)

    def test_video_result_fields(self):
        r = VideoResult(urls=["u"], task_id="t", raw={"k": 1})
        self.assertEqual(r.urls, ["u"])
        self.assertEqual(r.task_id, "t")
        self.assertEqual(r.raw, {"k": 1})

    def test_audio_result_fields(self):
        r = AudioResult(audio=b"x", format="wav")
        self.assertEqual(r.audio, b"x")
        self.assertEqual(r.format, "wav")

    def test_rerank_result_and_item(self):
        item = RerankItem(index=2, document="doc", relevance_score=0.9)
        result = RerankResult(results=[item])
        self.assertEqual(result.results[0].index, 2)
        self.assertEqual(result.results[0].relevance_score, 0.9)


# ---------------------------------------------------------------------------
# 3. 能力类型注册表
# ---------------------------------------------------------------------------
class TestCapabilityRegistry(unittest.TestCase):
    def test_get_by_type(self):
        cls = get_llm_backend(VENDOR, ModelType.IMAGE)
        self.assertIs(cls, FakeImageBackend)

    def test_list_by_type(self):
        image_vendors = list_llm_backends(ModelType.IMAGE)
        for v in ("dashscope", "orcarouter", "zhipu", "openai-compat"):
            self.assertIn(v, image_vendors)
        # deepseek / anthropic 不提供图像
        self.assertNotIn("deepseek", image_vendors)
        self.assertNotIn("anthropic", image_vendors)

    def test_rerank_vendors_exclude_anthropic(self):
        rerank_vendors = list_llm_backends(ModelType.RERANK)
        for v in ("dashscope", "orcarouter", "zhipu", "deepseek", "openai-compat"):
            self.assertIn(v, rerank_vendors)
        self.assertNotIn("anthropic", rerank_vendors)

    def test_unknown_backend_per_type_raises(self):
        with self.assertRaises(KeyError):
            get_llm_backend("nope", ModelType.VIDEO)

    def test_invalid_cls_rejected(self):
        class NotABackend:
            pass

        with self.assertRaises(TypeError):
            register_llm_backend("bad", NotABackend)

    def test_create_model_by_type(self):
        backend = create_model(ModelType.RERANK, "deepseek", api_key="dummy")
        self.assertIsInstance(backend, DeepSeekRerankBackend)


# ---------------------------------------------------------------------------
# 4. 多能力门面函数
# ---------------------------------------------------------------------------
class TestFacades(unittest.TestCase):
    def test_build_embedding(self):
        self.assertIsInstance(build_embedding(VENDOR), Embeddings)

    def test_build_image(self):
        self.assertIsInstance(build_image(VENDOR), ImageModel)

    def test_build_video(self):
        self.assertIsInstance(build_video(VENDOR), VideoModel)

    def test_build_tts(self):
        self.assertIsInstance(build_tts(VENDOR), TTSModel)

    def test_build_rerank(self):
        self.assertIsInstance(build_rerank(VENDOR), RerankModel)

    def test_build_model_generic_string(self):
        self.assertIsInstance(build_model("image", VENDOR), ImageModel)

    def test_facade_dispatch_wrong_type_raises(self):
        # 默认 dashscope 没有 "fakevendor" 的任何能力
        with self.assertRaises(KeyError):
            build_image("no-such-vendor")


# ---------------------------------------------------------------------------
# 5. AutoBackend 统一后端
# ---------------------------------------------------------------------------
class TestAutoBackend(unittest.TestCase):
    def test_delegation_returns_correct_types(self):
        ab = AutoBackend(vendor=VENDOR)
        self.assertIsInstance(ab.reasoning(), FakeChatModel)
        self.assertIsInstance(ab.embedding(), FakeEmbeddings)
        self.assertIsInstance(ab.image(), FakeImageModel)
        self.assertIsInstance(ab.video(), FakeVideoModel)
        self.assertIsInstance(ab.tts(), FakeTTSModel)

    def test_rerank_shortcut_returns_result(self):
        ab = AutoBackend(vendor=VENDOR)
        result = ab.rerank(query="q", documents=["a", "bb"])
        self.assertIsInstance(result, RerankResult)
        self.assertEqual(len(result.results), 2)

    def test_rerank_without_query_returns_model(self):
        ab = AutoBackend(vendor=VENDOR)
        self.assertIsInstance(ab.rerank(), FakeRerankModel)

    def test_build_by_type(self):
        ab = AutoBackend(vendor=VENDOR)
        self.assertIsInstance(ab.build("video"), FakeVideoModel)

    def test_common_kwargs_merged(self):
        # 通用参数应能被各方法继承
        ab = AutoBackend(vendor=VENDOR, api_key="k")
        self.assertIsInstance(ab.reasoning(), FakeChatModel)

    def test_build_auto_backend_facade(self):
        ab = build_auto_backend(vendor=VENDOR)
        self.assertIsInstance(ab, AutoBackend)


# ---------------------------------------------------------------------------
# 6. 消息风格类解析
# ---------------------------------------------------------------------------
class TestMessageClass(unittest.TestCase):
    def test_reasoning_vendors(self):
        self.assertIs(get_message_class("dashscope"), OpenAIMessage)
        self.assertIs(get_message_class("deepseek"), OpenAIMessage)
        self.assertIs(get_message_class("orcarouter"), OpenAIMessage)

    def test_anthropic_native(self):
        self.assertIs(get_message_class("anthropic"), AnthropicMessage)

    def test_fallback(self):
        self.assertIs(get_message_class(None), OpenAIMessage)
        self.assertIs(get_message_class("unknown-vendor"), OpenAIMessage)


# ---------------------------------------------------------------------------
# 7. 厂商全套子类: 仅校验 build() 返回的接口类型 (不发起网络请求)
# ---------------------------------------------------------------------------
class TestVendorBackendBuild(unittest.TestCase):
    def test_orcarouter_full_suite(self):
        self.assertIsInstance(OrcaRouterLLMBackend(api_key="dummy").build(), BaseChatModel)
        self.assertIsInstance(OrcaRouterImageBackend(api_key="dummy").build(), ImageModel)
        self.assertIsInstance(OrcaRouterVideoBackend(api_key="dummy").build(), VideoModel)
        self.assertIsInstance(OrcaRouterTTSBackend(api_key="dummy").build(), TTSModel)
        self.assertIsInstance(OrcaRouterRerankBackend(api_key="dummy").build(), RerankModel)

    def test_zhipu_full_suite(self):
        self.assertIsInstance(ZhipuLLMBackend(api_key="dummy").build(), BaseChatModel)
        self.assertIsInstance(ZhipuImageBackend(api_key="dummy").build(), ImageModel)
        self.assertIsInstance(ZhipuVideoBackend(api_key="dummy").build(), VideoModel)
        self.assertIsInstance(ZhipuRerankBackend(api_key="dummy").build(), RerankModel)

    def test_deepseek_partial_suite(self):
        self.assertIsInstance(DeepSeekLLMBackend(api_key="dummy").build(), BaseChatModel)
        self.assertIsInstance(DeepSeekEmbeddingBackend(api_key="dummy").build(), Embeddings)
        self.assertIsInstance(DeepSeekRerankBackend(api_key="dummy").build(), RerankModel)

    def test_openai_compat_generic(self):
        backend = OpenAICompatVideoBackend(api_key="dummy", base_url="https://api.example.com/v1")
        self.assertIsInstance(backend.build(), VideoModel)


if __name__ == "__main__":
    unittest.main()
