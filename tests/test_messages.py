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

"""消息模型测试: 历史会话与输出均为 Message 列表。"""
import unittest

from snowland_agent_base.messages import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    AnthropicMessage,
    OpenAIMessage,
    assistant_message,
    ensure_messages,
    render_messages,
    system_message,
    to_langchain_messages,
    tool_message,
    user_message,
)


class TestMessageConstructors(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(system_message("s").to_dict(), {"role": ROLE_SYSTEM, "content": "s"})
        self.assertEqual(user_message("u").to_dict(), {"role": ROLE_USER, "content": "u"})
        self.assertEqual(
            assistant_message("a").to_dict(), {"role": ROLE_ASSISTANT, "content": "a"}
        )

    def test_default_is_openai_style(self):
        self.assertIsInstance(system_message("s"), OpenAIMessage)

    def test_tool_message_with_name(self):
        msg = tool_message("结果", name="search")
        self.assertEqual(msg.role, ROLE_TOOL)
        self.assertEqual(msg.name, "search")
        self.assertEqual(msg.content, "结果")

    def test_tool_message_without_name(self):
        msg = tool_message("结果")
        self.assertIsNone(msg.name)

    def test_anthropic_style(self):
        msg = tool_message(
            "结果", name="search", tool_call_id="call_1", cls=AnthropicMessage
        )
        self.assertIsInstance(msg, AnthropicMessage)
        self.assertEqual(msg.tool_call_id, "call_1")


class TestEnsureMessages(unittest.TestCase):
    def test_none(self):
        self.assertEqual(ensure_messages(None), [])

    def test_empty_string(self):
        self.assertEqual(ensure_messages(""), [])

    def test_string_compat(self):
        """兼容旧的字符串用法: 包装为单条系统消息。"""
        result = ensure_messages("历史文本")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, ROLE_SYSTEM)
        self.assertEqual(result[0].content, "历史文本")

    def test_list_passthrough(self):
        msgs = [user_message("hi"), assistant_message("hello")]
        self.assertEqual(ensure_messages(msgs), msgs)

    def test_list_is_copied(self):
        """应返回副本, 避免调用方列表被外部修改污染。"""
        original = [user_message("hi")]
        copied = ensure_messages(original)
        copied.append(user_message("more"))
        self.assertEqual(len(original), 1)

    def test_dict_items_converted(self):
        result = ensure_messages([{"role": ROLE_USER, "content": "x"}])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], OpenAIMessage)
        self.assertEqual(result[0].content, "x")


class TestRenderMessages(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(render_messages([]), "")

    def test_with_role(self):
        text = render_messages([user_message("你好"), assistant_message("嗨")])
        self.assertIn("user: 你好", text)
        self.assertIn("assistant: 嗨", text)

    def test_without_role(self):
        text = render_messages([user_message("你好")], include_role=False)
        self.assertEqual(text, "你好")

    def test_tool_name_in_prefix(self):
        text = render_messages([tool_message("结果", name="search")])
        self.assertIn("tool(search)", text)

    def test_skips_empty_content(self):
        self.assertEqual(render_messages([user_message("")]), "")


class TestToLangchainMessages(unittest.TestCase):
    def test_role_mapping_openai(self):
        msgs = to_langchain_messages(
            [
                system_message("s"),
                user_message("u"),
                assistant_message("a"),
            ]
        )
        self.assertEqual(
            [type(m).__name__ for m in msgs],
            ["SystemMessage", "HumanMessage", "AIMessage"],
        )

    def test_tool_becomes_system_message_openai(self):
        """OpenAI 风格默认未用原生 tool use, 无 tool_call_id 时观测以系统消息注入。"""
        msgs = to_langchain_messages([tool_message("结果", name="search")])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(type(msgs[0]).__name__, "SystemMessage")
        self.assertIn("search", msgs[0].content)

    def test_tool_becomes_tool_message_anthropic(self):
        """Anthropic 风格带 tool_call_id 时转为原生 ToolMessage。"""
        msgs = to_langchain_messages(
            [tool_message("结果", name="search", tool_call_id="call_1", cls=AnthropicMessage)]
        )
        self.assertEqual(len(msgs), 1)
        self.assertEqual(type(msgs[0]).__name__, "ToolMessage")
        self.assertEqual(msgs[0].tool_call_id, "call_1")

    def test_empty_content_skipped(self):
        self.assertEqual(to_langchain_messages([user_message("")]), [])


if __name__ == "__main__":
    unittest.main()
