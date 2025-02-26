"""DeepSeek API 客户端"""

import json
from typing import AsyncGenerator

from app.utils.logger import logger
from .base_client import BaseClient


async def handle_origin_reasoning(delta: dict) -> AsyncGenerator[tuple[str, str], None]:
    """处理原生推理内容"""
    if delta.get("reasoning_content"):
        content = delta["reasoning_content"]
        logger.debug(f"提取推理内容：{content}")
        yield "reasoning", content
    elif delta.get("content"):
        content = delta["content"]
        logger.info(f"提取内容信息，推理阶段结束: {content}")
        yield "content", content


def get_delta(data: dict) -> dict:
    """从响应数据中提取 delta 信息"""
    return data.get("choices", [{}])[0].get("delta", {}) if data else {}


class DeepSeekClient(BaseClient):
    def __init__(
            self,
            api_key: str,
            api_url: str = "https://api.siliconflow.cn/v1/chat/completions",
    ):
        """初始化 DeepSeek 客户端

        Args:
            api_key: DeepSeek API密钥
            api_url: DeepSeek API地址
        """
        super().__init__(api_key, api_url)
        self.chunk_str_buffer = None
        self.is_collecting_think = None
        self.accumulated_content = None

    def _process_think_tag_content(self, content: str) -> tuple[bool, str]:
        """处理包含 think 标签的内容

        Args:
            content: 需要处理的内容字符串

        Returns:
            tuple[bool, str]:
                bool: 是否检测到完整的 think 标签对
                str: 处理后的内容
        """
        has_start = "<think>" in content
        has_end = "</think>" in content

        if has_start and has_end:
            return True, content
        elif has_start:
            return False, content
        elif not has_start and not has_end:
            return False, content
        else:
            return True, content

    async def stream_chat(
            self,
            messages: list,
            model: str = "deepseek-ai/DeepSeek-R1",
            is_origin_reasoning: bool = True,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话

        Args:
            messages: 消息列表
            model: 模型名称

        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "reasoning" 或 "content"
                内容: 实际的文本内容
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        logger.debug(f"开始流式对话：{data}")

        self.accumulated_content = ""
        self.is_collecting_think = False
        self.chunk_str_buffer = None

        async for chunk in self._make_request(headers, data):
            try:
                # 尝试解码当前chunk
                if self.chunk_str_buffer is not None:
                    # 如果有未解码的数据，与当前chunk合并
                    chunk = self.chunk_str_buffer + chunk
                    self.chunk_str_buffer = None
                
                try:
                    chunk_str = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    # 解码失败，存储当前chunk并等待下一个chunk
                    self.chunk_str_buffer = chunk
                    continue
                
            except Exception as e:
                logger.error(f"处理 chunk 时发生错误: {e}")
                continue
            try:
                for line in chunk_str.splitlines():
                    # 火山的联网bot相比其他模型在data:后少了一个空格，这里去掉，不影响判断
                    if not line.startswith("data:"):
                        continue

                    json_str = line[len("data:"):]
                    if json_str.strip() == "[DONE]":
                        return

                    data = json.loads(json_str)
                    delta = get_delta(data)
                    if not delta:
                        continue

                    # 处理 references（如果存在）
                    if "references" in data:
                        logger.debug("references: " + json.dumps(data["references"], ensure_ascii=False))
                        yield "references", data["references"]

                    if is_origin_reasoning:
                        async for content_type, content in handle_origin_reasoning(delta):
                            yield content_type, content
                    else:
                        async for content_type, content in self._handle_custom_reasoning(delta):
                            yield content_type, content

            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {e}")
            except Exception as e:
                logger.error(f"处理 chunk 时发生错误: {e}")

    async def _handle_custom_reasoning(self, delta: dict) -> AsyncGenerator[tuple[str, str], None]:
        """处理自定义推理内容"""
        if not delta.get("content"):
            return

        content = delta["content"]
        if content == "":
            return

        logger.debug(f"非原生推理内容：{content}")
        self.accumulated_content += content

        is_complete, _ = self._process_think_tag_content(self.accumulated_content)

        if "<think>" in content and not self.is_collecting_think:
            logger.debug(f"开始收集推理内容：{content}")
            self.is_collecting_think = True
            yield "reasoning", content
        elif self.is_collecting_think:
            if "</think>" in content:
                logger.debug(f"推理内容结束：{content}")
                self.is_collecting_think = False
                yield "reasoning", content
                yield "content", ""
                self.accumulated_content = ""
            else:
                yield "reasoning", content
        else:
            yield "content", content
