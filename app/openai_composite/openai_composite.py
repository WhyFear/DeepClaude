"""OpenAI 兼容的组合模型服务，用于协调 DeepSeek 和其他 OpenAI 兼容模型的调用"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, List

from app.clients import DeepSeekClient
from app.clients.openai_compatible_client import OpenAICompatibleClient
from app.utils.logger import logger


class OpenAICompatibleComposite:
    """处理 DeepSeek 和其他 OpenAI 兼容模型的流式输出衔接"""

    def __init__(
            self,
            deepseek_api_key: str,
            openai_api_key: str,
            deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
            openai_api_url: str = "",  # 将由具体实现提供
            is_origin_reasoning: bool = True,
    ):
        """初始化 API 客户端

        Args:
            deepseek_api_key: DeepSeek API密钥
            openai_api_key: OpenAI 兼容服务的 API密钥
            deepseek_api_url: DeepSeek API地址
            openai_api_url: OpenAI 兼容服务的 API地址
            is_origin_reasoning: 是否使用原始推理过程
        """
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
        self.openai_client = OpenAICompatibleClient(openai_api_key, openai_api_url)
        self.is_origin_reasoning = is_origin_reasoning

    async def chat_completions_with_stream(
            self,
            messages: List[Dict[str, str]],
            model_arg: tuple[float, float, float, float],
            deepseek_model: str = "deepseek-reasoner",
            target_model: str = "",
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程

        Args:
            messages: 初始消息列表
            model_arg: 模型参数 (temperature, top_p, presence_penalty, frequency_penalty)
            deepseek_model: DeepSeek 模型名称
            target_model: 目标 OpenAI 兼容模型名称

        Yields:
            字节流数据，格式如下：
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": content
                    }
                }]
            }
        """
        # 生成唯一的会话ID和时间戳

        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        output_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        reasoning_queue: asyncio.Queue[str] = asyncio.Queue()

        reasoning_content: List[str] = []
        references: List[Dict[str, str]] = []

        async def handle_references() -> str:
            """处理参考资料并返回格式化的输出"""
            if not references:
                return ""

            logger.debug(f"摘要结果返回给用户,摘要数量: {len(references)}")
            output_references = "\n\n参考资料：\n"
            for idx, ref in enumerate(references, 1):
                output_references += f"[{idx}. {ref['title']}]({ref['url']}): {ref['summary']}\n"
            return output_references

        async def send_chunk(model: str, role: str = "assistant",
                             reasoning_content: str = "", content: str = "") -> None:
            """发送一个数据块"""
            response = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": role,
                        "reasoning_content": reasoning_content,
                        "content": content,
                    }
                }]
            }
            await output_queue.put(f"data: {json.dumps(response)}\n\n".encode("utf-8"))

        async def process_deepseek() -> None:
            """处理 DeepSeek 模型的输出"""
            try:
                async for content_type, content in self.deepseek_client.stream_chat(
                        messages, deepseek_model, self.is_origin_reasoning
                ):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        await send_chunk(deepseek_model, reasoning_content=content)

                    elif content_type == "content":
                        if references:
                            ref_content = await handle_references()
                            await send_chunk(deepseek_model, reasoning_content=ref_content)

                        logger.info(f"DeepSeek 推理完成，推理内容长度：{len(''.join(reasoning_content))}")
                        await reasoning_queue.put("".join(reasoning_content))
                        break

                    elif content_type == "references":
                        logger.debug(f"收到参考信息：{content}")
                        # 返回的就是列表，直接extend
                        references.extend(content)

            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {str(e)}", exc_info=True)
                await reasoning_queue.put("")
            finally:
                logger.info("DeepSeek 任务处理完成")
                await output_queue.put(None)

        async def process_openai() -> None:
            """处理 OpenAI 兼容模型的输出"""
            try:
                reasoning = await reasoning_queue.get()
                if not reasoning:
                    logger.warning("未能获取到有效的推理内容，使用默认提示")
                    reasoning = "获取推理内容失败"

                openai_messages = messages.copy()
                last_message = openai_messages[-1]

                if not openai_messages or last_message.get("role") != "user":
                    raise ValueError("无效的消息列表或最后一条消息不是用户消息")

                # 构建提示内容
                prompt_parts = [
                    "这是我原始的输入:",
                    last_message["content"],
                    "\n这是另一个模型的全部推理过程:",
                    reasoning
                ]

                if references:
                    logger.info(f"参考资料数量: {len(references)}")
                    prompt_parts.extend([
                        "\n这是一些搜索结果(摘要)参考:",
                        str(references)
                    ])

                prompt_parts.append("\n基于上述信息，直接输出你的回应:")

                last_message["content"] = "\n".join(prompt_parts)
                logger.debug(f"构建的提示内容长度: {len(last_message['content'])}")

                async for role, content in self.openai_client.stream_chat(
                        messages=openai_messages,
                        model=target_model,
                ):
                    await send_chunk(target_model, role=role, content=content)

            except Exception as e:
                logger.error(f"处理 OpenAI 兼容流时发生错误: {str(e)}", exc_info=True)
            finally:
                logger.info("OpenAI 兼容任务处理完成")
                await output_queue.put(None)

        # 创建并发任务
        tasks = [
            asyncio.create_task(process_deepseek()),
            asyncio.create_task(process_openai())
        ]

        try:
            # 等待输出完成
            finished_tasks = 0
            while finished_tasks < len(tasks):
                item = await output_queue.get()
                if item is None:
                    finished_tasks += 1
                    continue
                yield item

            # 发送结束标记
            yield b"data: [DONE]\n\n"
        finally:
            # 确保所有任务都被清理
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def chat_completions_without_stream(
            self,
            messages: List[Dict[str, str]],
            model_arg: tuple[float, float, float, float],
            deepseek_model: str = "deepseek-reasoner",
            target_model: str = "",
    ) -> Dict[str, Any]:
        """处理非流式输出请求

        Args:
            messages: 初始消息列表
            model_arg: 模型参数
            deepseek_model: DeepSeek 模型名称
            target_model: 目标 OpenAI 兼容模型名称

        Returns:
            Dict[str, Any]: 完整的响应数据
        """
        full_response = {
            "id": f"chatcmpl-{hex(int(time.time() * 1000))[2:]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": target_model,
            "choices": [],
            "usage": {},
        }

        content_parts = []
        async for chunk in self.chat_completions_with_stream(
                messages, model_arg, deepseek_model, target_model
        ):
            if chunk != b"data: [DONE]\n\n":
                try:
                    response_data = json.loads(chunk.decode("utf-8")[6:])
                    if (
                            "choices" in response_data
                            and len(response_data["choices"]) > 0
                            and "delta" in response_data["choices"][0]
                    ):
                        delta = response_data["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            content_parts.append(delta["content"])
                except json.JSONDecodeError:
                    continue

        full_response["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "finish_reason": "stop",
            }
        ]

        return full_response
