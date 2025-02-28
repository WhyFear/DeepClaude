"""基础客户端类,定义通用接口"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import os

import aiohttp
from aiohttp.client_exceptions import ClientError, ServerTimeoutError

from app.utils.logger import logger


class BaseClient(ABC):
    """基础客户端类"""

    # 默认超时设置(秒)
    # total: 总超时时间
    # connect: 连接超时时间
    # sock_read: 读取超时时间
    # TODO: 默认时间的设置涉及到模型推理速度，需要根据实际情况进行调整
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=600, connect=10, sock_read=500)

    def __init__(
        self,
        api_key: str,
        api_url: str,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ):
        """初始化基础客户端

        Args:
            api_key: API密钥
            api_url: API地址
            timeout: 请求超时设置,None则使用默认值
        """
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        # 从环境变量获取代理设置
        self.proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if self.proxy:
            logger.info(f"使用代理: {self.proxy}")

    async def _make_request(
        self, headers: dict, data: dict, timeout: Optional[aiohttp.ClientTimeout] = None
    ) -> AsyncGenerator[bytes, None]:
        """发送请求并处理响应

        Args:
            headers: 请求头
            data: 请求数据
            timeout: 当前请求的超时设置,None则使用实例默认值

        Yields:
            bytes: 原始响应数据

        Raises:
            aiohttp.ClientError: 客户端错误
            ServerTimeoutError: 服务器超时
            Exception: 其他异常
        """
        request_timeout = timeout or self.timeout

        try:
            # 使用 connector 参数来优化连接池
            connector = aiohttp.TCPConnector(limit=100, force_close=True)
            
            # 创建会话时添加代理配置
            session_kwargs = {"connector": connector}
            if self.proxy:
                session_kwargs["proxy"] = self.proxy
                
            async with aiohttp.ClientSession(**session_kwargs) as session:
                async with session.post(
                    self.api_url, headers=headers, json=data, timeout=request_timeout
                ) as response:
                    # 检查响应状态
                    if not response.ok:
                        error_text = await response.text()
                        error_msg = f"API 请求失败: 状态码 {response.status}, 错误信息: {error_text}"
                        logger.error(error_msg)
                        raise ClientError(error_msg)

                    # 流式读取响应内容
                    async for chunk in response.content.iter_any():
                        if chunk:  # 过滤空chunks
                            yield chunk

        except ServerTimeoutError as e:
            error_msg = f"请求超时: {str(e)}"
            logger.error(error_msg)
            raise

        except ClientError as e:
            error_msg = f"客户端错误: {str(e)}"
            logger.error(error_msg)
            raise

        except Exception as e:
            error_msg = f"请求处理异常: {str(e)}"
            logger.error(error_msg)
            raise

    @abstractmethod
    async def stream_chat(
        self, messages: list, model: str
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话，由子类实现

        Args:
            messages: 消息列表
            model: 模型名称

        Yields:
            tuple[str, str]: (内容类型, 内容)
        """
        pass
