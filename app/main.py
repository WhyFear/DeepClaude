import os
import sys
import json

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.deepclaude.deepclaude import DeepClaude
from app.openai_composite import OpenAICompatibleComposite
from app.utils.auth import verify_api_key
from app.utils.logger import logger
from app.config import load_models_config

# 加载环境变量
load_dotenv()

app = FastAPI(title="DeepClaude API")

# 从环境变量获取 CORS配置, API 密钥、地址以及模型名称
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")

# 加载MODEL_LIST配置
MODEL_LIST_STR = os.getenv("MODEL_LIST", "{}")
try:
    MODEL_LIST = json.loads(MODEL_LIST_STR)
    logger.info("成功加载MODEL_LIST配置")
except json.JSONDecodeError as e:
    logger.critical(f"MODEL_LIST格式错误，请检查: {e}")
    sys.exit(1)

# 兼容旧配置
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
ENV_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
CLAUDE_PROVIDER = os.getenv("CLAUDE_PROVIDER", "anthropic")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

OPENAI_COMPOSITE_API_KEY = os.getenv("OPENAI_COMPOSITE_API_KEY")
OPENAI_COMPOSITE_API_URL = os.getenv("OPENAI_COMPOSITE_API_URL")
OPENAI_COMPOSITE_MODEL = os.getenv("OPENAI_COMPOSITE_MODEL")

IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true"

# CORS设置
allow_origins_list = ALLOW_ORIGINS.split(",") if ALLOW_ORIGINS else []  # 将逗号分隔的字符串转换为列表

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建模型实例字典
reasoning_models = {}
claude_models = {}
openai_models = {}
deep_claude_models = {}


# 初始化模型实例
def initialize_models():
    # 初始化推理模型
    for model_config in MODEL_LIST.get("reasoning_model", []):
        model_info_list = model_config["model"].strip().split(",")
        for model_info in model_info_list:
            # 空字符串跳过
            if not model_info:
                continue
            model_id = model_info.split(":")[0] if ":" in model_info else model_info
            reasoning_models[model_id] = {
                "api_key": model_config["api_key"],
                "api_url": model_config["api_url"],
                "is_origin_reasoning": model_config.get("is_origin_reasoning", True)
            }
            logger.info(f"已加载推理模型: {model_id}")

    # 初始化输出模型（包括Claude和OpenAI兼容模型）
    for model_config in MODEL_LIST.get("output_model", []):
        model_info_list = model_config["model"].strip().split(",")
        for model_info in model_info_list:
            # 空字符串跳过
            if not model_info:
                continue
            model_id = model_info.split(":")[0] if ":" in model_info else model_info

            # 判断是Claude模型还是OpenAI兼容模型
            if "claude_provider" in model_config:
                # Claude模型
                claude_models[model_id] = {
                    "api_key": model_config["api_key"],
                    "api_url": model_config["api_url"],
                    "claude_provider": model_config.get("claude_provider", "anthropic")
                }
                logger.info(f"已加载Claude模型: {model_id}")
            else:
                # OpenAI兼容模型
                openai_models[model_id] = {
                    "api_key": model_config["api_key"],
                    "api_url": model_config["api_url"]
                }
                logger.info(f"已加载OpenAI兼容模型: {model_id}")

    # 初始化DeepClaude组合模型
    for model_config in MODEL_LIST.get("deep_claude_model", []):
        model_name = model_config["model_name"]
        reasoning_model = model_config["reasoning_model"]
        output_model = model_config["output_model"]
        deep_claude_models[model_name] = {
            "reasoning_model": reasoning_model,
            "output_model": output_model
        }
        logger.info(f"已加载DeepClaude组合模型: {model_name}")


# 初始化模型
initialize_models()

# 验证日志级别
logger.debug("当前日志级别为 DEBUG")
logger.info("开始请求")


@app.get("/", dependencies=[Depends(verify_api_key)])
async def root():
    logger.info("访问了根路径")
    return {"message": "Welcome to DeepClaude API"}


@app.get("/v1/models")
async def list_models():
    """
    获取可用模型列表
    返回格式遵循 OpenAI API 标准
    """
    try:
        models = []
        # 暂时只加载DeepClaude组合模型
        # 添加DeepClaude组合模型
        for model_id in deep_claude_models:
            models.append({
                "id": model_id,
                "object": "model",
                "created": 1740268800,
                "owned_by": "deepclaude",
                "permission": [
                    {
                        "id": "modelperm-" + model_id,
                        "object": "model_permission",
                        "created": 1740268800,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": model_id,
                "parent": None
            })

        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"加载模型配置时发生错误: {e}")
        return {"error": str(e)}


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    """处理聊天完成请求，支持流式和非流式输出

    请求体格式应与 OpenAI API 保持一致，包含：
    - messages: 消息列表
    - model: 模型名称（必需）
    - stream: 是否使用流式输出（可选，默认为 True)
    - temperature: 随机性 (可选)
    - top_p: top_p (可选)
    - presence_penalty: 话题新鲜度（可选）
    - frequency_penalty: 频率惩罚度（可选）
    """

    try:
        # 1. 获取基础信息
        body = await request.json()
        messages = body.get("messages")
        model = body.get("model")

        if not model:
            raise ValueError("必须指定模型名称")

        # 2. 获取并验证参数
        model_arg = get_and_validate_params(body)
        stream = model_arg[4]  # 获取 stream 参数

        # 3. 根据模型选择不同的处理方式
        if model in deep_claude_models:
            # 使用DeepClaude组合模型
            config = deep_claude_models[model]
            reasoning_model_id = config["reasoning_model"]
            output_model_id = config["output_model"]

            # 获取推理模型配置
            if reasoning_model_id not in reasoning_models:
                raise ValueError(f"未找到推理模型: {reasoning_model_id}")
            reasoning_config = reasoning_models[reasoning_model_id]

            # 判断输出模型类型并获取配置
            if output_model_id in claude_models:
                # Claude输出模型
                output_config = claude_models[output_model_id]
                deep_claude = DeepClaude(
                    reasoning_config["api_key"],
                    output_config["api_key"],
                    reasoning_config["api_url"],
                    output_config["api_url"],
                    output_config["claude_provider"],
                    reasoning_config["is_origin_reasoning"]
                )

                if stream:
                    return StreamingResponse(
                        deep_claude.chat_completions_with_stream(
                            messages=messages,
                            model_arg=model_arg[:4],
                            deepseek_model=reasoning_model_id,
                            claude_model=output_model_id,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    return await deep_claude.chat_completions_without_stream(
                        messages=messages,
                        model_arg=model_arg[:4],
                        deepseek_model=reasoning_model_id,
                        claude_model=output_model_id,
                    )
            elif output_model_id in openai_models:
                # OpenAI兼容输出模型
                output_config = openai_models[output_model_id]
                openai_composite = OpenAICompatibleComposite(
                    reasoning_config["api_key"],
                    output_config["api_key"],
                    reasoning_config["api_url"],
                    output_config["api_url"],
                    reasoning_config["is_origin_reasoning"]
                )

                if stream:
                    return StreamingResponse(
                        openai_composite.chat_completions_with_stream(
                            messages=messages,
                            model_arg=model_arg[:4],
                            deepseek_model=reasoning_model_id,
                            target_model=output_model_id,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    return await openai_composite.chat_completions_without_stream(
                        messages=messages,
                        model_arg=model_arg[:4],
                        deepseek_model=reasoning_model_id,
                        target_model=output_model_id,
                    )
            else:
                raise ValueError(f"未找到输出模型: {output_model_id}")

        elif model in reasoning_models:
            # 直接使用推理模型
            # 这里需要实现直接使用推理模型的逻辑
            raise NotImplementedError("直接使用推理模型的功能尚未实现")

        elif model in claude_models:
            # 直接使用Claude模型
            # 这里需要实现直接使用Claude模型的逻辑
            raise NotImplementedError("直接使用Claude模型的功能尚未实现")

        elif model in openai_models:
            # 直接使用OpenAI兼容模型
            # 这里需要实现直接使用OpenAI兼容模型的逻辑
            raise NotImplementedError("直接使用OpenAI兼容模型的功能尚未实现")

        else:
            # 兼容旧版本的deepclaude模型
            if model == "deepclaude":
                # 使用 DeepClaude
                claude_model = ENV_CLAUDE_MODEL if ENV_CLAUDE_MODEL else "claude-3-5-sonnet-20241022"
                deep_claude = DeepClaude(
                    DEEPSEEK_API_KEY,
                    CLAUDE_API_KEY,
                    DEEPSEEK_API_URL,
                    CLAUDE_API_URL,
                    CLAUDE_PROVIDER,
                    IS_ORIGIN_REASONING,
                )

                if stream:
                    return StreamingResponse(
                        deep_claude.chat_completions_with_stream(
                            messages=messages,
                            model_arg=model_arg[:4],
                            deepseek_model=DEEPSEEK_MODEL,
                            claude_model=claude_model,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    return await deep_claude.chat_completions_without_stream(
                        messages=messages,
                        model_arg=model_arg[:4],
                        deepseek_model=DEEPSEEK_MODEL,
                        claude_model=claude_model,
                    )
            else:
                # 使用 OpenAI 兼容组合模型
                openai_composite = OpenAICompatibleComposite(
                    DEEPSEEK_API_KEY,
                    OPENAI_COMPOSITE_API_KEY,
                    DEEPSEEK_API_URL,
                    OPENAI_COMPOSITE_API_URL,
                    IS_ORIGIN_REASONING,
                )

                if stream:
                    return StreamingResponse(
                        openai_composite.chat_completions_with_stream(
                            messages=messages,
                            model_arg=model_arg[:4],
                            deepseek_model=DEEPSEEK_MODEL,
                            target_model=OPENAI_COMPOSITE_MODEL,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    return await openai_composite.chat_completions_without_stream(
                        messages=messages,
                        model_arg=model_arg[:4],
                        deepseek_model=DEEPSEEK_MODEL,
                        target_model=OPENAI_COMPOSITE_MODEL,
                    )

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        return {"error": str(e)}


def get_and_validate_params(body):
    """提取获取和验证请求参数的函数"""
    # TODO: 默认值设定允许自定义
    temperature: float = body.get("temperature", 0.6)
    top_p: float = body.get("top_p", 1)
    presence_penalty: float = body.get("presence_penalty", 0.0)
    frequency_penalty: float = body.get("frequency_penalty", 0.0)
    stream: bool = body.get("stream", True)

    if "sonnet" in body.get("model", ""):  # Only Sonnet 设定 temperature 必须在 0 到 1 之间
        if (
                not isinstance(temperature, (float))
                or temperature < 0.0
                or temperature > 1.0
        ):
            raise ValueError("Sonnet 设定 temperature 必须在 0 到 1 之间")

    return temperature, top_p, presence_penalty, frequency_penalty, stream
