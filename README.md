<div>
<h1>DeepClaude 🐬🧠 - OpenAI Compatible（deepclaude & deepgemini）</h1>

<a href="https://github.com/getasterisk/deepclaude"> Inspiration from getasterisk/deepclaude</a>

[![GitHub license](https://img.erlich.fun/personal-blog/uPic/deepclaude.svg)](#)
[![Compatible with](https://img.shields.io/badge/-ChatGPT-412991?style=flat-square&logo=openai&logoColor=FFFFFF)](https://openai.com)

</div>

---
最近更新：

2025-03-01.1:

添加模型组合支持，设置MODEL_LIST参数即可使用。

可设置多个模型，用英文逗号隔开，单个模型支持id:name的格式，方便区分。

claude模型通过claude_provider区分来源。

终端使用的模型，以model_name命名

**示例：**

```json
{
   "reasoning_model": [
      {
         "api_key": "api_key",
         "api_url": "https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions",
         "model": "bot-*****:DeepSeekR1",
         "is_origin_reasoning": true
      },
      {
         "api_key": "api_key",
         "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
         "model": "ep-****:DeepSeekR1",
         "is_origin_reasoning": true
      }
   ],
   "output_model": [
      {
         "api_key": "api_key",
         "api_url": "https://api.anthropic.com/v1/messages",
         "model": "claude-3-5-sonnet-20241022",
         "claude_provider": "anthropic"
      },
      {
         "api_key": "api_key",
         "api_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
         "model": "gemini-2.0-flash-thinking-exp-01-21,gemini-2.0-pro-exp-02-05"
      }
   ],
   "deep_claude_model": [
      {
         "model_name": "deep-gemini-flash",
         "reasoning_model": "bot-*****",
         "output_model": "gemini-2.0-flash-thinking-exp-01-21"
      },
      {
         "model_name": "deep-gemini-pro",
         "reasoning_model": "ep-****",
         "output_model": "gemini-2.0-pro-exp-02-05"
      }
   ]
}
```
<details>
<summary><strong>历史更新日志：</strong></summary>
<div>
2025-03-05.1: 更改docker compose配置, 使用volume将容器配置文件绑定至本地, 避免重启容器时丢失配置. 同时设置失败自动重启.
   
2025-03-02.1: 更新 1.0 版本，支持图形化配置界面，取消 .env 配置，预配置模板，配置更方便

2025-02-28.1: fork，添加火山引擎bots机器人支持，可联网+输出参考内容。

2025-02-25.1: 添加 system message 对于 Claude 3.5 Sonnet 的支持

2025-02-23.1: 重构代码，支持 OpenAI 兼容模型，deepgeminiflash 和 deepgeminipro 配置更方便（请详细查看 READEME 和
.env.example 内的说明）。

2025-02-21.1: 添加 Claude 这段的详细数据结构安全检查。

2025-02-16.1: 支持 claude 侧采用请求体中的自定义模型名称。（如果你采用 oneapi 等中转方，那么现在可以通过配置环境变量或在
API 请求中采用任何 Gemini 等模型完成后半部分。接下来将重构代码，更清晰地支持不同的思考模型组合。）

2025-02-08.2: 支持非流式请求，支持 OpenAI 兼容的 models 接口返回。（⚠️ 当前暂未实现正确的 tokens 消耗统计，稍后更新）

2025-02-08.1: 添加 Github Actions，支持 fork 自动同步、支持自动构建 Docker 最新镜像、支持 docker-compose 部署

2025-02-07.2: 修复 Claude temperature 参数可能会超过范围导致的请求失败的 bug

2025-02-07.1: 支持 Claude temputerature 等参数；添加更详细的 .env.example 说明

2025-02-06.1：修复非原生推理模型无法获得到推理内容的 bug

2025-02-05.1: 支持通过环境变量配置是否是原生支持推理字段的模型，满血版本通常支持

2025-02-04.2: 支持跨域配置，可在 .env 中配置

2025-02-04.1: 支持 Openrouter 以及 OneAPI 等中转服务商作为 Claude 部分的供应商

2025-02-03.3: 支持 OpenRouter 作为 Claude 的供应商，详见 .env.example 说明

2025-02-03.2: 由于 deepseek r1 在某种程度上已经开启了一个规范，所以我们也遵循推理标注的这种规范，更好适配支持的更好的
Cherry Studio 等软件。

2025-02-03.1: Siliconflow 的 DeepSeek R1 返回结构变更，支持新的返回结构

</div>
</details>

# 简介

最近 DeepSeek 推出了 [DeepSeek R1 模型](https://platform.deepseek.com)，在推理能力上已经达到了第一梯队。但是 DeepSeek R1
在一些日常任务的输出上可能仍然无法匹敌 Claude 3.5 Sonnet。Aider
团队最近有一篇研究，表示通过[采用 DeepSeek R1 + Claude 3.5 Sonnet 可以实现最好的效果](https://aider.chat/2025/01/24/r1-sonnet.html)。

<img src="https://img.erlich.fun/personal-blog/uPic/heiQYX.png" alt="deepseek r1 and sonnet benchmark" style="width=400px;"/>

> **R1 as architect with Sonnet as editor has set a new SOTA of 64.0%** on
the [aider polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html). They achieve this at **14X less cost**
compared to the previous o1 SOTA result.

本项目受到该项目的启发，通过 fastAPI 完全重写，经过 15 天大量社区用户的真实测试，我们创作了一些新的组合使用方案。

**1.编程：推荐使用 deepclaude = deepseek r1 + claude 3.7 sonnet;
2.内容创作：推荐使用 deepgeminipro = deepseek r1 + gemini 2.0 pro (该方案可以完全免费使用);
3.日常实验：推荐 deepgeminiflash = deepseek r1 + gemini 2.0 flash (该方案可以完全免费使用)。**

项目**支持 OpenAI 兼容格式的输入输出**，支持 DeepSeek 官方 API 以及第三方托管的 API、生成部分也支持 Claude 官方 API 以及中转
API，并对 OpenAI 兼容格式的其他 Model 做了特别支持。

**🔥推荐使用方法：**
1.用户可以自行运行在自己的服务器，并对外提供开放 API 接口，接入 [OneAPI](https://github.com/songquanpeng/one-api) 等实现统一分发。

2.接入你的日常大语言模型对话聊天使用。

# Implementation

![image-20250201212456050](https://img.erlich.fun/personal-blog/uPic/image-20250201212456050.png)

# How to run

> 项目支持本地运行和服务器运行，推荐使用服务器部署，实现随时随处可访问的最强大语言模型服务，甚至可以完全免费使用。

## 1. 获得运行所需的 API

1. 获取 DeepSeek API，因为最近 DeepSeek 官方的供应能里不足，所以经常无法使用，不推荐。目前更推荐使用派欧算力云的 DeepSeek r1，因为我们对思维链的准确性要求很高，派欧算力云的准确性是目前最好的。并且赠送的额度也是最多的，通过我的邀请码注册可以获得 50 元，可以点击链接注册：https://ppinfra.com/user/register?invited_by=TXTPQF 或者扫码注册：![派欧算力云邀请链接](https://img.erlich.fun/personal-blog/uPic/ppinfra-invite-poster.png)
2. 获取 Claude 的 API KEY：https://console.anthropic.com。(也可采用其他中转服务，如 Openrouter 以及其他服务商的 API KEY)
3. 获取 Gemini 的 API KEY：https://aistudio.google.com/apikey (有免费的额度，日常够用)

## 2. 开始运行（本地运行）

Step 1. 克隆本项目到适合的文件夹并进入项目

```bash
git clone https://github.com/ErlichLiu/DeepClaude.git
cd DeepClaude
```

Step 2. 通过 uv 安装依赖（如果你还没有安装 uv，请看下方注解）

```bash
# 通过 uv 在本地创建虚拟环境，并安装依赖
uv sync
# macOS 激活虚拟环境
source .venv/bin/activate
# Windows 激活虚拟环境
.venv\Scripts\activate
```

Step 3. 本地运行
```bash
uvicorn app.main:app --port 8000
```

Step 4. 打开浏览器访问 http://127.0.0.1:8000/config 输入默认 api key：123456 （如果你运行在云端，请尽快登录后在系统设置内更改，避免被其他人盗用，本地登录则无需更改）
![配置授权页面](https://img.erlich.fun/personal-blog/uPic/HW7YfK.png)

按照提示在“推理模型这一栏”配置一个火山云引擎的 api key，点击编辑，粘贴进去 api key 后点击保存即可
![配置火山云引擎的 api key](https://img.erlich.fun/personal-blog/uPic/PNfOcU.png)

`是否支持原生推理`选项控制了两套针对推理模型返回思考内容.

- 支持原生推理: 推理模型在返回体`reasoning_content`字段返回推理内容, 在`content`字段返回回答内容. 例如:
  - DeepSeek官方 `deepseek-reasoner`
  - Siliconflow `deepseek-ai/deepseek-r1`
- 不支持原生推理: 推理模型在`content`字段中以`<think></think>`标签包裹推理内容返回. 例如:
  - 派欧算力云 `deepseek/deepseek-r1`, `deepseek/deepseek-r1/community`, `deepseek/deepseek-r1-turbo`
  - AiHubMix `aihubmix-DeepSeek-R1`
  - Cluade 3.7 Sonnet Thinking

大多数服务商提供的deepseek-r1均支持原生推理, 所以推荐默认开启. 如果不确定可以在外部使用聊天框架(Chatbox)测试模型响应内容. 如果出现`<think></think>`标签则需要关闭`支持原生推理`选项.

不支持原生推理的deepseek-r1可能需要prompt来触发思考, 若日志中收集到推理内容长度一直为0, 而且出现`<think>`字样, 则考虑检查此因素:

![image](https://github.com/user-attachments/assets/63bf0a9f-19cf-49d4-aa28-e916b2a62138)

    
按照提示在“目标模型”配置一个 Claude 3.7 Sonnet 的 api key 以及一个 Gmeini 的 api key，Gemini 的 api key 可以在：https://aistudio.google.com/apikey 获取
![配置 Claude 3.7 Sonnet 的 api key](https://img.erlich.fun/personal-blog/uPic/ydKSHW.png)
同理，也可以配置一个 Gemini 的 api key 分别到 deepgeminiflash 和 deepgeminipro
![配置 Gemini api key](https://img.erlich.fun/personal-blog/uPic/XGXDkz.png)

Step 5. 配置程序到你的 Chatbox（推荐 [Cherry Studio](https://cherry-ai.com) [NextChat](https://nextchat.dev/)、[ChatBox](https://chatboxai.app/zh)、[LobeChat](https://lobechat.com/)）

**如果你的客户端是 Cherry Studio、Chatbox（选择 OpenAI API 模式，注意不是 OpenAI 兼容模式）**
API 地址为 http://127.0.0.1:8000
API 密钥为默认的 123456，如果你在系统设置内进行修改，则改为你修改过的即可
需要手动配置三个模型，分别为 deepclaude、deepgeminiflash 和 deepgeminipro 模型

**如果你的客户端是 LobeChat**
API 地址为：http://127.0.0.1:8000/v1
API 密钥为默认的 123456，如果你在系统设置内进行修改，则改为你修改过的即可
支持获取模型列表，可以同时获取到 deepclaude、deepgeminiflash 和 deepgeminipro 模型


**注：本项目采用 uv 作为包管理器，这是一个更快速更现代的管理方式，用于替代
pip，你可以[在此了解更多](https://docs.astral.sh/uv/)**

# 部署到服务器

> 项目支持 Docker 服务器部署，可自行调用接入常用的 Chatbox，也可以作为渠道一直，将其视为一个特殊的 `DeepClaude`
模型接入到 [OneAPI](https://github.com/songquanpeng/one-api) 等产品使用。

## Railway 一键部署（推荐）

<details>
<summary><strong>一键部署到 Railway</strong></summary> 

<div>
1. 首先 fork 一份代码。

2. 点击打开 Railway 主页：https://railway.com?referralCode=RNTGCA

3. 点击 `Deploy a new project`
   ![image-20250209164454358](https://img.erlich.fun/personal-blog/uPic/image-20250209164454358.png)

4. 点击 `Deploy from GitHub repo`
   ![image-20250209164638713](https://img.erlich.fun/personal-blog/uPic/image-20250209164638713.png)

5. 点击 `Login with GitHub`
   ![image-20250209164843566](https://img.erlich.fun/personal-blog/uPic/image-20250209164843566.png)

6. 选择升级，选择只需 5 美金的 Hobby Plan 即可
   ![image-20250209165034070](https://img.erlich.fun/personal-blog/uPic/image-20250209165034070.png)
   ![image-20250209165108355](https://img.erlich.fun/personal-blog/uPic/image-20250209165108355.png)

1. 点击 `Create a New Project`
   ![create-a-new-project](https://img.erlich.fun/personal-blog/uPic/rvfGTE.png)

1. 继续选择 `Deploy from GitHub repo`
   ![image-20250209164638713](https://img.erlich.fun/personal-blog/uPic/image-20250209164638713.png)

1. 输入框内搜索`DeepClaude`，选中后点击。
   ![deploy-from-github-repo](https://img.erlich.fun/personal-blog/uPic/ihOzXU.png)

1. 选择`Variable`，并点击`New Variable` 按钮，按照环境变量内的键值对进行填写
   ![variable](https://img.erlich.fun/personal-blog/uPic/VrZgxp.png)

1. 填写完成后重新点击 `Deploy` 按钮，等待数秒后即可完成部署
   ![deploy](https://img.erlich.fun/personal-blog/uPic/5kvkLI.png)

1. 部署完成后，点击 `Settings` 按钮，然后向下查看到 `Networking` 区域，然后选择 `Generate Domain`，并输入 `8000` 作为端口号
   ![networking](https://img.erlich.fun/personal-blog/uPic/PQyAtG.png)
   ![generate-domain](https://img.erlich.fun/personal-blog/uPic/i5JnX8.png)
   ![port](https://img.erlich.fun/personal-blog/uPic/ZEwxRm.png)

1. 接下来就可以在你喜欢的 Chatbox 内配置使用或作为 API 使用了
   ![using](https://img.erlich.fun/personal-blog/uPic/hD8V6e.png)

注：模型名称为 deepclaude 和 deepgemini

</div>
</details>

## 使用 docker-compose 部署（Docker 镜像将随着 main 分支自动更新到最新）

推荐可以使用 `docker-compose.yml` 文件进行部署，更加方便快捷。

1. 确保已安装 Docker Compose。
2. 复制 `docker-compose.yml` 文件到项目根目录。
3. 修改 `docker-compose.yml` 文件中的环境变量配置，将 `your_allow_api_key`，`your_allow_origins`，`your_deepseek_api_key` 和
   `your_claude_api_key` 等值替换为你的实际配置。
4. 在项目根目录下运行 Docker Compose 命令启动服务：

   ```bash
   docker-compose up -d
   ```

服务启动后，DeepClaude API 将在 `http://宿主机IP:8000/v1/chat/completions` 上进行访问。

5. 模型名称为 deepclaude 和 deepgemini

## Docker 部署（自行 Build）

1. **构建 Docker 镜像**

   在项目根目录下，使用 Dockerfile 构建镜像。请确保已经安装 Docker 环境。

   ```bash
   docker build -t deepclaude:latest .
   ```

2. **运行 Docker 容器**

   运行构建好的 Docker 镜像，将容器的 8000 端口映射到宿主机的 8000 端口。同时，通过 `-e` 参数设置必要的环境变量，包括 API
   密钥、允许的域名等。请根据 `.env.example` 文件中的说明配置环境变量。

   ```bash
   docker run -d \
       -p 8000:8000 \
       -e ALLOW_API_KEY=your_allow_api_key \
       -e ALLOW_ORIGINS="*" \
       -e DEEPSEEK_API_KEY=your_deepseek_api_key \
       -e DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions \
       -e DEEPSEEK_MODEL=deepseek-reasoner \
       -e IS_ORIGIN_REASONING=true \
       -e CLAUDE_API_KEY=your_claude_api_key \
       -e CLAUDE_MODEL=claude-3-5-sonnet-20241022 \
       -e CLAUDE_PROVIDER=anthropic \
       -e CLAUDE_API_URL=https://api.anthropic.com/v1/messages \
       -e OPENAI_COMPOSITE_API_KEY=your_gemini_api_key
       -e OPENAI_COMPOSITE_API_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
       -e OPENAI_COMPOSITE_MODEL=gemini-2.0-flash
       -e LOG_LEVEL=INFO \
       --restart always \
       deepclaude:latest
   ```

   请替换上述命令中的 `your_allow_api_key`，`your_allow_origins`，`your_deepseek_api_key` 和 `your_claude_api_key` 为你实际的
   API 密钥和配置。`ALLOW_ORIGINS` 请设置为允许访问的域名，如 `"http://localhost:3000,https://chat.example.com"` 或 `"*"`
   表示允许所有来源。

   # Automatic fork sync

项目已经支持 Github Actions 自动更新 fork 项目的代码，保持你的 fork 版本与当前 main 分支保持一致。如需开启，请 frok 后在
Settings 中开启 Actions 权限即可。

# Technology Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [UV as package manager](https://docs.astral.sh/uv/#project-management)
- [Docker](https://www.docker.com/)

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ErlichLiu/DeepClaude&type=Date)](https://star-history.com/#ErlichLiu/DeepClaude&Date)

# Buy me a coffee

<img src="https://img.erlich.fun/personal-blog/uPic/IMG_3625.JPG" alt="微信赞赏码" style="width: 400px;"/>

# About Me

- Email: erlichliu@gmail.com
- Website: [Erlichliu](https://erlich.fun)
