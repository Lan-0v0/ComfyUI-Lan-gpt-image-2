# ComfyUI-Lan-gpt-image-2 插件说明文档

## 插件简介

本插件参考了 https://github.com/oshtz/ComfyUI-oshtz-nodes 项目，但解决了其核心缺陷：原插件的 API 地址被硬编码为 https://api.openai.com/v1，无法使用自定义代理或本地端点。本插件将 base\_url 作为节点输入参数暴露出来，你可以填写任意兼容端点，例如：

* OpenAI 官方：https://api.openai.com/v1
* 本地代理（cli-proxy-api）：http://localhost:8317/v1
* Azure OpenAI：你的部署地址
* 任何其他 OpenAI 兼容的图像 API

整个插件只有一个节点：Lan-gpt-image-2



## 功能列表

【核心功能】

1. 自定义 base\_url —— 可连接任意 OpenAI 兼容 API 端点
2. 可配置模型名 —— 支持 gpt-image-1、dall-e-3 或代理暴露的任意模型名
3. 文生图 —— 完整的提示词控制，支持多行文本
4. 图像编辑与局部重绘 —— 连接图像和可选遮罩进行编辑
5. 批量图像编辑 —— 自动循环处理多张参考图像

【参数控制】
6. 质量（quality）：auto / low / medium / high
7. 尺寸（size）：auto / 1024x1024 / 1024x1536 / 1536x1024 / 1024x1792 / 1792x1024
8. 生成数量（n）：1-10 张每请求
9. 背景（background）：opaque（不透明）/ transparent（透明）
10. 内容审核（moderation）：auto / low / none
11. 自动回退（auto\_fallback\_moderation）：当 API 拒绝 none 时自动改用 low 重试
12. 输出格式（output\_format）：png / jpeg / webp
13. 输出压缩率（output\_compression）：0-100，仅对 jpeg/webp 生效
14. 负面提示词（negative\_prompt）：描述不希望出现的内容（需 API 支持）
15. 随机种子（seed）：用于结果复现

【高级功能】
16. 指数退避重试 —— 可配置最大重试次数和基础延迟时间
17. 自定义 HTTP 头（extra\_headers）—— JSON 格式，用于代理认证或自定义路由
18. 保存到磁盘（save\_to\_disk）—— 可选自动保存，文件名带时间戳
19. 调试信息输出（info）—— 第二个输出端口，返回端点、模型、耗时、张量形状等信息
20. 超时控制（timeout）—— 10-600 秒可配
21. 环境变量回退 —— OPENAI\_API\_KEY、OPENAI\_BASE\_URL / OPENAI\_API\_BASE
22. 自动图像缩放 —— 大图自动缩放到 API 限制内
23. Web UI 增强 —— 节点配色、工具提示样式优化



## 安装方法

1. 将 ComfyUI-Lan-gpt-image-2 文件夹复制到 ComfyUI/custom\_nodes/ 目录下
2. 安装依赖（大部分 ComfyUI 已自带）：
pip install -r requirements.txt
3. 重启 ComfyUI
4. 在节点菜单中搜索 "Lan-gpt-image-2"，分类路径为 Lan/gpt-image



## 使用方法

【基础文生图】

1. 添加 Lan-gpt-image-2 节点
2. 在 prompt 字段输入提示词
3. 在 api\_key 字段输入 API 密钥（或设置 OPENAI\_API\_KEY 环境变量）
4. 在 base\_url 字段填写 API 地址（如 http://localhost:8317/v1）
5. 按需调整 quality、size 等参数
6. 运行工作流

【图像编辑 / 局部重绘】

1. 将 IMAGE 输出连接到 image 输入端口
2.（可选）将 MASK 输出连接到 mask 输入端口进行局部编辑
2. 在 prompt 中描述想要的编辑效果
3. 运行工作流，节点会自动使用 /images/edits 端点

【配合 cli-proxy-api 使用】
base\_url：http://localhost:8317/v1
api\_key：你的代理 API 密钥
model：gpt-image-1（或你的代理暴露的模型名）

【使用环境变量】
可以设置环境变量来避免每次手动输入：
set OPENAI\_API\_KEY=sk-...
set OPENAI\_BASE\_URL=http://localhost:8317/v1
将 api\_key 和 base\_url 留空（默认值），节点会自动读取环境变量。

【内容审核 none 模式与自动回退】
将 moderation 设为 none 即可尝试禁用审核。如果代理后端支持该值则直接生效；
如果 API 拒绝该值（返回 400 错误且消息包含 moderation 关键字），且
auto\_fallback\_moderation 为 True（默认），则自动切换到 low 模式重新发送请求。
如果将 auto\_fallback\_moderation 设为 False，则不回退，直接报错。



## 参数说明

参数名                      类型      必填  默认值                        说明
─────────────────────────────────────────────────────────────────────────────────────────────────
prompt                     STRING    是    ""                            图像生成/编辑的文本提示词
api\_key                    STRING    是    ""                            API 密钥（密码字段），留空则读取 OPENAI\_API\_KEY 环境变量
base\_url                   STRING    是    https://api.openai.com/v1     API 基础地址
model                      STRING    否    gpt-image-1                   模型名称
quality                    COMBO     否    auto                          auto / low / medium / high
size                       COMBO     否    auto                          auto / 1024x1024 / 1024x1536 / 1536x1024 / 1024x1792 / 1792x1024
n                          INT       否    1                             生成数量（1-10）
background                 COMBO     否    opaque                        opaque / transparent
moderation                 COMBO     否    auto                          auto / low / none
auto\_fallback\_moderation   BOOLEAN   否    True                          API 拒绝当前审核级别时自动回退到 low
output\_format              COMBO     否    png                           png / jpeg / webp
output\_compression         INT       否    85                            压缩率 0-100，仅对 jpeg/webp 生效
seed                       INT       否    0                             随机种子
negative\_prompt            STRING    否    ""                            负面提示词（需 API 支持）
image                      IMAGE     否    无                            编辑用的参考图像
mask                       MASK      否    无                            局部重绘遮罩（白色=保留，黑色=编辑）
timeout                    INT       否    120                           HTTP 请求超时秒数
max\_retries                INT       否    3                             瞬时失败的最大重试次数
retry\_delay                FLOAT     否    2.0                           重试基础延迟秒数（指数退避）
extra\_headers              STRING    否    ""                            额外 HTTP 头，JSON 格式
save\_to\_disk               BOOLEAN   否    False                         是否保存图像到 output\_dir
output\_dir                 STRING    否    lan\_gpt\_image\_output          保存目录名



## 输出端口

端口名   类型    说明
──────────────────────────────────────────────
images   IMAGE   生成/编辑后的图像批次张量
info     STRING  调试信息（端点、模型、耗时、张量维度等）



## 依赖

requests >= 2.28.0
Pillow >= 9.0.0
numpy >= 1.21.0
torch >= 1.13.0
（以上依赖 ComfyUI 环境通常已自带）

