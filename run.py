from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# Qwen-7B-Chat, chatglm3-6b Baichuan-13B-Chat
model_name = "./Baichuan-13B-Chat"  # 本地路径
questions = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。"
]

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用float16以节省显存
    device_map="auto"  # 自动管理模型在不同设备上的分布
).eval()

# 依次处理每个问题
for i, question in enumerate(questions):
    # 准备输入
    inputs = tokenizer(question, return_tensors="pt").input_ids
    inputs = inputs.to(model.device)  # 将输入移至模型所在设备
    # 创建流式输出器
    streamer = TextStreamer(tokenizer)
    with torch.no_grad():  # 推理时无需计算梯度
        outputs = model.generate(
            inputs,
            streamer=streamer,
            max_new_tokens=500,
            temperature=0.7,  # 控制生成的随机性
            top_p=0.85,       # 核采样参数
            do_sample=True    # 启用采样
        )