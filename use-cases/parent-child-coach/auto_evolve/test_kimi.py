"""测试 Kimi Code 平台 key (会员订阅)"""
from openai import OpenAI

client = OpenAI(
    api_key="sk-kimi-gpXMnHKus7PXmdMraGgpxo3kQTEUJDisbgLdcnglmUsA2qdp1fcaINm0HCCwmV8K",
    base_url="https://api.kimi.com/coding/v1",
)

resp = client.chat.completions.create(
    model="kimi-for-coding",
    messages=[
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "说'Kimi K3 可用'四个字"},
    ],
)
print("✅ Kimi K3 (Code 平台) 可用")
print("Response:", resp.choices[0].message.content)
