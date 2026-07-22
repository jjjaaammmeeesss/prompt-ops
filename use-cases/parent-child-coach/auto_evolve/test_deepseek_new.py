"""测试公司 DeepSeek key 是否恢复余额"""
from openai import OpenAI

client = OpenAI(
    api_key="DEEPSEEK_API_KEY_PLACEHOLDER",
    base_url="https://api.deepseek.com",
)

for m in ["deepseek-chat", "deepseek-v4-pro", "deepseek-v4-flash"]:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "说'可用'两个字"}],
            max_tokens=10,
            temperature=0.1,
        )
        print(f"✅ {m}: {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"❌ {m}: {str(e)[:150]}")
