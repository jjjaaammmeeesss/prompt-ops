"""测试两个 key 对 k3 和 kimi-for-coding 的访问权限"""
from openai import OpenAI

keys = {
    "key2 (2Nd...)": "sk-kimi-2NdEZfhRf1WJJMF4jB0eXm1Xzw3NETqSzGH5cLXNOG9owgw2Josa3bLegEUgRnpO",
    "apikey (gpX...)": "sk-kimi-gpXMnHKus7PXmdMraGgpxo3kQTEUJDisbgLdcnglmUsA2qdp1fcaINm0HCCwmV8K",
}
models = ["k3", "kimi-for-coding", "kimi-for-coding-highspeed"]

for kname, key in keys.items():
    for m in models:
        try:
            client = OpenAI(api_key=key, base_url="https://api.kimi.com/coding/v1")
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=5,
            )
            print(f"✅ {kname} + {m}: {resp.choices[0].message.content!r}")
        except Exception as e:
            msg = str(e)[:120]
            print(f"❌ {kname} + {m}: {msg}")
