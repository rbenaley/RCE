from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# TEST CHAT
resp = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Donne seulement {\"ok\":1}"}],
)
print(resp.choices[0].message.content)

# TEST EMBEDDING
resp = client.embeddings.create(
    model="bge-m3",
    input="test",
)
print(resp)
