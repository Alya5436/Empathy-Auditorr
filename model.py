from groq import Groq

client = Groq(api_key="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")

models = client.models.list()

for m in models.data:
    print(m.id)