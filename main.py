import os
from huggingface_hub import InferenceClient
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

message = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B:featherless-ai",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer with minimum tokens."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)


embedding_client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

# result = embedding_client.sentence_similarity(
#     "That is a happy person",
#     [
#         "That is a happy dog",
#         "That is a very happy person",
#         "Today is a sunny day"
#     ],
#     model="google/embeddinggemma-300m",
# )


result = embedding_client.feature_extraction("What is the capital of France?")

print(message.choices[0].message.content)
print(result)
