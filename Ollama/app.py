from ollama import chat

response = chat(
    model="gemma3",
    messages=[
        {
            "role": "user",
            "content": "Explain artificial intelligence in simple terms."
        }
    ]
)

print(response.message.content)