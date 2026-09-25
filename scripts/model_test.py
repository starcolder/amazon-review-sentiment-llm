import ollama
models = ['myqwen', 'myphi']
response = ollama.chat(
    model= models[1],
    messages=[
        {
            'role': 'user',
            'content': 'what is 2+2=?'
        }
    ]
)

print(response['message']['content'])