import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "google/gemma-3n-e4b-it",
  "messages": [{"role":"user","content":""}],
  "max_tokens": 512,
  "temperature": 0.20,
  "top_p": 0.70,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
