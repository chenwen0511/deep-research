import openai
import asyncio
from config import get_config

config = get_config()
openai_config = config.get("openai", {})
api_key = openai_config.get("api_key", "")
model = openai_config.get("model", "demand_charge_agent_4o")
base_url = openai_config.get("base_url", None)

# Azure OpenAI 需要特殊的配置
client_args = {
    "api_key": api_key,
    "azure_endpoint": base_url,
    "api_version": "2024-08-01-preview",
    "azure_deployment": model,
}

client = openai.AzureOpenAI(**client_args)
async_client = openai.AsyncAzureOpenAI(**client_args)

messages = [
    {"role": "system", "content": "You are a helpful assistant. Please provide your response in JSON format."},
    {"role": "user", "content": "Please provide information about the moon's capital in JSON format."}
]

# Prepare the request parameters
request_params = {
    "model": model,
    "messages": messages,
    "temperature": 0.7,
    "frequency_penalty": 0.3,
    "response_format": {"type": "json_object"},
    "stream": True
}

print(request_params)

async def main(query):
    request_params["messages"][1]["content"] = query
    response = await async_client.chat.completions.create(**request_params)
    async for chunk in response:
        # 安全地检查和访问数据
        if (hasattr(chunk, 'choices') and 
            len(chunk.choices) > 0 and 
            hasattr(chunk.choices[0], 'delta') and 
            hasattr(chunk.choices[0].delta, 'content') and 
            chunk.choices[0].delta.content is not None):
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

if __name__ == "__main__":
    query = "三国演义是谁写的？"
    asyncio.run(main(query))