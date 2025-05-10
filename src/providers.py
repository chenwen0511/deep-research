# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import Dict, Any, Optional
import openai
from .config import get_config


def get_model(is_report: bool = False) -> Dict[str, Any]:
    """
    Get model configuration including client and model name.

    :param is_report: Whether to get the model configuration for a report
    
    Returns:
        Dict containing model configuration
    """
    config = get_config()
    if is_report:
        report_config = config.get("report_llm", {})
        api_key = report_config.get("api_key", "")
        model = report_config.get("model", "gpt-4o")
        api_version = report_config.get("api_version", "2024-08-01-preview")
        base_url = report_config.get("base_url", None)
    else:
        openai_config = config.get("openai", {})
        api_key = openai_config.get("api_key", "")
        model = openai_config.get("model", "gpt-4o-mini")
        base_url = openai_config.get("base_url", None)
        api_version = openai_config.get("api_version", "2024-08-01-preview")

    # Azure OpenAI 需要特殊的配置
    client_args = {
        "api_key": api_key,
        "azure_endpoint": base_url,
        "api_version": api_version,
        "azure_deployment": model,
    }


    client = openai.AzureOpenAI(**client_args)
    async_client = openai.AsyncAzureOpenAI(**client_args)

    return {
        "client": client,
        "async_client": async_client,
        "model": model
    }


def get_search_provider(search_source=None):
    """
    Get the appropriate search provider based on configuration.
    
    Returns:
        An instance of the search provider class
    """
    if search_source is None:
        config = get_config()
        search_source = config.get("research", {}).get("search_source", "serper")

    if search_source == "mp_search":
        from .mp_search_client import MPSearchClient
        return MPSearchClient()
    elif search_source == "tavily":
        from .tavily_client import TavilyClient
        return TavilyClient()
    else:  # Default to serper
        from .serper_client import SerperClient
        return SerperClient()
