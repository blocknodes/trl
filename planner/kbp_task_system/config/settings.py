# 配置项
LLM_CONFIGS = {
    "deepseek-v3": {
        "url": "https://aix-backup.hismarttv.com/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer {key}"},
        "key": "x31ctKZ0ONfi1jkO",
        "model": "deepseek-v3",
        "url_params": {}
    },
    "gpt-4": {
        "url": "https://inner-apisix.hisense.com/openai/deployments/gpt-4-1/chat/completions",
        "headers": {"Content-Type": "application/json", "api-key": "Oi4rzFyLbMOmqVn8YYEyT2Pt0mkr3lgU"},
        "key": "nregzh6g2oviajyjstgzlhjsjmp9rtql",
        "model": "gpt-4-1",
        "url_params": {"user_key": "{key}"}
    },
    "qwen3-4b": {
        "url": "http://localhost:8088/v1/chat/completions",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer {key}"},
        "key": "x31ctKZ0ONfi1jkO",
        "model": "qwen4b",
        "url_params": {}
    }
}

# KBP 检索配置
KBP_CONFIG = {
    "base_url": "https://inner-apisix-test.hisense.com",
    "user_key": "qimfvt7lwtqeyangfl259vjg8fzdhh5l",
    "api_key": "83dd8d9d-6a77-4954-9071-aa195fb6b406"
}

# 服务配置
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 9000,
    "reload": True,
    "log_level": "info"
}

# 默认配置
DEFAULT_MAX_ROUNDS = 2
DEFAULT_LLM_MODEL = "qwen3-4b"
DEFAULT_MODE = "infer"