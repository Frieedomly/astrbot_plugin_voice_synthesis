import json
from pathlib import Path

class VoiceSynthesisPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        
        # 加载配置
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 默认配置
            self.config = {
                "voice_synthesis": {
                    "llm_provider": "openai",
                    "model": "gpt-4o-mini"
                }
            }