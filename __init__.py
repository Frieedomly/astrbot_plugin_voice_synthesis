from astrbot.core.star import Star
from astrbot.core.star.context import Context
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api import logger
import json
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

class VoiceSynthesisPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        
        # 加载配置
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {"voice_synthesis": {}}
        
        self.voice_config = self.config.get("voice_synthesis", {})
        logger.info("语音合成插件已加载")