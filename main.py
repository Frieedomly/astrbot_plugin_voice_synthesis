# main.py - AstrBot语音合成插件
import numpy as np
import wave
from pathlib import Path
from typing import Optional

# 使用实际可用的导入
from astrbot.api import logger

# 备用类定义
class AstrMessageEvent:
    def __init__(self):
        self.message = type('Message', (), {'message_str': ''})()

class Plain:
    def __init__(self, text):
        self.text = text

class Record:
    def __init__(self, file_path):
        self.file_path = file_path

# 简单的注册装饰器
def register(name, author, description, version, other):
    def decorator(cls):
        cls.plugin_info = {
            'name': name,
            'author': author,
            'description': description,
            'version': version
        }
        return cls
    return decorator

@register("voice_synthesis", "截图人", "让截图人说话（纯机械合成音）", "1.0.0", "")
class Main:
    """
    主插件类
    """
    
    def __init__(self, context, config=None):
        self.context = context
        self.config = config
        
        # 音频采样率
        self.sample_rate = 22050
        
        # 五元音谐波配置
        self.vowels = {
            'a': {
                'harmonics': [
                    (237, 0.22),
                    (420, 0.27),
                    (630, 0.24),
                    (840, 0.57),  
                    (1050, 0.24)
                ],
                'duration': 0.15
            },
            'i': {
                'harmonics': [
                    (237, 0.54), 
                    (474, 0.09), 
                    (2607, 0.04),
                    (2844, 0.02), 
                    (3081, 0.04)
                ], 
                'duration': 0.12
            },
            'u': {
                'harmonics': [
                    (237, 0.64), 
                    (474, 0.21), 
                    (948, 0.05), 
                    (1185, 0.05), 
                    (1422, 0.02)
                ], 
                'duration': 0.15
            },
            'e': {
                'harmonics': [
                    (237, 0.42), 
                    (474, 0.37), 
                    (711, 0.26), 
                    (948, 0.08), 
                    (2370, 0.04)
                ], 
                'duration': 0.13
            },
            'o': {
                'harmonics': [
                    (237, 0.22), 
                    (474, 0.28), 
                    (711, 0.2), 
                    (948, 0.48)
                ], 
                'duration': 0.15
            }
        }
        
        # 汉字→元音映射
        self.char_to_vowels = {
            '草': ['a', 'o'],
            '啊': ['a'],
            '哦': ['o'],
            '诶': ['e'],
            '呜': ['u'],
            '依': ['i'],
            '哈': ['a'],
            '嘿': ['e', 'i'],
            '呼': ['u'],
            '呵': ['o'],
            '呀': ['a'],
            '哟': ['o'],
            '哇': ['a'],
            '喂': ['e', 'i'],
        }
        
        # 插件数据目录
        self.data_dir = Path("data/voice_synthesis")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("截图人语音合成插件初始化完成")

    def get_command_map(self):
        """返回命令映射 - 这是AstrBot的标准命令注册方式"""
        return {
            "speak": {
                "func": self.speak_command,
                "description": "让截图人说话（纯机械合成音）",
                "usage": "/speak <文本>"
            }
        }

    def get_llm_tools(self):
        """返回LLM可用的函数工具"""
        return [
            {
                "name": "speak_text",
                "description": "让截图人说出指定的文本内容",
                "function": self.llm_speak_tool,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "要说的文本内容"
                        }
                    },
                    "required": ["text"]
                }
            }
        ]

    def _synthesize_vowel(self, vowel: str, duration: Optional[float] = None) -> np.ndarray:
        """合成单个元音"""
        if vowel not in self.vowels:
            logger.warning(f"未知元音: {vowel}")
            return np.array([])
        
        config = self.vowels[vowel]
        if not config['harmonics']:
            logger.warning(f"元音 '{vowel}' 未配置谐波数据")
            return np.array([])
        
        dur = duration if duration is not None else config['duration']
        num_samples = int(self.sample_rate * dur)
        t = np.linspace(0, dur, num_samples)
        
        signal = np.zeros_like(t)
        for frequency, amplitude in config['harmonics']:
            harmonic_wave = amplitude * np.sin(2 * np.pi * frequency * t)
            signal += harmonic_wave
        
        # 应用包络
        attack_len = min(int(0.02 * self.sample_rate), num_samples // 4)
        release_len = min(int(0.03 * self.sample_rate), num_samples // 4)
        
        envelope = np.ones_like(signal)
        if attack_len > 0:
            envelope[:attack_len] = np.linspace(0, 1, attack_len)
        if release_len > 0:
            envelope[-release_len:] = np.linspace(1, 0, release_len)
        
        signal *= envelope
        
        # 归一化
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            signal = signal / max_amplitude * 0.8
        
        return signal
    
    def _text_to_audio(self, text: str) -> Optional[np.ndarray]:
        """文本转音频波形"""
        audio_segments = []
        
        for char in text:
            phoneme_list = self.char_to_vowels.get(char, ['a'])
            for phoneme in phoneme_list:
                segment = self._synthesize_vowel(phoneme)
                if len(segment) > 0:
                    audio_segments.append(segment)
        
        if not audio_segments:
            return None
        
        return np.concatenate(audio_segments)
    
    def _save_as_wav(self, audio_data: np.ndarray, filename: str) -> Path:
        """保存为WAV文件"""
        file_path = self.data_dir / filename
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(file_path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return file_path

    async def speak_command(self, event):
        """
        speak命令处理器
        使用方式: /speak 你好
        """
        try:
            # 获取消息文本内容
            text_content = event.message.message_str.strip()
            
            # 移除命令部分，只保留内容
            if text_content.startswith('/speak'):
                text_content = text_content[6:].strip()
            elif text_content.startswith('speak'):
                text_content = text_content[5:].strip()
            
            if not text_content:
                return self._create_plain_result("ldm让我说啥（）用法: /speak 要说的内容")
            
            # 调用语音合成
            result = await self._synthesize_speech(text_content)
            return result
            
        except Exception as error:
            logger.error(f"语音合成失败: {error}", exc_info=True)
            return self._create_plain_result(f"草 合成崩了（）错误: {str(error)}")

    async def llm_speak_tool(self, text: str):
        """
        LLM调用的语音合成工具
        """
        try:
            if not text or not text.strip():
                return {"success": False, "message": "文本内容不能为空"}
            
            result = await self._synthesize_speech(text)
            
            # 根据返回结果判断是否成功
            if result and hasattr(result[0], 'type'):
                if result[0].type == 'record':
                    return {"success": True, "message": f"已成功合成语音: {text}"}
                else:
                    return {"success": False, "message": result[0].content}
            else:
                return {"success": False, "message": "语音合成失败"}
                
        except Exception as error:
            logger.error(f"LLM语音合成失败: {error}", exc_info=True)
            return {"success": False, "message": f"语音合成失败: {str(error)}"}

    async def _synthesize_speech(self, text: str):
        """通用的语音合成逻辑"""
        # 1. 文本转音频
        audio_waveform = self._text_to_audio(text)
        if audio_waveform is None:
            return self._create_plain_result("这些字我还不会说（）")
        
        # 2. 保存为WAV文件
        wav_filename = f"voice_{hash(text)}_{id(text)}.wav"
        wav_file_path = self._save_as_wav(audio_waveform, wav_filename)
        
        logger.info(f"生成语音文件: {wav_file_path}")
        
        # 3. 返回语音消息
        return self._create_record_result(str(wav_file_path))

    def _create_plain_result(self, text: str):
        """创建文本结果"""
        result = type('Result', (), {})()
        result.type = 'plain'
        result.content = text
        return [result]

    def _create_record_result(self, file_path: str):
        """创建语音结果"""
        result = type('Result', (), {})()
        result.type = 'record'
        result.content = file_path
        return [result]

    async def initialize(self):
        """插件初始化完成后的回调"""
        logger.info("语音合成插件初始化完成，LLM工具已注册")