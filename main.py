# main.py - AstrBot语音合成插件
import numpy as np
import wave
from pathlib import Path
from typing import Optional, List, Tuple

from astrbot.api.star import Star, register
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain, Record
from astrbot.api import logger, CommandResult


class Main(Star):
    """
    主插件类
    按照AstrBot插件规范，类名必须为Main
    """
    
    def __init__(self, context):
        """
        插件初始化
        
        Args:
            context: AstrBot提供的上下文对象
        """
        super().__init__(context)
        
        # 音频参数
        self.sample_rate = 22050  # 采样率 22.05kHz
        
        # 五元音的谐波配置
        # 每个元音由多个正弦波（谐波）叠加而成
        # 格式: [(频率Hz, 振幅0-1), ...]
        self.vowels = {
            'a': {
                'harmonics': [
                    (237, 0.22),   # 第1谐波
                    (420, 0.27),   # 第2谐波
                    (630, 0.24),   # 第3谐波
                    (840, 0.57),   # 第4谐波
                    (1050, 0.24)   # 第5谐波
                ],
                'duration': 0.15  # 默认持续时间（秒）
            },
            'i': {
                'harmonics': [ (237, 0.54), (474, 0.09), (2607, 0.04),
                    (2844, 0.02), (3081, 0.04)],  # TODO: ldm补充
                'duration': 0.12
            },
            'u': {
                'harmonics': [(237, 0.64), (474, 0.21), (948, 0.05), 
                    (1185, 0.05), (1422, 0.02)],  # TODO: ldm补充
                'duration': 0.15
            },
            'e': {
                'harmonics': [(237, 0.42), (474, 0.37), (711, 0.26), 
                    (948, 0.08), (2370, 0.04)],  # TODO: ldm补充
                'duration': 0.13
            },
            'o': {
                'harmonics': [ (237, 0.22), (474, 0.28), (711, 0.2), (948, 0.48)],  # TODO: ldm补充
                'duration': 0.15
            }
        }
        
        # 简单的汉字到元音映射
        # 生产环境需要完整的拼音库
        self.char_to_vowels = {
            '草': ['a', 'o'],
            '啊': ['a'],
            '哦': ['o'],
            '诶': ['e'],
            '呜': ['u'],
            '依': ['i'],
        }
        
        # 数据存储目录
        self.data_dir = Path("data/voice_synthesis")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("截图人语音合成插件已加载")
    
    def _synthesize_vowel(
        self, 
        vowel: str, 
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        合成单个元音
        
        Args:
            vowel: 元音字母 (a/i/u/e/o)
            duration: 持续时间（秒），不传则使用默认值
            
        Returns:
            音频波形数据（归一化后的numpy数组）
        """
        # 检查元音是否存在且已配置
        if vowel not in self.vowels:
            logger.warning(f"未知元音: {vowel}")
            return np.array([])
        
        config = self.vowels[vowel]
        if not config['harmonics']:
            logger.warning(f"元音 {vowel} 未配置谐波数据")
            return np.array([])
        
        # 使用自定义或默认持续时间
        dur = duration if duration else config['duration']
        
        # 生成时间轴
        num_samples = int(self.sample_rate * dur)
        t = np.linspace(0, dur, num_samples)
        
        # 叠加所有谐波分量
        signal = np.zeros_like(t)
        for freq, amplitude in config['harmonics']:
            # 生成正弦波: A * sin(2π * f * t)
            harmonic = amplitude * np.sin(2 * np.pi * freq * t)
            signal += harmonic
        
        # 添加ADSR包络（简化版：只做淡入淡出）
        # Attack: 20ms 淡入，避免爆音
        attack_samples = min(int(0.02 * self.sample_rate), num_samples // 4)
        # Release: 30ms 淡出
        release_samples = min(int(0.03 * self.sample_rate), num_samples // 4)
        
        envelope = np.ones_like(signal)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        signal *= envelope
        
        # 归一化到 [-0.8, 0.8] 范围（留20%余量防止削波）
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.8
        
        return signal
    
    def _text_to_audio(self, text: str) -> Optional[np.ndarray]:
        """
        将文本转换为音频波形
        
        Args:
            text: 输入文本
            
        Returns:
            完整的音频波形，失败返回None
        """
        segments = []
        
        for char in text:
            # 查找字符对应的元音序列
            phonemes = self.char_to_vowels.get(char, ['a'])  # 默认发 'a' 音
            
            for phoneme in phonemes:
                segment = self._synthesize_vowel(phoneme)
                if len(segment) > 0:
                    segments.append(segment)
        
        if not segments:
            return None
        
        # 拼接所有音频片段
        return np.concatenate(segments)
    
    def _save_as_wav(self, audio: np.ndarray, filename: str) -> Path:
        """
        将音频数据保存为WAV文件
        
        Args:
            audio: 音频波形数据（浮点数数组）
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        filepath = self.data_dir / filename
        
        # 转换为16bit PCM格式（QQ语音标准格式）
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(str(filepath), 'w') as wav_file:
            wav_file.setnchannels(1)          # 单声道
            wav_file.setsampwidth(2)          # 16bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return filepath
    
    @register(name="speak", description="让截图人用机械音说话")
    async def speak_command(
        self, 
        event: AstrMessageEvent, 
        message
    ) -> CommandResult:
        """
        /speak 指令处理函数
        
        用法: /speak 要说的文字
        
        Args:
            event: 消息事件对象
            message: 消息对象
            
        Returns:
            CommandResult: 指令执行结果
        """
        # 获取指令后的文本内容
        text = message.message_str.strip()
        
        if not text:
            return CommandResult().message("ldm让我说啥（）")
        
        try:
            # 1. 文本转音频
            audio_data = self._text_to_audio(text)
            if audio_data is None:
                return CommandResult().message("这些字我还不会说（）")
            
            # 2. 保存为WAV文件
            filename = f"voice_{hash(text)}.wav"
            wav_path = self._save_as_wav(audio_data, filename)
            logger.info(f"生成语音文件: {wav_path}")
            
            # 3. 构造语音消息
            # 使用Record组件发送语音
            voice_message = MessageChain([
                Record(file=str(wav_path))
            ])
            
            # 4. 发送消息
            await self.context.send_message(
                event.unified_msg_origin,  # 发送到原消息来源
                voice_message
            )
            
            # 5. 返回成功结果（hit=True表示指令已处理）
            return CommandResult(hit=True)
            
        except Exception as e:
            # 错误处理
            logger.error(f"语音合成失败: {e}", exc_info=True)
            return CommandResult().message(f"草 合成失败了（）错误: {str(e)}")