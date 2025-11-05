# main.py - AstrBot语音合成插件
import numpy as np
import wave
from pathlib import Path
from typing import Optional

# 基础导入
from astrbot.api import logger
from astrbot.api.star import Star
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain, Record


class Main(Star):
    """
    主插件类
    """
    
    def __init__(self, context):
        super().__init__(context)
        
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
        }
        
        # 插件数据目录
        self.data_dir = Path("data/voice_synthesis")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("截图人语音合成插件初始化完成")
    
    def _synthesize_vowel(
        self, 
        vowel: str, 
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        合成单个元音
        """
        if vowel not in self.vowels:
            logger.warning(f"未知元音: {vowel}")
            return np.array([])
        
        config = self.vowels[vowel]
        if not config['harmonics']:
            logger.warning(f"元音 '{vowel}' 未配置谐波数据")
            return np.array([])
        
        # 确定持续时间
        dur = duration if duration is not None else config['duration']
        
        # 生成时间轴
        num_samples = int(self.sample_rate * dur)
        t = np.linspace(0, dur, num_samples)
        
        # 叠加所有谐波分量
        signal = np.zeros_like(t)
        for frequency, amplitude in config['harmonics']:
            harmonic_wave = amplitude * np.sin(2 * np.pi * frequency * t)
            signal += harmonic_wave
        
        # 应用包络（淡入淡出）
        attack_len = min(int(0.02 * self.sample_rate), num_samples // 4)
        release_len = min(int(0.03 * self.sample_rate), num_samples // 4)
        
        envelope = np.ones_like(signal)
        if attack_len > 0:
            envelope[:attack_len] = np.linspace(0, 1, attack_len)
        if release_len > 0:
            envelope[-release_len:] = np.linspace(1, 0, release_len)
        
        signal *= envelope
        
        # 归一化到[-0.8, 0.8]
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            signal = signal / max_amplitude * 0.8
        
        return signal
    
    def _text_to_audio(self, text: str) -> Optional[np.ndarray]:
        """
        文本转音频波形
        """
        audio_segments = []
        
        for char in text:
            # 查找字符对应的元音序列
            phoneme_list = self.char_to_vowels.get(char, ['a'])
            
            for phoneme in phoneme_list:
                segment = self._synthesize_vowel(phoneme)
                if len(segment) > 0:
                    audio_segments.append(segment)
        
        if not audio_segments:
            return None
        
        # 拼接所有片段
        return np.concatenate(audio_segments)
    
    def _save_as_wav(self, audio_data: np.ndarray, filename: str) -> Path:
        """
        保存为WAV文件
        """
        file_path = self.data_dir / filename
        
        # 转换为16bit PCM整数格式
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 写入WAV文件
        with wave.open(str(file_path), 'w') as wav_file:
            wav_file.setnchannels(1)       # 单声道
            wav_file.setsampwidth(2)       # 16bit = 2字节
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return file_path
    
    # v4版本使用 @Star.register 装饰器
    @Star.register(
        name="speak",
        description="让截图人说话（纯机械合成音）",
        usage="/speak 要说的内容"
    )
    async def speak_command(self, event: AstrMessageEvent):
        """
        /speak 指令处理器
        
        v4版本注意：参数只有 self 和 event
        """
        # 获取消息文本内容
        text_content = event.message.message_str.strip()
        
        if not text_content:
            yield event.plain_result("让我说啥（）")
            return
        
        try:
            # 1. 文本转音频
            audio_waveform = self._text_to_audio(text_content)
            if audio_waveform is None:
                yield event.plain_result("这些字我还不会说（）")
                return
            
            # 2. 保存为WAV文件
            wav_filename = f"voice_{hash(text_content)}.wav"
            wav_file_path = self._save_as_wav(audio_waveform, wav_filename)
            
            logger.info(f"生成语音文件: {wav_file_path}")
            
            # 3. 发送语音消息
            yield event.record_result(str(wav_file_path))
            
        except Exception as error:
            logger.error(f"语音合成失败: {error}", exc_info=True)
            yield event.plain_result(f"草 合成崩了（）错误: {str(error)}")