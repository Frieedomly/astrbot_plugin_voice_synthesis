import numpy as np
import wave
from pathlib import Path
from typing import Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

@register("voice_synthesis", "截图人", "让截图人说话（纯机械合成音）", "1.0.0", "")
class VoiceSynthesisPlugin(Star):
    """
    语音合成插件
    """
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config
        
        # 音频采样率
        self.sample_rate = 22050
        
        # 五元音谐波配置
        self.vowels = {
            'a': {
                'harmonics': [
                    (237, 0.22), (420, 0.27), (630, 0.24),
                    (840, 0.57), (1050, 0.24)
                ],
                'duration': 0.15
            },
            'i': {
                'harmonics': [
                    (237, 0.54), (474, 0.09), (2607, 0.04),
                    (2844, 0.02), (3081, 0.04)
                ], 
                'duration': 0.12
            },
            'u': {
                'harmonics': [
                    (237, 0.64), (474, 0.21), (948, 0.05),
                    (1185, 0.05), (1422, 0.02)
                ], 
                'duration': 0.15
            },
            'e': {
                'harmonics': [
                    (237, 0.42), (474, 0.37), (711, 0.26),
                    (948, 0.08), (2370, 0.04)
                ], 
                'duration': 0.13
            },
            'o': {
                'harmonics': [
                    (237, 0.22), (474, 0.28), (711, 0.2),
                    (948, 0.48)
                ], 
                'duration': 0.15
            }
        }
        
        # 汉字→元音映射
        self.char_to_vowels = {
            '草': ['a', 'o'], '啊': ['a'], '哦': ['o'],
            '诶': ['e'], '呜': ['u'], '依': ['i'],
            '哈': ['a'], '嘿': ['e', 'i'], '呼': ['u'],
            '呵': ['o'], '呀': ['a'], '哟': ['o'],
            '哇': ['a'], '喂': ['e', 'i'],
        }
        
        # 插件数据目录
        self.data_dir = Path("data/voice_synthesis")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("截图人语音合成插件初始化完成")

    

    @filter.llm_tool(name="speak_text")
    async def speak_text_tool(self, event: AstrMessageEvent, text: str):
        '''让截图人说出指定的文本内容
    
        Args:
            text(string): 要说的文本内容
        '''
        logger.info(f"speak_text被调用 文本: {text}")
    
        # 根据配置选择provider
        provider_id = self.voice_config.get("llm_provider", "openai")
        provider = self.context.get_provider_by_id(provider_id)
    
        if not provider:
            # 降级到当前provider
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
    
        # 用OpenAI provider调用语音合成
        result = await self._synthesize_speech(text, provider)
    
        if result.get("success"):
            yield event.record_result(result.get("file_path"))
        else:
            yield event.plain_result(f"合成失败（）{result.get('error')}")
            
    @filter.command(command_name="speak")
    async def speak_command(self, event: AstrMessageEvent, text: str):
        '''手动让截图人说话'''
        result = await self._synthesize_speech(text)
        if result.get("success"):
            yield event.record_result(result.get("file_path"))
        else:
            yield event.plain_result(f"语音合成失败（）{result.get('error')}")

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

    async def _synthesize_speech(self, text: str):
        """通用的语音合成逻辑"""
        try:
            if not text or not text.strip():
                return {"success": False, "error": "文本内容不能为空"}
            
            audio_waveform = self._text_to_audio(text)
            if audio_waveform is None:
                return {"success": False, "error": "这些字我还不会说（）"}
            
            wav_filename = f"voice_{hash(text)}_{id(text)}.wav"
            wav_file_path = self._save_as_wav(audio_waveform, wav_filename)
            
            logger.info(f"生成语音文件: {wav_file_path}")
            
            return {"success": True, "file_path": str(wav_file_path)}
            
        except Exception as error:
            logger.error(f"语音合成失败: {error}", exc_info=True)
            return {"success": False, "error": f"草 语音崩了: {str(error)}"}