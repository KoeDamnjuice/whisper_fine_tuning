import transformers
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import torchaudio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer
)
# 准备数据
class IterWhisperDataSet(IterableDataset):
    def __init__(self, wav_scp, text, feature_extractor, tokenizer):
        # 设定字典
        self.data_list = {}
        self.whisper_feature_extractor = feature_extractor
        self.whisper_tokenizer = tokenizer
        self.whisper_tokenizer.set_prefix_tokens(language="chinese", task="transcribe")
        with open(wav_scp, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                idx = line.split(" ")[0]
                # 音频路径
                wav_path = " ".join(line.split(" ")[1: ])
                self.data_list[idx] = []
                self.data_list[idx].append(wav_path)
        with open(text, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                idx = line.split(" ")[0]
                # 音频路径
                text = " ".join(line.split(" ")[1: ])
                self.data_list[idx] = []
                self.data_list[idx].append(text)
        print(self.data_list)
        print("文本全部个数为：", len(self.data_list))
    def __len__(self):
        return len(self.data_list)

    # 传入模型遍历
    def __iter__(self):
        # 遍历所有数据
        for idx in self.data_list:
            # 音频的路径
            wav_path = self.data_list[idx][0]
            # 音频的文本
            text = self.data_list[idx][1]

            example = {}

            # 音频提取特征
            # 读取音频
            audio_data = torchaudio.load(wav_path)
            example['input_features'] = self.whisper_feature_extractor(audio_data[0].numpy(), sample_rate=16000).input_features[0]
            # text tokenizer
            example['labels'] = self.whisper_tokenizer(text).input_ids

            yield example

whisper_model_path ="D:\\wsl\\large_model\\whisper_model\\whisper-tiny"

train_wav_scp = "D:\\wsl\\large_model\\project\\data\\wav.scp"
train_text = "D:\\wsl\\large_model\\project\\data\\text"
# 特征提取器
whisper_feature_extractor = WhisperFeatureExtractor()

whisper_tokenizer = WhisperTokenizer().from_pretrained(whisper_model_path)
train_data_list = IterWhisperDataSet(train_wav_scp, train_text, whisper_feature_extractor)

train_data_list = IterWhisperDataSet(
    train_wav_scp,
    train_text,
    whisper_feature_extractor,
    whisper_tokenizer
)
# 加载资源

# 初始化训练器

# 训练

# 测试