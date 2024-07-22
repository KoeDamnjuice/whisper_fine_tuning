import transformers
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import torchaudio
from dataclasses import dataclass, field
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer, WhisperForConditionalGeneration,
    WhisperProcessor
)
from typing import Any, Dict, List, Optional, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions

    pred.label_ids[pred.label_ids == -100] = whisper_tokenizer.pad_token_id

    pred_str = whisper_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # we do not want to group tokens when computing the metrics
    label_str = whisper_tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
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

whisper_model ="D:\\projects\\wsl\\large_model\\whisper_model\\whisper-tiny"

train_wav_scp = "D:\\projects\\wsl\\large_model\\project\\data\\wav.scp"
train_text = "D:\\projects\\wsl\\large_model\\project\\data\\text"
# 特征提取器
whisper_feature_extractor = WhisperFeatureExtractor()
whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model)
whisper_tokenizer.set_prefix_tokens(language="chinese", task="transcribe")

# data_audio = torchaudio.load("D:\\projects\\wsl\\large_model\\project\\data\\wav\\BAC009S0150W0001.wav")

# 处理数据
train_data_list = IterWhisperDataSet(train_wav_scp,
                                     train_text,
                                     whisper_feature_extractor,
                                     whisper_tokenizer)

# 加载资源
model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
print(model)
# 初始化训练器
processor = WhisperProcessor.from_pretrained(training_args.output_dir)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    forward_attention_mask=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data_list,
    eval_dataset=train_data_list,
    tokenizer=whisper_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)
# 训练
train_result = trainer.train(resume_from_checkpoint=checkpoint)
# 测试