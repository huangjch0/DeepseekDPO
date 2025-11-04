"""
DPO 训练数据加载和处理
"""

import json
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch


class DPODataset(Dataset):
    """
    DPO 训练数据集
    
    数据格式：
    {
        "prompt": "用户提示",
        "chosen": "更优的回复",
        "rejected": "较差的回复"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径（JSON 格式）
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载了 {len(self.data)} 个样本")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # 构建完整的文本序列
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        # 分词
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_ids": chosen_encodings["input_ids"].squeeze(0),
            "chosen_mask": chosen_encodings["attention_mask"].squeeze(0),
            "rejected_ids": rejected_encodings["input_ids"].squeeze(0),
            "rejected_mask": rejected_encodings["attention_mask"].squeeze(0),
        }


def create_data_loader(
    data_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    max_length: int = 2048,
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        max_length: 最大序列长度
        
    Returns:
        DataLoader 对象
    """
    dataset = DPODataset(data_path, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    
    return dataloader