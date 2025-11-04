"""
DeepSeek V3 DPO 训练器实现
Direct Preference Optimization (DPO) 用于模型对齐
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    DeepSeek V3 DPO 训练器
    
    该类实现了 Direct Preference Optimization 算法，
    用于根据人类偏好对大语言模型进行微调。
    """
    
    def __init__(
        self,
        model_name: str,
        beta: float = 0.5,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        """
        初始化 DPO 训练器
        
        Args:
            model_name: 模型名称或路径
            beta: DPO 温度参数（控制对偏好的强制程度）
            learning_rate: 学习率
            device: 设备类型
        """
        self.model_name = model_name
        self.beta = beta
        self.learning_rate = learning_rate
        self.device = device
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload"
        )
        
        # 设置参考模型（用于计算 DPO 损失）
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload"
        )
        self.reference_model.eval()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 DPO 损失函数
        
        DPO 损失 = -log(sigmoid(beta * (log(π(y_w|x)) - log(π(y_l|x)) 
                                        - (log(π_ref(y_w|x)) - log(π_ref(y_l|x))))))
        
        Args:
            policy_chosen_logps: 策略模型对选中回复的 log probability
            policy_rejected_logps: 策略模型对拒绝回复的 log probability
            reference_chosen_logps: 参考模型对选中回复的 log probability
            reference_rejected_logps: 参考模型对拒绝回复的 log probability
            
        Returns:
            损失值和统计信息字典
        """
        # 计算 log 概率差异
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        # DPO 损失
        logits = policy_logratios - reference_logratios
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()
        
        # 计算准确率（有多少比例选择了更优的回复）
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
        
        return dpo_loss, {
            "loss": dpo_loss.item(),
            "accuracy": accuracy.item(),
            "policy_logratios": policy_logratios.mean().item(),
            "reference_logratios": reference_logratios.mean().item(),
        }
    
    def get_batch_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        计算批次中每个序列的 log probability
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            model: 使用的模型（默认为策略模型）
            
        Returns:
            每个序列的平均 log probability
        """
        if model is None:
            model = self.model
        
        with torch.no_grad() if model == self.reference_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
        
        # 计算每个位置的对数概率
        log_probs = F.log_softmax(logits[..., :-1, :], dim=-1)
        
        # 获取实际 token 的 log prob
        selected_log_probs = torch.gather(
            log_probs,
            2,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # 应用掩码和平均
        selected_log_probs = selected_log_probs * attention_mask[:, 1:]
        return selected_log_probs.sum(-1) / attention_mask[:, 1:].sum(-1)
    
    def train_step(self, batch: Dict) -> Dict:
        """
        执行单个训练步骤
        
        Args:
            batch: 包含选中/拒绝样本的批次数据
            
        Returns:
            包含损失和指标的字典
        """
        self.model.train()
        
        # 获取批次数据
        chosen_ids = batch["chosen_ids"].to(self.device)
        chosen_mask = batch["chosen_mask"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)
        rejected_mask = batch["rejected_mask"].to(self.device)
        
        # 计算 log probabilities
        policy_chosen_logps = self.get_batch_logps(
            chosen_ids, chosen_mask, self.model
        )
        policy_rejected_logps = self.get_batch_logps(
            rejected_ids, rejected_mask, self.model
        )
        
        with torch.no_grad():
            reference_chosen_logps = self.get_batch_logps(
                chosen_ids, chosen_mask, self.reference_model
            )
            reference_rejected_logps = self.get_batch_logps(
                rejected_ids, rejected_mask, self.reference_model
            )
        
        # 计算 DPO 损失
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return metrics
    
    def save_model(self, output_dir: str):
        """保存模型和分词器"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到 {output_dir}")