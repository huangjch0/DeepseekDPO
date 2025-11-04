"""
DeepSeek V3 DPO 训练主脚本
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dpo_trainer import DPOTrainer
from data_processor import create_data_loader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(config: dict):
    """
    主训练循环
    
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化训练器
    logger.info(f"初始化模型: {config['model_name']}")
    trainer = DPOTrainer(
        model_name=config["model_name"],
        beta=config.get("beta", 0.5),
        learning_rate=config.get("learning_rate", 1e-4),
        device=config.get("device", "cuda"),
        lora_config=config.get("lora", None)  # 传递 LoRA 配置
    )
    
    # 创建数据加载器
    logger.info("加载训练数据...")
    train_loader = create_data_loader(
        data_path=config["train_data_path"],
        tokenizer=trainer.tokenizer,
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        max_length=config.get("max_length", 2048),
    )
    
    # 创建 TensorBoard 写入器
    writer = SummaryWriter(output_dir / "logs")
    
    # 训练参数
    num_epochs = config.get("num_epochs", 3)
    eval_steps = config.get("eval_steps", 100)
    save_steps = config.get("save_steps", 500)
    
    global_step = 0
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        epoch_accuracy = 0
        
        progress_bar = tqdm(train_loader, desc="训练进度")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 执行训练步骤
            metrics = trainer.train_step(batch)
            
            # 累积指标
            epoch_loss += metrics["loss"]
            epoch_accuracy += metrics["accuracy"]
            
            # 记录到 TensorBoard
            writer.add_scalar("train/loss", metrics["loss"], global_step)
            writer.add_scalar("train/accuracy", metrics["accuracy"], global_step)
            writer.add_scalar(
                "train/policy_logratios",
                metrics["policy_logratios"],
                global_step
            )
            
            progress_bar.set_postfix({
                "loss": metrics["loss"],
                "acc": metrics["accuracy"],
            })
            
            global_step += 1
            
            # 定期保存检查点(保存 LoRA 适配器)
            if global_step % save_steps == 0:
                save_path = output_dir / f"checkpoint-{global_step}"
                trainer.save_model(str(save_path), merge_lora=False)
                logger.info(f"检查点已保存到 {save_path}")
        
        # 输出 epoch 统计
        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        
        logger.info(
            f"Epoch {epoch + 1} 完成 - "
            f"平均损失: {avg_loss:.4f}, "
            f"平均准确率: {avg_accuracy:.4f}"
        )
    
    # 保存最终模型 - 合并 LoRA 权重
    final_save_path = output_dir / "final_model"
    merge_lora = config.get("lora") is not None  # 如果使用了 LoRA 就合并
    trainer.save_model(str(final_save_path), merge_lora=merge_lora)
    
    if merge_lora:
        logger.info(f"LoRA 权重已合并并保存到 {final_save_path}")
    else:
        logger.info(f"最终模型已保存到 {final_save_path}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 DPO 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载配置并开始训练
    config = load_config(args.config)
    train(config)