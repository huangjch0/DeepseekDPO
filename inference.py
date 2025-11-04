import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Optional
import json


class DeepSeekInference:
    """推理类"""
    
    def __init__(self, model_path: str, device: str = "cuda", base_model: Optional[str] = None):
        """
        初始化推理器
        
        Args:
            model_path: 模型或 LoRA 适配器路径（包含 adapter_config.json）
            device: 设备类型
            base_model: 基座模型名称/路径（当适配器目录下没有 tokenizer 时可指定）
        """
        self.device = device
        path = Path(model_path)
        is_adapter = (path / "adapter_config.json").exists()

        use_cuda = (device == "cuda") and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else None

        if is_adapter:
            # 选择 tokenizer 来源：优先适配器目录，其次 base_model 参数，再次从适配器配置推断
            if (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists():
                tokenizer_src = model_path
            elif base_model:
                tokenizer_src = base_model
            else:
                try:
                    with open(path / "adapter_config.json", "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    tokenizer_src = cfg.get("base_model_name_or_path")
                except Exception:
                    tokenizer_src = None
                if not tokenizer_src:
                    raise ValueError("无法确定 tokenizer 来源，请传入 base_model 参数。")

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)

            # 加载 LoRA 适配器（自动恢复基座模型）
            from peft import AutoPeftModelForCausalLM
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            # 加载完整模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )

        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus sampling 参数
            num_beams: beam search 束数
            
        Returns:
            生成的文本
        """
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        return generated_text


if __name__ == "__main__":
    # 使用示例：当 final_model 里只有 LoRA 适配器时也可直接传该目录
    inference = DeepSeekInference("./output/dpo_model/final_model")
    
    prompt = "请介绍一下人工智能的应用:"
    result = inference.generate(prompt)
    
    print(f"提示: {prompt}")
    print(f"生成结果:\n{result}")