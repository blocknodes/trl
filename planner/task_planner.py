# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward



dataset = load_dataset("../../data/DeepMath-103K/",split="train")



trainer = GRPOTrainer(
    model="../../models/Qwen3-0.6B/",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()