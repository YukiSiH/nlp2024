import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW,
                          get_linear_schedule_with_warmup)
from utils_train import set_seed, get_labels, load_and_cache_examples

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per GPU = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    global_step = 0
    tr_loss = 0.0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

    return global_step, tr_loss / global_step

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="agnews", type=str, help="数据集名称")
    parser.add_argument("--data_dir", default="agnews", type=str, help="数据集路径")
    parser.add_argument("--output_dir", default="output", type=str, help="模型保存路径")
    parser.add_argument("--max_seq_length", default=128, type=int, help="最大序列长度")
    parser.add_argument("--train_batch_size", default=16, type=int, help="训练批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--num_train_epochs", default=3, type=float, help="训练周期")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam优化器的epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大梯度裁剪")
    parser.add_argument("--warmup_steps", default=0, type=int, help="预热步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=len(get_labels(args.data_dir)))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, args.dataset_name, tokenizer, evaluate=False)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(f"Training completed. global_step = {global_step}, average loss = {tr_loss}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
