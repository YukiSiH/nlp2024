import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from utils_train import set_seed, load_and_cache_examples

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def evaluate(args, model, tokenizer):
    eval_dataset = load_and_cache_examples(args, args.dataset_name, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    preds, out_label_ids = None, None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[3])
            logits = outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    return {
        "f1_micro": f1_score(out_label_ids, preds, average="micro"),
        "f1_macro": f1_score(out_label_ids, preds, average="macro"),
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="agnews", type=str, help="数据集名称")
    parser.add_argument("--data_dir", default="agnews", type=str, help="数据集路径")
    parser.add_argument("--output_dir", default="output", type=str, help="模型路径")
    parser.add_argument("--max_seq_length", default=128, type=int, help="最大序列长度")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="评估批量大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, config=config)
    model.to(args.device)

    results = evaluate(args, model, tokenizer)
    logger.info(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()
