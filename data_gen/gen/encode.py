import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from trl.trainer.utils import pad

from on_policy_data_gen.gen.common import process, tokenize_row
# from distill.utils import 


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    n = len(data[0]['responses'])
    data_flat = []
    for row in data:
        assert len(row['responses']) == n
        for res in row['responses']:
            data_flat.append(
                [
                    {
                        "role": "user",
                        "content": row["prompt"]
                    },
                    {
                        "role": "assistant",
                        "content": res
                    }
                ]
            )
    return data_flat, n


def collate_fn(batch, tokenizer, max_length):
    texts = [tokenizer.apply_chat_template(item, tokenize=False) for item in batch]
    inputs = tokenizer(texts, add_special_tokens=False, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Distributed Text Generation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--model_name", type=str, required=True, help="Model name from Hugging Face model hub")
    # parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum length of generated text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to input JSON data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the embeddings file")
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    ).to(device)
    model.eval()

    # Load the input data
    # data, n_res = load_json_data(args.input_data_path)
    # dataset = CustomDataset(data)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: collate_fn(x, tokenizer, args.max_tokens), shuffle=False)
    # load data
    with open(args.input_data_path, encoding="utf-8") as f:
        data_sampling = json.load(f)

    dataset = Dataset.from_list(data_sampling)

    dataset = dataset.map(
        process,
        fn_kwargs={
            "tokenizer": tokenizer,
            # "pairwise": training_args.loss_type == "simpo"
        },
        num_proc=12,
    )
    # tokenize the dataset
    dataset = dataset.map(
        tokenize_row,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        num_proc=12,
    )
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Prepare a list to store embeddings
    all_embeddings = []
    all_probs = []
    all_positions = []

    n_res = len(dataset[0]['responses_input_ids'])
    # Process each batch with progress bar
    for row in tqdm(dataset, desc="Processing Batches"):
        batch = {
            'input_ids': pad([torch.tensor(d) for d in row['responses_input_ids']], padding_value=0, padding_side='left').to(device),
            'attention_mask': pad([torch.tensor(d) for d in row['responses_attention_mask']], padding_value=0, padding_side='left').to(device),
            'labels': pad([torch.tensor(d) for d in row['responses_labels']], padding_value=-100, padding_side='left').to(device),
        }
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
        
        # Get the last token's embedding for each item in the batch
        last_token_embeddings = outputs.hidden_states[-1][:, -1, :].cpu()
        last_token_embeddings = [last_token_embeddings[i] for i in range(last_token_embeddings.shape[0])]
        # Append the embeddings to the list
        all_embeddings.extend(last_token_embeddings)

        max_label_len = (batch['labels'] != -100).sum(-1).max()
        logprobs = outputs.logits[:, -max_label_len:].log_softmax(dim=-1)
        logprobs_val, logprobs_idx = logprobs.topk(100, -1)
        # all_logits.append(outputs.logits.cpu())
        all_probs.append(logprobs_val.cpu())
        all_positions.append(logprobs_idx.cpu())

    # [n_prompt, n_response, n_hidden]
    all_embeddings = torch.stack(all_embeddings)
    all_embeddings = all_embeddings.reshape(len(dataset), n_res, all_embeddings.shape[-1])
    # Save all embeddings to a single file
    to_save = {
        "embeddings": all_embeddings,
        "probs": all_probs,
        "positions": all_positions
        # "logits": all_logits
    }
    torch.save(to_save, args.output_path)
    # print(f"Saved all embeddings to {args.output_path}")

if __name__ == "__main__":
    main()
