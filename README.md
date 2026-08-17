# bert_imdb_finetuning_vs_lora
This project benchmarks BERT-base on the IMDB dataset, comparing Full Fine-Tuning against Low-Rank Adaptation (LoRA, $r=4, \alpha=8$). It evaluates test accuracy, training speed, and storage efficiency, showing that LoRA cuts disk usage from ~418 MB to 0.58 MB while matching full fine-tuning performance (~91.7% vs 91.9% accuracy).
