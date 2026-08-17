# BERT-base Fine-Tuning Benchmarks: Full Fine-Tuning vs. LoRA

## 🚀 Project Overview

This notebook demonstrates and compares two popular methods for fine-tuning pre-trained BERT-base models for text classification: **Full Fine-Tuning (FFT)** and **Low-Rank Adaptation (LoRA)**. The goal is to classify movie reviews as positive or negative using the IMDB dataset.

We will evaluate both approaches based on:

1.  **Test Accuracy:** How well each model performs on unseen data.
2.  **Training Time:** The duration required to train each model.
3.  **Checkpoint Size:** The disk space occupied by the saved model weights.

This comparison aims to highlight the trade-offs between performance, computational cost, and storage efficiency when using parameter-efficient fine-tuning (PEFT) methods like LoRA.

## 🛠️ Setup and Installation

To run this notebook, you need to install the following libraries. The initial cells in this notebook handle these installations:

```bash
!pip install -q transformers datasets peft accelerate evaluate
!pip install decord
!pip install -U datasets transformers torchvision
!pip uninstall -y torchao # (If torchao was previously installed, as it can cause conflicts)
```

**Note on `torchvision` and `decord`:** Due to potential conflicts or runtime issues with `torchvision.io.VideoReader` in some environments (especially after `torchvision` updates), a mock implementation is provided in Cell `znM-GweTo-lE` to ensure smooth execution, as `torchvision` is re-installed with a newer version in this environment.

## 📚 Data

The project uses the `stanfordnlp/imdb` dataset, a large movie review dataset for binary sentiment classification. It contains 50,000 highly polar movie reviews, split into 25,000 for training and 25,000 for testing.

### Data Preparation:

1.  **Loading:** The dataset is loaded using `datasets.load_dataset('stanfordnlp/imdb')`.
2.  **Tokenization:** Reviews are tokenized using `AutoTokenizer.from_pretrained('bert-base-uncased')`. A `tokenize_function` is applied to pad and truncate sequences to a maximum length of 256 tokens.
3.  **Formatting:** The tokenized dataset is formatted for PyTorch, removing the original 'text' column and renaming 'label' to 'labels' as expected by Hugging Face `Trainer`.

## ⚙️ Methodology

### A. Full Fine-Tuning (FFT)

In Full Fine-Tuning, all parameters of the pre-trained `bert-base-uncased` model are updated during training. This involves a large number of trainable parameters (approximately 110 million for BERT-base).

-   **Model:** `AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`.
-   **Training Arguments:** Standard `TrainingArguments` with a learning rate of `2e-5`, batch size of 32, and 2 epochs.
-   **Metrics:** Accuracy is computed using `evaluate.load("accuracy")`.

### B. Low-Rank Adaptation (LoRA)

LoRA is a parameter-efficient fine-tuning technique that injects small, trainable rank decomposition matrices into the transformer architecture. This drastically reduces the number of trainable parameters while maintaining competitive performance.

-   **Model:** The same `bert-base-uncased` model is used, but a `LoraConfig` is applied to it using `get_peft_model` from the `peft` library.
-   **LoRA Configuration:**
    -   `task_type=TaskType.SEQ_CLS` (Sequence Classification)
    -   `r=4`: The rank of the update matrices, controlling the number of parameters.
    -   `lora_alpha=8`: LoRA scaling factor. The actual scaling is `lora_alpha / r` (8/4 = 2 in this case).
    -   `lora_dropout=0.1`
    -   `target_modules=["query", "value"]`: LoRA adapters are applied to the query and value projections in the self-attention layers.
-   **Trainable Parameters:** With `r=4`, LoRA only trains approximately 147K parameters, which is about 0.13% of the full model's parameters.
-   **Training Arguments:** A slightly higher learning rate (`1e-3`) is used, as is common for PEFT methods, with similar batch size and epochs as FFT.

## 📊 Results

The following table summarizes the key performance indicators for both fine-tuning approaches:

| Model                  | Trainable Parameters | Training Time (s) | Checkpoint Size (MB) | Accuracy (%) |
| :--------------------- | :------------------- | :---------------- | :------------------- | :----------- |
| Full Fine-Tuning (FFT) | ~110M (100%)         | 884.49            | 417.67               | 91.90        |
| LoRA (r=4, a=8)        | ~147K (0.13%)        | 781.90            | 0.58                 | 91.66        |

These results are also visualized in a professional comparison chart (Cell `UMx9ybs03isH`) that graphically compares the accuracy, training duration, and checkpoint size.

## 📈 Conclusion

From the results, we can draw significant conclusions:

-   **Accuracy:** LoRA achieves an accuracy of 91.66%, which is very close to the Full Fine-Tuning accuracy of 91.90%. This demonstrates that LoRA can maintain strong performance despite significantly reducing trainable parameters.
-   **Training Time:** LoRA training was faster (781.90s) than Full Fine-Tuning (884.49s). This is expected as fewer parameters need to be updated during backpropagation, leading to faster iterations.
-   **Checkpoint Size:** The most dramatic difference is in checkpoint size. The LoRA adapter is only 0.58 MB, whereas the fully fine-tuned model is 417.67 MB. This nearly 700x reduction in size makes LoRA adapters incredibly efficient for deployment, storage, and sharing.

In summary, LoRA offers an excellent balance between performance, training efficiency, and model compactness, making it an ideal choice for fine-tuning large language models in resource-constrained environments or when rapid experimentation is desired.
