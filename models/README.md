# Models

Trained checkpoints (e.g., the fine-tuned DistilBERT classifier, the LSTM baseline) are saved here and are
**git-ignored**. Pretrained weights (Whisper, DistilBART, DistilBERT) are downloaded on first use into the
Hugging Face / Whisper caches, not into Git. On Colab, set the cache to Google Drive to survive disconnects
(see the Colab setup notebook).
