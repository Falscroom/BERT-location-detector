from transformers import AutoModelForSequenceClassification

MODEL_DIR = "out/move-det/checkpoint-528"  # e.g. ../../out/move-det
clf = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

print("num_labels:", clf.config.num_labels)
print("id2label:", clf.config.id2label)     # e.g. {0: 'no_move', 1: 'move'}
print("label2id:", clf.config.label2id)     # e.g. {'no_move': 0, 'move': 1}
