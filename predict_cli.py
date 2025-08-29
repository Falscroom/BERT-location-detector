from predict import predict, _load_session
import sys

MODEL = "out/qa-distil"
_load_session(MODEL)  # прогреваем

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    print(predict(MODEL, line))
