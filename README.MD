# Movement Detection + Location QA (Two-Heads)

This repository contains an **open-source experiment** for detecting user **movement intent** in text and extracting the **destination location**.  
It combines two heads:

- **Binary classifier** → detects if the text expresses a *movement directive* or not.  
- **QA head** → extracts the actual *location span* when movement is present.

Together, they form a robust pipeline for roleplay / chat environments where user messages may or may not imply changing the scene.

Latency:
Runs: 160
TOTAL  — mean 23.9 ms | p95 23.9 ms | p99 38.5 ms


Synthetically achieved rules + dataset:
=== EVAL SUMMARY ===
Total: 379
Overall accuracy: 1.000
Move EM accuracy: 1.000  (n=154)
No-move correctness (pred=None): 1.000  (n=225)
Movement detection — precision: 1.000, recall: 1.000