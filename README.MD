# Movement Detection + Location QA (Two-Heads)

This repository contains an **open-source experiment** for detecting user **movement intent** in text and extracting the **destination location**.  
It combines two heads:

- **Binary classifier** → detects if the text expresses a *movement directive* or not.  
- **QA head** → extracts the actual *location span* when movement is present.

Together, they form a robust pipeline for roleplay / chat environments where user messages may or may not imply changing the scene.

Latency:
Runs: 160
TOTAL  — mean 23.9 ms | p95 23.9 ms | p99 38.5 ms


Synthetically achieved:
=== EVAL SUMMARY ===
Total: 380
Overall accuracy: 0.987
Move EM accuracy: 0.987  (n=157)
No-move correctness (pred=None): 0.987  (n=223)
Movement detection — precision: 0.981, recall: 0.987
[ERR] 02 exp=None | pred=found (conf=0.728)  :: hey, miss me?
[ERR] 25 exp=found | pred=None (conf=0.557)  :: hold my hand in the market square
[ERR] 72 exp=None | pred=found (conf=0.460)  :: lay with me on the bed
[ERR] 300 exp=None | pred=found (conf=0.500)  :: stay with me, don’t leave
[ERR] 337 exp=found | pred=None (conf=0.599)  :: watch the stars from the roofto

Almost whole project done by ChatGPT, I've changed a lot to fix binary model but half functions doesnt work now and I am lazy to fix it
