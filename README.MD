# Movement Detection + Location QA (Two-Heads)

This repository contains an **open-source experiment** for detecting user **movement intent** in text and extracting the **destination location**.  
It combines two heads:

- **Binary classifier** → detects if the text expresses a *movement directive* or not.  
- **QA head** → extracts the actual *location span* when movement is present.

Together, they form a robust pipeline for roleplay / chat environments where user messages may or may not imply changing the scene.
