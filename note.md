```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt --break-system-packages
pnpm install
pnpm develop
```

下载 keep 跑步数据：

```bash
python run_page/run_to_csv.py
```
