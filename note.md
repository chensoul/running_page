```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt --break-system-packages
pnpm install
pnpm develop
```

下载 keep 运动数据：

```bash
python run_page/data_to_csv.py
```
