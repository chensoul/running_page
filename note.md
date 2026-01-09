1、下载

```bash
git clone git@github.com:yihong0618/running_page.git
cd running_page
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt --break-system-packages
pnpm install
pnpm develop
```

2. 下载 keep 跑步数据：

```bash
python3 run_page/keep_sync.py 15608658617 Czj2638keep --sync-types=running
python3 run_page/run_to_csv.py
```

3. 更新 github action

4. 更新网站信息
