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
python3 run_page/keep_sync.py $KEEP_USERNAME $KEEP_PASSWORD --with-gpx --sync-types=running
python3 run_page/run_to_csv.py
```

3. 更新 github action

4. 更新网站信息

5. 同步代码

```bash
# 1) 检查远端与分支
git status -sb
git remote -v
git branch --show-current

# 2) 拉取所有远端最新
git fetch --all --prune

# 3) 先把 fork(origin) 拉齐到本地（推荐快进）
git pull --ff-only origin master

# 4) 合并上游(upstream)到本地
git merge upstream/master

# 5) 如有冲突：优先保留你自己的“数据/产物文件”
#    （常见：assets/*.svg, src/static/activities.json, assets/running.csv 等）
git checkout --ours -- assets
git checkout --ours -- src/static/activities.json
git checkout --ours -- assets/running.csv

# 6) 处理“deleted by us”的冲突（若这些年份文件你本地本来就删掉了）
git rm -f assets/year_{2012,2013,2014,2015,2016,2017,2018,2019,2020,2021}.svg

# 7) 标记冲突已解决并提交合并
git add assets src/static/activities.json assets/running.csv
git commit -m "Merge upstream/master"

# 8) 推送到你的 fork
git push origin master

# 9) 验证（按本项目 scripts）
pnpm install --frozen-lockfile
pnpm run lint
pnpm run build

# 10) 如果 push 被拒绝（远端有新提交）
git pull --rebase origin master
git push origin master

# 11) 需要强制让本地对齐远端（谨慎，会丢本地未推送提交/改动）
git fetch origin
git reset --hard origin/master
```
