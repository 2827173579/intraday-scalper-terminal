# SOXL / SOXS Intraday Terminal

SOXL / SOXS 日内信号终端（Alpaca IEX 数据；只提醒、人工下单）。

## 本地运行

```bash
python -m venv .venv
.venv/Scripts/pip install -r intraday_scalper/requirements.txt
cp intraday_scalper/.env.example intraday_scalper/.env   # 填入 Alpaca Paper key
PYTHONPATH=. python -m intraday_scalper.web.terminal_server --port 8080
```

打开 http://127.0.0.1:8080/

## 云端部署（Render，免费版）

1. 打开 https://render.com/deploy?repo=https://github.com/2827173579/intraday-scalper-terminal
2. 用 GitHub 登录，选择本仓库，`render.yaml` 会自动创建 Web Service。
3. 在 Render 的 Environment 里填写：
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - `TERMINAL_PASSWORD`（强烈建议；浏览器会弹 Basic Auth）
4. 部署完成后得到 `https://<service>.onrender.com`，在外面任何电脑都能查看。
   免费实例闲置会休眠，首次打开需要约 30–60 秒唤醒。

## 安全与口径

- 不要提交 `.env`；仓库只包含 `.env.example`。
- Alpaca IEX 是单一交易所口径，不是全市场 consolidated tape；成交量/VWAP 仅 IEX。
- 系统只产生提醒，不自动下单；不构成投资建议。
- 免费云实例不是低延迟交易通道，不能用于依赖秒级成交的实盘执行。

## 目录

本仓库是独立快照：`intraday_scalper/` 为完整包，`render.yaml`、`start.sh`、`Dockerfile` 用于云部署。
