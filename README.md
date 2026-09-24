# BTCUSDT 4H ML-Gated DCA

可复现的 BTCUSDT USD-M 永续研究回测。系统使用 4H 已收盘 OHLCV 特征判断未来 7 天的危险下跌概率，由独立 DCA 与 Risk Engine 执行下单、止盈与风险退出。项目只运行历史研究，不连接真实交易账户。

## 安装与运行

在 PowerShell 项目目录运行：

```powershell
python -m pip install -e ".[dev]"
btc-dca download
btc-dca validate --skip-download
btc-dca run --skip-download
```

`download` 只补充官方标记价格与资金费率归档；已有 5 分钟执行数据不重复下载。SHA-256 校验失败、执行 OHLCV 有缺口、K 线非法或数据时间轴无法对齐时，验证和回测会停止。若官方标记价格仍有缺值，系统保留缺失并阻止 V3 与 V1_realistic，不插值或用成交价冒充标记价。`config.toml` 可修改数据目录、费用、资金和策略参数。

## 数据与输出

默认数据目录为 `data/ohlcv_5m`、`data/mark_5m` 和 `data/funding`，可在 `config.toml` 中修改。将 Binance 官方 5 分钟 USD-M 永续 OHLCV 月度 ZIP 与 `.CHECKSUM` 文件放入 `data/ohlcv_5m`；`btc-dca download` 会补齐官方标记价格和资金费率归档。原始行情文件由 Binance 提供，不随代码仓库发布。文件命名、字段和校验方式见 [Binance Public Data 文档](https://github.com/binance/binance-public-data/blob/master/python/README.md)。研究结果保存在 `reports/generated`。

主要输出：

- `RESULTS.md` 和 PNG：策略对比、资金曲线、回撤、逐期模型指标、压力测试和研究边界。
- `*_equity.parquet`：每根 5 分钟权益、持仓、敞口和信号。
- `*_trades.csv`、`*_cycles.csv`：逐笔订单、手续费、滑点、资金费、Cycle 盈亏和退出原因；OHLC 多路径成交会标记 `ambiguous_bar`。
- `oos_predictions.parquet`、`walk_forward_metrics.csv`：逐期预测和分类指标。
- `stress_results.csv`：费用、滑点、信号延迟、执行延迟、漏单及跳空压力场景。

本仓库附带一次完整运行的结果快照，包括策略报告、权益曲线、逐笔成交、Cycle、逐期预测与压力测试。复算这些结果需要按 `config.toml` 指定范围准备对应的 Binance 历史数据。

- [完整回测结果](reports/generated/RESULTS.md)
- [2024 Q4 历史时点预测说明](reports/generated/ML_PREDICTIONS_2024Q4.md)
- `data_quality.json`、`features_4h.parquet`：数据审计和模型输入/标签。

## 回测规则

- 5 分钟 UTC K 线每 48 根聚合为完整 4H K 线，信号在收盘后确定，从下一根 5 分钟 K 线执行。
- Unsafe 标签检查之后 42 根 4H K 线最低价是否相对当前收盘价下跌至少 12%。预测特征只来自当时已收盘的 OHLCV。
- 逐期训练采用 24 个月训练、3 个月验证、3 个月测试，每期前移 3 个月；训练/验证边界各 purge 42 根 4H 标签。Logistic Regression 和固定参数 LightGBM 都用训练期 purged OOF 预测拟合 sigmoid 概率校准器。LightGBM 为 DCA 的主门控信号。
- 初始权益 2,000 USDT；Base 15%，五个 SO 各 13%；价格触发位为初始成交价的 −4%、−9%、−16%、−26%、−40%。成交后的新增订单受本轮初始权益 80% 名义仓位上限约束，价格上涨本身不触发减仓。
- 浮盈达到均价的 1.5% 后启动 trailing，最高价回撤 0.5% 全平。权益跌至 Cycle 初始权益的 55%、维持保证金比率达到 50% 或模型到达强平比率时，Risk Engine 先退出并停止本测试期后续交易。
- V1/V2 均使用单边 0.05% taker 手续费和 0.05% 滑点；V3 增加官方历史资金费率，并用标记价格计算未实现盈亏和风控。V1_realistic 为 V3 的成本/价格口径对照。若现实口径因数据质量门槛被阻止，报告会注明 V1/V2 未计资金费率。

## 结果解释边界

公开历史数据不包含你的逐时账户保证金档位，因此当前版本以 `config.toml` 中 1% 维持保证金率进行研究模拟。Binance 维持保证金率依仓位档位而变，模拟强平次数不是逐账户精确复原。5 分钟 OHLC 不包含 K 线内成交顺序；同一根 K 线触发多个 Safety Order 或止盈时使用引擎规定的保守规则。压力测试是情景分析，不能证明未来不会强平。

V1/V2 的对比在完全相同的样本外区间进行。压力测试覆盖手续费乘数、滑点、信号/API 延迟、漏单和单根 5 分钟急跌。预设研究门槛要求 V3_realistic 相对 V1_realistic 的最差 Cycle 收益和最大回撤都改善，且模拟强平为零；报告会如实显示未通过或因数据不完整而无法判定的结果，并保留逐期证据。
