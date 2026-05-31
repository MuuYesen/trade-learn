<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>公式ドキュメント</b></a> |
  <a href="./CHANGELOG.md"><b>更新履歴</b></a> |
  <a href="./README_zh.md"><b>中文简体</b></a> |
  <a href="./README.md"><b>English</b></a>
</p>

<p align="center">
  <strong>Python で戦略と研究を、Rust でイベント駆動型バックテストエンジンを。</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v0.2.4-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Changelog-v0.2.4-blue?style=flat-square" alt="Changelog">
  <a href="https://discord.gg/JbqZ7p33ra"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
</p>

**trade-learn** は、投資研究（Learn）とバックテスト実行（Trade）の間の摩擦を解消することを目指しています。「Python 戦略ロジック + Rust ネイティブコア」のハイブリッドアーキテクチャにより、Backtrader との **100% 厳格なロジック同期** を維持しながら、多資産バックテストで **110倍以上** のパフォーマンス向上を実現しました。大規模な検証時間を「時間単位」から「秒単位」に短縮し、インデックス強化や機械学習戦略に極速の反復体験を提供します。

高性能なエンジンに加え、**trade-learn** は完全な投資研究インフラを提供します。**JupyterLab** と **MLflow** を内蔵し、ファクターマイニング、戦略検証、実験監査をシームレスに結合。**再現可能、追跡可能、監査可能**なフルライフサイクルの研究パイプラインを構築します。これにより、研究プロセスは「結果重視」からシステム化されたエンジニアリングワークフローへと進化し、研究者は戦略の本質に集中できるようになります。

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

**極限の効率から科学的な意思決定へ**：研究パイプラインによる効率向上を基盤に、**trade-learn** は投資研究の核心である「科学性」の解決に注力しています。機械学習戦略で頻発する「偽の相関」リスクに対処するため、**因果推論 (Causal Inference)** を研究ワークフローに深く統合しました。真の因果パスを特定することで、アウトオブサンプルの劣化を抑え、高い説明性と安定性を備えたクオンツシステムを構築します。

## 実現パス

**trade-learn** は単なる機能の積み上げではなく、「デュアルモード、デュアルコア」設計により、専門的な深みと研究効率の両立を図っています。**Engine** レイヤーは Backtrader セマンティクスと厳密に同期して論理的正当性を確保し、**Lite** レイヤーは迅速な反復のための最小限の Python 互換インターフェースを提供します。

研究段階に応じて戦略の深さを定義できます：
- **Engine モード (高度な研究)**：Backtrader の Analyzer/Sizer/Signal エコシステムを完全サポート。論理的に精密で高精度なプロダクション級システムの構築に適しています。
- **Lite モード (迅速な検証)**：`backtesting.py` のミニマリズムを継承し、モデルのウェイト直結をサポート。ファクターマイニング段階での高頻度な反復とプロトタイプ検証に最適です。

エコシステム面では、TA-Lib、Pandas-TA-Classic、TDX、TradingView の主要指標をサポートし、カスタム指標やデータソースの柔軟な拡張が可能です。

## コアハイライト

#### ⚡️ 高性能カーネル：Rust 駆動の極速
- **Rust ハイブリッド動力**：マッチングエンジンとコア計算を Rust で実装し、Backtrader 比較で単一銘柄 **28倍**、多資産リバランス **110倍以上** の高速化を実現。
- **自動ランナー配置**：データの形状に応じて、Bar 毎の逐次実行かパネル一括実行を自動選択。**インデックス強化シナリオ**に最適化されたメモリレイアウト。

#### 🛡️ 厳格な金融工学：Backtrader との 100% 同期
- **Engine レベルの同期**：Analyzer / Sizer / Signal 体系を完全サポートし、Backtrader Oracle との論理的乖離をゼロに。
- **Lite 最小限の表現**：同じランタイム上に構築された軽量な構文。機械学習モデルの出力を即座にバックテストの意思決定に変換する `target_weights` インターフェースを内蔵。

#### 🧪 因果投研：相関を超えた科学的フロー
- **因果優先のファクター選択**：PC / FCI などの因果発見アルゴリズムを内蔵し、真の因果パスを特定。「偽の相関」や過学習を防止します。
- **フルリンク・パイプライン**：特徴量エンジニアリング、因果スクリーニング、スコアリングモデル、ポートフォリオウェイト、レポート生成を再現可能なループとして結合。

#### 📦 モジュール化プラットフォーム：軽量コアと柔軟な拡張
- **デカップリングされたコア**：デフォルトでは最小限の依存関係で高性能カーネルのみをインストール。サーバーや自動取引システムへの統合が容易です。
- **弾力的な拡張**：`[lab]` または `[all]` オプションにより、投資研究環境（**JupyterLab + MLflow + AI アシスタント**）を一括で有効化。

#### 🌍 グローバルな視点：多基準指標とモダンなエコシステム
- **ダブルマーケット基準**：国内 (TDX) と海外 (TradingView) の両指標基準を明示的にサポート。TA-Lib や Pandas-TA-Classic とも深く互換。
- **モダンなツール**：HTML インタラクティブレポート、MLflow 実験管理、JupyterLab / MCP の深い統合を標準提供。

## 因果推論：偽の相関の罠を越えて

多くのクオンツ研究は**統計的相関 (Correlation)** に留まっており、バックテストで良好でも実盤で急速に劣化する（過学習）リスクを抱えています。trade-learn は内蔵の **因果発見 (Causal Discovery)** メカニズムを通じて、収益の真の要因を特定します：

- **因果特徴量選択**：`CausalSelector` を使用して、共通の観測要因による「偽の相関」を排除し、収益に直接的な駆動能力を持つ特徴量のみを保持。
- **アウトオブサンプルの劣化を抑制**：因果グラフに基づくアルファファクターは、市場環境の変化に対して高い耐性を持ち、研究と実盤の性能乖離を効果的に縮小します。
- **産業レベルの統合**：`causal-learn` エコシステムを深く統合し、高度な因果推論を `corr()` を呼び出すのと同じくらいスムーズに実行できます。

## ターゲットユーザー

*   **迅速な開発者とプロトタイプ検証者**：数行のコードでアイディアをレポートに変換。`backtesting.py` のような軽量な体験を提供。
*   **インデックス強化とポートフォリオマネージャー**：Rust パネルランナーにより、1000以上の銘柄の調算を数秒でシミュレーション。
*   **機械学習とファクター研究者**：特徴量、**因果発見**、MLflow 管理のモデル訓練、バックテストをワンストップで自動化。
*   **Backtrader パワーユーザー**：信頼性の高いイベント駆動セマンティクスを維持しながら、最新のレポート体系と Rust の高速バックテストを享受。
*   **クロスマーケット / マルチ戦略チーム**：
    *   **市場横断の一貫性**：A株 (TDX) と海外市場 (TradingView) を、統一された指標基準とレポート体系で扱えます。
    *   **戦略体系の統一管理**：ルールベース戦略とモデルベース戦略を同じスタックで管理し、分断されたツールチェーンによる研究・保守コストを避けられます。
*   **因果推論の探求者**：ファクター選択に因果グラフ技術を導入し、「偽の相関」を取り除くことで、説明性と頑健性の高いクオンツシステムを構築できます。

## インストール

```bash
pip install trade-learn
```

最新バージョンの取得：

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

オプション（Extras）：

| Extra | 用途 |
|---|---|
| `[lab]` | JupyterLab / Jupyter AI / MCP / Pygwalker 投資研究環境 |
| `[mlflow]` | MLflow 実験管理とアーティファクト記録 |
| `[all]` | 完全な研究環境 (Lab, MLflow, Riskfolio-Lib, Optuna, DuckDB など) |

> **💡 インストールのヒント**:
> 標準インストールはコアエンジンのみです。JupyterLab と MLflow を含むフル体験には `[all]` を推奨します。
> ```bash
> pip install "trade-learn[all]"
> ```
> `tradelearn lab` コマンドで起動し、8888 ポートで JupyterLab、5050 ポートで MLflow にアクセス可能です。

## クイックスタート

**Lite — 最短パス**（迅速な検証、教育、ターゲットウェイト戦略に最適）：

```python
import tradelearn.lite as tl
from tradelearn.data import TradingViewProvider

class LiteSmaCross(tl.Strategy):
    fast = 10
    slow = 20

    def init(self):
        self.fast_ma = tl.tdx.MA(self.data.close, N=self.fast)
        self.slow_ma = tl.tdx.MA(self.data.close, N=self.slow)
        self.start_on_bar(self.slow + 1)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and not self.position():
            self.buy(size=100)
        elif self.fast_ma[0] < self.slow_ma[0] and self.position():
            self.position().close()

provider = TradingViewProvider(n_bars=5000)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

bt = tl.Backtest(bars, LiteSmaCross, cash=100_000, commission=0.0003, trade_on_close=True)
stats = bt.run()

print(stats.summary)
bt.plot()
bt.report("report.html")
```

> [!TIP]
> **マルチアセットロジックについて：** マルチアセットバックテストでは、戦略はデフォルトで `self.data`（主データフィード）にバインドされます。そのため、上記の例に複数銘柄を渡しても、意思決定は最初の銘柄のシグナルに基づきます。複数銘柄を独立して取引するには、戦略の `init` で `self.datas` を走査し、各フィードごとに指標を作成してください。

**Engine — Backtrader スタイル**（複雑なポートフォリオや将来の実盤モードに最適）：

```python
import tradelearn.engine as bt
from tradelearn.data import TradingViewProvider

class SmaCross(bt.Strategy):
    params = (("fast", 10), ("slow", 20))

    def __init__(self):
        self.fast = bt.tdx.MA(self.data.close, N=self.p.fast)
        self.slow = bt.tdx.MA(self.data.close, N=self.p.slow)

    def next(self):
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=100)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()

provider = TradingViewProvider(n_bars=5000)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

cerebro = bt.Cerebro(trade_on_close=True)
cerebro.setcash(100_000)
cerebro.setcommission(0.0003)
cerebro.adddata(bars, name="AAPL")
cerebro.addstrategy(SmaCross)

[strategy] = cerebro.run()
print(strategy.stats.summary)

cerebro.plot()
cerebro.report("report.html")
```

> [!TIP]
> **マルチアセットロジックについて：** マルチアセットバックテストでは、戦略はデフォルトで `self.data`（主データフィード）にバインドされます。そのため、上記の例に複数銘柄を渡しても、意思決定は最初の銘柄のシグナルに基づきます。複数銘柄を独立して取引するには、戦略の `init` で `self.datas` を走査し、各フィードごとに指標を作成してください。

## 投資研究パイプライン例

README には最短で読める版のみを載せています。完全なスクリプトは [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) と [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py) を参照してください。

**1. Research：生の行情データから特徴量を生成し、訓練 / テストに分割**

```python
import tradelearn.research as research
import tradelearn.research.preprocess as pp

feature_set = research.FeatureSet(
    {
        "alpha": lambda p: p.close.pct_change(20)
        / p.close.pct_change().rolling(20).std(),
        "size": lambda p: p.close,
    },
    target={"label": lambda p: p.close.shift(-20) / p.close - 1.0},
)

features = feature_set.fit_transform(bars, include_target=True).dropna()
train, test = research.time_split(features, split="2023-09-01", level="timestamp")
```

**2. Pipeline：前処理、モデルスコアリング、ウェイト生成**

```python
from sklearn.ensemble import GradientBoostingRegressor
import tradelearn.research.portfolio as pf

pipe = research.Pipeline(
    [
        pp.Winsorizer(columns=["alpha"], limits=(0.05, 0.95)),
        pp.Neutralizer(columns=["alpha"], exposures=["size"]),
        pp.StandardScaler(columns=["alpha"]),
    ]
)
train = pipe.fit_transform(train)
test = pipe.transform(test)

model = GradientBoostingRegressor(random_state=7)
model.fit(train[["alpha"]], train["label"])
scores = research.ModelScorer(model, features=("alpha",), current=False).predict(test)

weights = pf.Allocator(
    select=pf.TopK(k=2),
    weight=pf.EqualWeight(gross=0.95),
    constrain=pf.Constraints(max_weight=0.5, normalize=True),
).build(scores)
```

**3. Portfolio：目標ウェイトを Lite / Engine に渡して執行**

```python
class LitePortfolio(tl.Strategy):
    def next(self):
        if len(self.data) % 20 == 0:
            self.target_weights(self.research_result.weights[0], close_missing=True)


test_bars = research.split_bars(bars, split="2023-09-01")
stats = tl.Backtest(test_bars, LitePortfolio, cash=100_000).run(
    research_result=research_result
)
```

**4. Live-style：戦略内では現在見えているウィンドウだけで推論**

投資研究パイプラインはオフライン訓練と検証に適しています。より実盤に近いセマンティクスにする場合は、モデルと allocator を戦略パラメータとして渡し、`next()` の中で `history_panel()` を使って、すでに発生したデータだけを読み取ります。

```python
class LiveStylePortfolio(tl.Strategy):
    lookback = 20

    def init(self):
        self.start_on_bar(self.lookback)

    def next(self):
        if len(self.data) % 20 != 0:
            return

        panel = self.history_panel(self.lookback)
        features = self.feature_set.transform(panel).dropna()
        scores = self.scorer.predict(features)
        weights = self.allocator.build(scores)
        self.target_weights(weights, close_missing=True)
```

完全版：

| 目的 | 完全なスクリプト |
|---|---|
| Lite 投資研究 + バックテスト + report + MLflow | [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) |
| Engine 投資研究 + バックテスト + report + MLflow | [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py) |
| Lite live-style 現在ウィンドウ推論 | [`examples/research/index_enhance_lite_live.py`](./examples/research/index_enhance_lite_live.py) |
| Engine live-style 現在ウィンドウ推論 | [`examples/research/index_enhance_engine_live.py`](./examples/research/index_enhance_engine_live.py) |
| Engine Backtrader スタイルのポートフォリオ調整 | [`examples/engine/11_target_percent_portfolio.py`](./examples/engine/11_target_percent_portfolio.py) |
| 資産クラス別ポートフォリオ戦略 | [`examples/engine/12_asset_class_portfolios.py`](./examples/engine/12_asset_class_portfolios.py) |

## 同期とパフォーマンス

ローカル基準では、**結果が一致しているか**、そして Backtrader と比較して**スループットが明確に速いか**という 2 点を重視しています。完全な再現コマンドは [性能基準](./docs/benchmarks.md) を参照してください。

#### 1. 単一資産高頻度：SMA 交差 (55万 Bar)
* **戦略原理**：標準的な二重移動平均クロスを実行します。長い単一データストリームに対する Rust のイベント駆動性能と状態管理効率を測り、単一コア推進の限界を確認します。

| エンジンモード | 処理時間 | スループット (Bars/s) | **加速比** | 最終資産 | 注文数 | クローズ済み取引 | 状態 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 | - |

#### 2. 大規模インデックス強化：Top-50 ターゲットウェイト (504万 Bar)
* **戦略原理**：1000銘柄規模の全市場選股とリバランスを模擬します。Rust の大規模 Panel データ向けメモリレイアウト最適化と並行処理能力を測り、機械学習戦略の研究シナリオを再現します。

| エンジンモード | 処理時間 | スループット (Bars/s) | **加速比** | 最終資産 | 完了注文 | リバランス意図 | リバランス回数 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **2.40s** | **2,094,237** | **119.1x** | **4,199,638.26** | 23,249 | 23,249 | 239 |
| **Tradelearn Engine** | **4.11s** | **1,225,594** | **69.7x** | **4,199,638.26** | 23,249 | 23,249 | 239 |
| Backtrader (Oracle) | 286.53s | 17,589 | 1.0x | 4,199,638.26 | 23,249 | 23,249 | 239 |

## 一致性の保証

**trade-learn** は「ベンチマーク同期」を核心的な工程規律としています。各計算結果が厳密な検証に耐えられるよう、以下の層で数値一致を維持します：

*   **金融指標同期**：`metrics`（Sharpe, MaxDD, Sortino など）は `empyrical` と `rtol=1e-10` で一致。
*   **複数ソースの指標同期**：
    *   `tl.pta`（クラシック指標）は `pandas-ta-classic` と `rtol=1e-10` で一致。
    *   `tl.tdx`（TDX セマンティクス）は `MyTT` と `rtol=1e-10` で一致。
    *   `tl.tv`（TradingView セマンティクス）は `pyneCore` と `rtol=1e-6` で一致。
*   **バックテストエンジン同期**：
    *   **意思決定層**：取引記録 (**Trades**) は Backtrader 公式実装と、時刻・方向・ポジションで **0 差異**。
    *   **純資産層**：Equity 曲線は `rtol=1e-6`、サマリー統計は `rtol=1e-4` で一致。

> [!IMPORTANT]
> すべての数値差分に対してゼロトレランスで臨みます。すべての偏差は登録され、理由を説明します。詳しくは [設計ノート → セマンティクス一致性監査](docs/internals/consistency.md) を参照してください。

## 完全ドキュメント

*   **公式オンラインドキュメント**：[**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **ローカル技術マニュアル**：[`docs/`](./docs/README.md)

| テーマ | 入口 |
|---|---|
| 30 行で最初のバックテスト | [クイックスタート](./docs/quickstart.md) |
| Lite / Engine の使い方 | [Lite ガイド](./docs/guides/lite.md) · [Engine ガイド](./docs/guides/engine.md) |
| アーキテクチャと境界 | [アーキテクチャ](./docs/concepts/architecture.md) |
| Factor / ML / Weight 研究パイプライン | [Research ガイド](./docs/guides/research.md) |
| 二重基準指標（`tl.talib` / `tl.pta` / `tl.tdx` / `tl.tv`） | [Indicators ガイド](./docs/guides/indicators.md) |
| 性能基準 | [Benchmarks](./docs/benchmarks.md) |
| カーネル内部（契約 / マッチング / portfolio / イベントループ） | [設計ノート](./docs/internals/contracts.md) |
| 完全 API | [API リファレンス](./docs/api/reference.md) |

## 🚀 ロードマップ (Roadmap)

現在の工程計画に基づき、**trade-learn** は以下の核心領域で進化を続けます：

#### バックテストエンジンと核心基盤
- [x] **Rust ハイブリッドカーネル**：Clocked Multi-Data Runner により、多資産バックテストで **110x+** 加速。
- [x] **Backtrader セマンティクス同期**：マッチングロジック 100%、`bt.Strategy` によるランタイム共有。
- [x] **インデックス強化パイプライン**：`Data → Factor → Score → Weights → target_weights()` の全フローを接続。
- [x] **自動実験監査**：MLflow と深く統合し、コードスナップショット、パラメータ、指標、レポートを自動記録。
- [x] **高性能データバックエンド**：**DuckDB ネイティブコネクタが実装済み**。億単位の Bar に対するローカル秒級読み込みと横断クエリをサポート。
- [ ] **リスクモデル統合**：Barra スタイルのリスクエクスポージャー分析と超過収益分解をサポート。

#### 科学的投資研究能力
- [x] **因果発見基盤**：`CausalSelector` (PC/FCI) を統合し、特徴量工程の段階で Alpha の真のドライバーを特定。
- [ ] **アルゴリズム拡張**：GIES、Direct-LiNGAM などを導入し、説明性と安定性を向上。
- [ ] **因果駆動クローズドループ**：因果分析、パラメータ最適化、リスク管理を自動化された研究ループとして接続。

#### エージェントと AI 能力
- [x] **MCP 知識ゲートウェイ**：**MCP Server は公開済み**。AI による API の構造理解と自動コード生成を可能にします。
- [ ] **Agentic 戦略診断**：LLM でバックテスト結果を解析し、損失要因を特定してロジック改善案を提示。
- [ ] **LLM ファクター解釈器**：因果発見の結果を直感的な金融投資ロジックへ翻訳。

#### エンジニアリングと ML ライフサイクル
- [x] **モデルレジストリ**：**MLflow ベースのモデル登録**により、特徴量指紋とモデルバージョンを全ライフサイクルで追跡。
- [ ] **分散パラメータ最適化**：Ray / Optuna によるマルチマシンのパラメータ探索とモンテカルロシミュレーション。

#### 実盤取引とエコシステム構想
- [x] **共通実盤イベントリンク**：`EventRunner` セマンティクスを完成し、バックテストと実盤取引でコードを 100% 再利用可能に。
- [ ] **実盤取引接続**：`QMT`, `IBKR` などのブローカーを統合し、研究から実行までの最後の一歩を完成。
- [ ] **Agentic Quant プラットフォーム**：自然言語駆動のエンドツーエンド投資研究自動化基盤へ進化。

## 免責事項 (Disclaimer)

本プロジェクトは学術研究および技術交流のみを目的としており、投資勧誘を目的としたものではありません。クオンツ取引には高いリスクが伴い、バックテストの結果は将来の運用成績を保証するものではありません。本プロジェクトの使用によって生じた、いかなる経済的損失についても、開発者は一切の責任を負いません。投資に関する最終決定は、利用者ご自身の判断と責任で行ってください。

## 謝辞

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## お問い合わせ

WeChat: 知守溪の收纳屋 · Email: muyes88@gmail.com
