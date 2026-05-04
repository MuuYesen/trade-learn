<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>公式ドキュメント</b></a> |
  <a href="./README.md"><b>中文版</b></a> |
  <a href="./README_en.md"><b>English</b></a>
</p>

<p align="center">
  <strong>Python で戦略と投資研究を、Rust で高性能バックテストエンジンを。</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v0.2.2-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square" alt="Code style">
</p>

**trade-learn** は、投資研究（Learn）とバックテスト実行（Trade）の間の摩擦を解消することを目指しています。「Python ロジック + Rust ネイティブコア」のハイブリッドエンジンにより、Backtrader との **100% 厳格なロジック同期** を維持しながら、多資産バックテストで **110倍以上** のパフォーマンス向上を実現しました。大規模な検証時間を「時間単位」から「秒単位」に短縮し、インデックス強化や機械学習戦略に極速の反復体験を提供します。

極限のパフォーマンスに加え、本プロジェクトは研究の科学性にも注力しています。機械学習戦略で頻発する「偽の相関」リスクに対処するため、**因果推論 (Causal Inference)** を研究ワークフローに深く統合しました。ファクターの真の因果駆動パスを特定することで、アウトオブサンプルの劣化を効果的に抑え、高い説明性と安定性を備えたクオンツシステムの構築を支援します。

科学的な手法を実用的な生産性に変えるため、trade-learn は単なる高性能エンジンに留まらず、**JupyterLab と MLflow を内蔵**したフルライフサイクルの投資研究パイプラインを提供します。ファクターマイニング、戦略検証、実験監査をシームレスに結合し、あらゆる研究上の意思決定を追跡可能にすることで、研究者が戦略開発の本質に集中できる環境を整えます。

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

## 実現パス

**trade-learn** は「分層設計、デュアルモード駆動」のアーキテクチャを採用し、専門的な深みと研究効率の両立を図っています。

研究の段階に応じて、戦略の表現方法を柔軟に選択できます：
- **Engine モード (深度同期)**：Analyzer / Sizer / Signal エコシステムをフルサポート。下層の Engine は Backtrader のセマンティクスと高度に同期しており、精密で複雑なシステムの構築に適しています。
- **Lite モード (アジャイル反復)**：極限のミニマリズムを追求し、モデルの重み（Weights）との直接連携をサポート。内蔵の `target_weights` インターフェースにより、研究成果をワンクリックでバックテストの意思決定に変換でき、ファクターマイニング段階での高頻度な反復に最適です。

さらに、**CausalSelector** を通じて因果探索を自動化パイプラインに組み込んでいます。特徴量選定、パラメータ最適化、バックテスト監査を自動的に接続し、選定されたすべてのファクターが単なる統計的な偶然ではなく、真の因果的な説明力を持つことを保証します。

## 主要なハイライト

#### ⚡️ 高性能コア：Rust による極限のパフォーマンス
- **Rust ハイブリッド駆動**：コアマッチングエンジンを Rust で実装し、単一銘柄で **28倍**、多資産のリバランスで **110倍以上**（対 Backtrader 比）の加速を実現。
- **インテリジェント Runner スケジューリング**：データ形状に応じて実行モードを自動切り替え。**インデックス強化シナリオ向けに最適化されたメモリレイアウト**により、大規模バックテストでも極低レイ延性を維持。

#### 🛡️ 厳格な金融工学：Backtrader セマンティクス 100% 同期
- **Engine 級の深度同期**：Analyzer / Sizer / Signal 体系を完全サポート。すべての取引（Trades）の約定ロジックが Backtrader 公式の結果と完全に一致することを保証し、コンポーネントの自己拡張を高度にサポートします。
- **Lite アジャイル構文**：同一の高性能 Runtime 上に構築された軽量な構文。**内蔵の `target_weights` インターフェース**により、ML モデルの出力を即座に意思決定に変換。

#### 🧪 因果探索による投資研究：相関を超えた科学的フロー
- **Causal-First ファクター選定**：PC / FCI 因果探索アルゴリズムを内蔵し、真の駆動パスを特定。根源から「偽の相関」や過学習を防止。
- **パイプライン実験ループ**：特徴量エンジニアリング、因果選定、スコアリングモデル、ポートフォリオの重み、およびバックテストレポートを再現可能な実験ループとしてシームレスに結合。

#### 📦 モジュール型プラットフォーム：軽量コアと柔軟な拡張
- **デカップリング設計**：デフォルトインストールは高性能コアのみを含み、依存関係を最小化。サーバーバックエンドや自動取引システムへの統合が容易。
- **弾力的な拡張**：`[lab]` または `[all]` エクストラにより、統合環境（**JupyterLab + MLflow + AI アシスタント**）を一括有効化。「必要な時に必要な分だけ、どこでも実行可能」。

#### 🌍 グローバルな視野：マルチ基準のインジケーターと現代的なエコシステム
- **デュアルマーケット互換性**：TDX（A株）/ TradingView（海外）のインジケーター基準を明示的にサポート。TA-Lib や pandas-ta と深く互換。
- **モダンなツール群**：HTML インタラクティブレポート、MLflow 実験追跡、JupyterLab / MCP との統合を標準提供。

## 因果研究：「偽の相関」の罠を越えて

多くのクオンツ研究は **統計的な相関 (Correlation)** に留まっており、これがバックテストでは優秀でも実運用では急速に劣化する（過学習）主な原因となります。trade-learn は内蔵の **因果探索 (Causal Discovery)** メカニズムにより、収益の背後にある真の要因を特定します。

- **因果的な特徴量選定**：`CausalSelector` と PC / FCI アルゴリズムを組み合わせ、「共通の観測」によって生じる偽の相関ファクターを排除し、収益に対して直接的な駆動能力を持つ特徴量のみを保持します。
- **アウトオブサンプルの劣化への耐性**：因果グラフに基づいて特定されたアルファファクターは、市場のスタイルが変化した際にも強い生存能力を持ち、研究から実運用への性能ギャップを効果的に低減します。
- **産業級の統合**：`causal-learn` エコシステムと深く統合されており、最先端の因果推論技術を `corr()` を呼び出すようにスムーズに利用でき、アカデミックなアルゴリズムの導入障壁を大幅に下げます。

## 対象ユーザー

*   **アジャイル開発者とアイデア検証**:
    重い設定を嫌い、数行のコードでアイデアをバックテストレポートに変換したい方。backtesting.py のような軽快な体験を求める方に最適です。
*   **インデックス強化とポートフォリオ管理**:
    1000銘柄以上の大規模なバックテストに直面している方。Rust Panel Runner を利用して秒単位のリバランスシミュレーションを実現します。
*   **機械学習とファクター研究**:
    特徴量エンジニアリング、**因果探索**、モデル訓練（MLflow 追跡）、バックテストをワンストップの自動化ループとして構築したい方。
*   **Backtrader のパワーユーザー**:
    成熟したイベント駆動型のセマンティクスを保持しつつ、より現代的なレポート体系、フルリンク・パイプライン、および高性能な Rust バックテストコアを求める方。
*   **跨市場・多戦略チーム**:
    *   **基準の統一**: A株（TDX）と海外（TradingView）の両方をカバーし、インジケーター基準とレポート体系の完全な一致を求めるチーム。
    *   **全体系の維持**: ルールベース戦略とモデル戦略を統一管理し、ツールチェーンの断片化による開発・維持コストを削減したい方。
*   **因果推論の探求者**:
    ファクター選定段階に因果グラフ技術を導入し、「偽の相関」を排除することで、説明性が高く堅牢なクオンツシステムを構築したい方。

## インストール

```bash
pip install trade-learn
```

最新バージョンの取得：

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

オプションの extras：

| extra | 用途 |
|---|---|
| `[lab]` | JupyterLab / Jupyter AI / MCP / Pygwalker 交互研究環境 |
| `[mlflow]` | MLflow tracking server と実験アーティファクトの記録 |
| `[all]` | Lab, MLflow, Riskfolio-Lib, Optuna, DuckDB 等を含むフル環境 |

> **💡 インストールのアドバイス**:
> デフォルトのインストールにはコアバックテストエンジンのみが含まれています。JupyterLab や MLflow を含むフルスタックの研究体験を有効にするには、`[all]` エクストラを指定してください：
> ```bash
> pip install "trade-learn[all]"
> ```
> プロジェクトルートで `tradelearn lab` を起動した後、デフォルトで `8888` ポートから研究環境に、`5050` ポートから MLflow ダッシュボードにアクセスできます。

## クイックスタート

**Lite — 最短パス**（迅速な検証、教育、多資産のターゲットウェイトに最適）：

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
> **マルチシンボル・ロジックについて：** マルチシンボル（複数銘柄）のバックテストでは、ストラテジーはデフォルトで `self.data`（プライマリーデータソース）にバインドされます。つまり、上記のコードに複数の銘柄を渡しても、最初の銘柄のシグナルのみに基づいて判断されます。複数銘柄で独立した並列取引を行う場合は、ストラテジーの `init` 内で `self.datas` をループし、各データソースに対してインジケーターを構築する必要があります。

**Engine — Backtrader スタイル**（複雑 / 組合せ戦略や将来の paper / live モードに最適）：

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
> **マルチシンボル・ロジックについて：** マルチシンボル（複数銘柄）のバックテストでは、ストラテジーはデフォルトで `self.data`（プライマリーデータソース）にバインドされます。つまり、上記のコードに複数の銘柄を渡しても、最初の銘柄のシグナルのみに基づいて判断されます。複数銘柄で独立した並列取引を行う場合は、ストラテジーの `init` 内で `self.datas` をループし、各データソースに対してインジケーターを構築する必要があります。

## 投資研究パイプラインの例

README には最短の可読バージョンのみを掲載しています。完全なスクリプトは [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) および [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py) を参照してください。

**1. Research：生データから特徴量を生成し、訓練/テストセットを分割**

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

**2. Pipeline：前処理、モデルスコアリング、ターゲットウェイトの生成**

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

**3. Portfolio：Lite / Engine を通じてターゲットウェイトを実行**

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

**4. Live-style：現在の可視ウィンドウのみを使用した推論**

投資研究パイプラインはオフラインの学習に適しています。実運用に近いセマンティクスが必要な場合は、`next()` 内で `history_panel()` を使用してください。

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

完全版のスクリプト一覧：

| 目標 | 完全版スクリプト |
|---|---|
| Lite 研究 + バックテスト + レポート + MLflow | [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) |
| Engine 研究 + バックテスト + レポート + MLflow | [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py) |
| Lite 実運用スタイル 現在ウィンドウ推論 | [`examples/research/index_enhance_lite_live.py`](./examples/research/index_enhance_lite_live.py) |
| Engine 実運用スタイル 現在ウィンドウ推論 | [`examples/research/index_enhance_engine_live.py`](./examples/research/index_enhance_engine_live.py) |
| Engine Backtrader スタイル ポートフォリオ再構築 | [`examples/engine/11_target_percent_portfolio.py`](./examples/engine/11_target_percent_portfolio.py) |
| 資産クラス別ポートフォリオ戦略 | [`examples/engine/12_asset_class_portfolios.py`](./examples/engine/12_asset_class_portfolios.py) |

## 同期と性能

ローカルのベースラインは、「**結果の同期**」と「**Backtrader より明らかに高速か**」の 2 点に焦点を当てています。詳細は [性能ベンチマーク](./docs/benchmarks.md) を参照してください。

#### 1. 単一銘柄の高頻度負荷テスト：SMA Cross (55万 Bar)
* **戦略**: 標準的な SMA クロス。長期間・単一データ流における Rust のイベント駆動性能と状態維持効率をテストします。

| エンジンモード | 処理時間 | スループット (Bars/s) | **加速比** | 最終資産 | 約定数 | 決済済み取引 | 状態 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 | - |

#### 2. 多銘柄の大規模指増：Top-50 ターゲットウェイト (504万 Bar)
* **戦略**: 1000銘柄の市場全体選別。大規模 Panel データに対する Rust のメモリレイアウト最適化と並列処理能力をテストします。

| エンジン | 実行時間 | bars/s | **加速比** | 同期値 |
|---|---:|---:|---:|---:|
| **Lite** | 2.40s | **2,094,237** | **119.1x** | 最終資産 4,199,638.26 |
| **Engine** | 4.11s | **1,225,594** | **69.7x** | 最終資産 4,199,638.26 |
| Backtrader | 286.53s | 17,589 | 1.0x | 最終資産 4,199,638.26 |

## 一致性のコミットメント

**trade-learn** は「ベースラインとの比較」を核心的なエンジニアリング規律としています。すべての計算結果が以下の次元で数値的に一致することを保証します：

*   **金融指標の同期**: Sharpe, MaxDD, Sortino 等は `empyrical` と完全に一致（誤差 `rtol=1e-10` 以内）。
*   **マルチソース指標の同期**:
    *   `tl.pta`: `pandas-ta-classic` と一致（`rtol=1e-10`）。
    *   `tl.tdx`: `MyTT` と一致（`rtol=1e-10`）。
    *   `tl.tv`: `pyneCore` と一致（`rtol=1e-6`）。
*   **バックテストエンジンの同期**:
    *   **意思決定層**: 取引記録 (**Trades**) は Backtrader 公式実装と **0 差異**（時間、方向、ポジションが完全一致）。
    *   **資産曲線層**: Equity 曲線誤差 `rtol=1e-6`、要約統計量誤差 `rtol=1e-4`。

> [!IMPORTANT]
> 私たちは数値の微細な不一致に対して「ゼロ容忍」の姿勢をとっています。すべての偏差は理由分析とともに記録されています。詳細は [設計ノート → セマンティクス一致性監査](docs/internals/consistency.md) を参照してください。

## 完整ドキュメント

*   **公式オンラインドキュメント**: [**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **ローカル技術マニュアル**: [`docs/`](./docs/README.md)

| テーマ | 入口 |
|---|---|
| 30行で最初のバックテスト | [クイックスタート](./docs/quickstart.md) |
| Lite / Engine の使い方 | [Lite ガイド](./docs/guides/lite.md) · [Engine ガイド](./docs/guides/engine.md) |
| アーキテクチャと境界 | [アーキテクチャ](./docs/concepts/architecture.md) |
| ファクター / ML / 投資研究パイプライン | [Research ガイド](./docs/guides/research.md) |
| マルチ基準インジケーター | [インジケーターガイド](./docs/guides/indicators.md) |
| 性能ベンチマーク | [ベンチマーク](./docs/benchmarks.md) |
| 内部構造（コントラクト / 撮合 / ポートフォリオ） | [設計ノート](./docs/internals/contracts.md) |
| API リファレンス | [API リファレンス](./docs/api/reference.md) |

## 🚀 ロードマップ

*   **v1.0.x (Stable Release - 現在)**
    *   [x] Rust ベースの多銘柄 Clocked Runner。
    *   [x] 完全なインデックス強化パイプライン (Research -> Weight -> Backtest)。
    *   [x] MLflow 追跡とモダンな HTML レポートの統合。
*   **v1.1.x (Advanced Research)**
    *   [ ] **因果推論の強化**: GIES, Direct-LiNGAM 等の導入。
    *   [ ] **高性能コネクタ**: DolphinDB および DuckDB との直接連携。
    *   [ ] **リスクモデル**: Barra スタイル分析と収益分解。
*   **v1.2.x (Live & Production)**
    *   [ ] **実運用アダプター**: QMT 等の証券会社接続。
    *   [ ] **分散最適化**: Ray/Optuna による並列検索。
    *   [ ] **Agent 統合**: MCP による研究の自動制御。

## 謝辞

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## 連絡先

Email: muyes88@gmail.com
