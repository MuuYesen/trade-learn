<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>公式ドキュメント</b></a> |
  <a href="./CHANGELOG.md"><b>更新履歴</b></a> |
  <a href="./README.md"><b>中文简体</b></a> |
  <a href="./README_en.md"><b>English</b></a>
</p>

<p align="center">
  <strong>Python で戦略と研究を、Rust でイベント駆動型バックテストエンジンを。</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v0.2.4-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Changelog-v0.2.4-blue?style=flat-square" alt="Changelog">
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
*   **グローバル戦略チーム**：国内 (TDX) と海外 (TradingView) の市場を、完全に一致した指標基準とレポート体系で管理。

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

## 同期とパフォーマンス

ベンチマークは、**結果の一致性**と Backtrader 比較での**スループット加速**を重視しています。

#### 1. 単一資産高頻度：SMA 交差 (55万 Bar)
| エンジンモード | 処理時間 | スループット (Bars/s) | **加速比** | 最終状態 |
|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | - |

#### 2. 大規模インデックス強化：Top-50 ターゲットウェイト (504万 Bar)
| エンジンモード | 処理時間 | スループット (Bars/s) | **加速比** |
|---|---|---|---|
| **Tradelearn Lite** | **2.40s** | **2,094,237** | **119.1x** |
| **Tradelearn Engine** | **4.11s** | **1,225,594** | **69.7x** |
| Backtrader (Oracle) | 286.53s | 17,589 | 1.0x |

## 一致性の保証

**trade-learn** は「ベンチマーク同期」を核心的な工程規律としています：
- **金融指標同期**: Sharpe, MaxDD 等は `empyrical` と `rtol=1e-10` で一致。
- **テクニカル指標同期**: `tl.pta` および `tl.tdx` (国内基準) は `rtol=1e-10` で一致。
- **エンジン同期**: 売買記録 (Trades) は Backtrader 公式実装と **0 差異** で一致。

## 🚀 ロードマップ (Roadmap)

[PROJECT.md](./design/PROJECT.md) の計画に基づき、以下の核心領域で進化を続けます：

#### 🏗️ v1.x（バックテストエンジンと研究インフラ）
- [x] **Rust ハイブリッドカーネル**: Clocked Multi-Data Runner による **110x+** 加速。
- [x] **Backtrader 同期**: ロジック一致率 100%、`bt.Strategy` によるランタイム共有。
- [x] **インデックス強化パイプライン**: `Data → Factor → Score → Weights` の一貫性。
- [x] **自動監査**: MLflow 統合によるコード、パラメータ、レポートの自動記録。
- [x] **高性能バックエンド**: **DuckDB ネイティブ接続の実装**。億単位の Bar を秒速で読み込み。
- [ ] **リスクモデル統合**: Barra スタイルのリスク分析と要因分解。

#### 🧪 v1.x+（科学的投資研究能力）
- [x] **因果発見の基礎**: `CausalSelector` (PC/FCI) によるアルファの真因特定。
- [ ] **アルゴリズム拡張**: GIES, Direct-LiNGAM 等による説明性の向上。
- [ ] **因果クローズドループ**: 因果分析とパラメータ最適化、リスク管理の自動統合。

#### 🤖 v1.x+（エージェントと AI 能力）
- [x] **MCP 知識ゲートウェイ**: **MCP Server の公開**。LLM による API とドキュメントの構造的理解。
- [ ] **エージェントによる戦略診断**: LLM によるバックテスト結果の自動解析と改善提案。
- [ ] **LLM ファクター解説**: 因果推論の結果を直感的な金融ロジックに翻訳。

#### ⚙️ v1.x+（エンジニアリングと ML ライフサイクル）
- [x] **モデルレジストリ**: **MLflow によるモデル管理**。特徴量とモデルバージョンの追跡。
- [ ] **分散パラメータ最適化**: Ray / Optuna によるマルチマシン並列検索。

#### 🌍 v2.x（実盤取引とエコシステム）
- [x] **共通イベントリンク**: バックテストと実盤取引でコードを 100% 再利用。
- [ ] **実盤コネクティビティ**: `QMT`, `IBKR` 等のブローカー統合。
- [ ] **Agentic Quant プラットフォーム**: 自然言語駆動の投資研究自動化基座への進化。

## 免責事項 (Disclaimer)

本プロジェクトは学術研究および技術交流のみを目的としており、投資勧誘を目的としたものではありません。クオンツ取引には高いリスクが伴い、バックテストの結果は将来の運用成績を保証するものではありません。本プロジェクトの使用によって生じた、いかなる経済的損失についても、開発者は一切の責任を負いません。投資に関する最終決定は、利用者ご自身の判断と責任で行ってください。

## 謝辞

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [DolphinDB](https://github.com/dolphindb) · [mpquant](https://github.com/mpquant)

## お問い合わせ

WeChat: 知守溪の收纳屋 · Email: muyes88@gmail.com
