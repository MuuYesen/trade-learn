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
  <img src="https://img.shields.io/badge/pypi-v1.0.0-orange?style=flat-square" alt="PyPI version">
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

## クイックスタート

**1. 投資研究プロジェクトの初期化**
```bash
# プロジェクトの骨組みを作成（設定、サンプル戦略、研究用 Notebook を含む）
tradelearn new my_research
cd my_research
```

**2. フルスタック実験環境の起動**
```bash
# JupyterLab + MCP Server + MLflow を一括起動
# 実験追跡と AI アシスタント機能をシームレスに統合
tradelearn lab
```

## 主要なハイライト

#### ⚡️ 高性能コア：Rust による極限のパフォーマンス
- **Rust ハイブリッド駆動**：コアマッチングエンジンを Rust で実装し、単一銘柄で **28倍**、多資産のリバランスで **110倍以上**（対 Backtrader 比）の加速を実現。
- **インテリジェント Runner スケジューリング**：データ形状に応じて実行モードを自動切り替え。インデックス強化シナリオ向けに最適化されたメモリレイアウトにより、大規模バックテストでも極低レイ延性を維持。

#### 🛡️ 厳格な金融工学：Backtrader セマンティクス 100% 同期
- **Engine 級の深度同期**：Analyzer / Sizer / Signal 体系を完全サポート。すべての取引（Trades）の約定ロジックが Backtrader 公式の結果と完全に一致することを保証。
- **Lite アジャイル構文**：同一の高性能 Runtime 上に構築された軽量な構文。内蔵の `target_weights` インターフェースにより、ML モデルの出力を即座に意思決定に変換。

#### 🧪 因果探索による投資研究：相関を超えた科学的フロー
- **Causal-First ファクター選定**：PC / FCI 因果探索アルゴリズムを内蔵し、真の駆動パスを特定。根源から「偽の相関」や過学習を防止。
- **パイプライン実験ループ**：特徴量エンジニアリング、因果選定、バックテスト監査をシームレスに結合し、再現可能で監査可能なプロフェッショナルな研究フローを構築。

#### 📦 モジュール型プラットフォーム：軽量コアと柔軟な拡張
- **デカップリング設計**：デフォルトインストールは高性能コアのみを含み、依存関係を最小化。サーバーバックエンドや自動取引システムへの統合が容易。
- **弾力的な拡張**：`[lab]` または `[all]` エクストラにより、統合環境（**JupyterLab + MLflow + AI アシスタント**）を一括有効化。「必要な時に必要な分だけ、どこでも実行可能」。

#### 🌍 グローバルな視野：マルチ基準のインジケーターと現代的なエコシステム
- **デュアルマーケット互換性**：TDX（A株）/ TradingView（海外）のインジケーター基準を明示的にサポート。TA-Lib や pandas-ta と深く互換。
- **モダンなツール群**：HTML インタラクティブレポート、MLflow 実験追跡、JupyterLab / MCP との統合を標準提供。

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
> `tradelearn lab` を起動した後、デフォルトで `8888` ポートから研究環境に、`5050` ポートから MLflow ダッシュボードにアクセスできます。

## ドキュメント

*   **公式オンラインドキュメント**: [**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **ローカルマニュアル**: [`docs/`](./docs/README.md)

## ロードマップ

*   **v1.0.x (現在 - 安定版)**: Rust 版クロックランナー、インデックス強化パイプライン、MLflow 統合。
*   **v1.1.x (高度な研究)**：因果推論の強化、高性能コネクタ（DolphinDB/DuckDB）、リスクモデル（Barra スタイル）。
*   **v1.2.x (実運用と自動化)**：実取引アダプター（QMT 等）、分散最適化（Ray/Optuna）、エージェント統合（MCP）。

## 連絡先

Email: muyes88@gmail.com
