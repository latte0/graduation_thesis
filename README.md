# 卒論プロジェクト
## カーネル密度推定を用いた汎用的な活性化関数  

卒業論文で開発したソフトウェアと実験のリポジトリです。

卒業論文は公開リポジトリとしてアーカイブしてあります。
[卒業論文](https://github.com/latte0/RG-Thesis-Template "卒業論文")

## ディレクトリ構成

```
.
├── misc              # 実験には直接関係ないテストプログラム
├── exp               # 実験の途中で使用していたプログラム
...
├── requirement.txt   # kaf.pyの実行に必要なライブラリのバージョン
└── kaf.py            # 本論文のメインの実験プログラム
```



## 実行方法

```
python kaf.py --use_clipping=false --show_kaf=false
```

- use_clipping: クリッピングを用いるかどうか
- show_kaf: K-AFの形状を表示するかどうか。（表示すると遅くなる。)