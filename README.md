# Audio Processor

## プロジェクト概要

日本語の営業・買取商談音声を解析し、商品情報の抽出、会話要約、商談フェーズ判定、フォローアップ質問生成を行うバックエンドです。

## 主な機能

- S3 上の音声ファイルを取得し、音声前処理と文字起こしを実行
- 文字起こし結果から商品情報、営業質問、顧客感情、会話フェーズを抽出
- 会話内容を要約し、商品情報とあわせてベクトルDBに保存
- 過去の類似会話を参照して、商談フェーズに応じた追加質問を生成
- 商品名をもとに外部情報を取得し、商品説明を補完

## 使用した技術

- FastAPI
- OpenAI API
- AWS S3 / boto3
- Qdrant / LlamaIndex
- Pydantic
- librosa / soundfile / numpy
- BeautifulSoup / SerpApi / ZenRows
