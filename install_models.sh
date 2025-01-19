#!/bin/bash

models=(
  "896ab96347d5bd781ab97959d4fd6885620e5aab52405d3445626eb7c1414b00 https://models.silero.ai/models/tts/ru/v4_ru.pt"
)

if [ ! -d "models" ]; then
  mkdir models
fi

cd models

verify_hash() {
  local file="$1"
  local expectedHash="$2"
  local currentHash

  if command -v sha256sum >/dev/null 2>&1; then
    currentHash=$(sha256sum "$file" | cut -d ' ' -f 1)
  elif command -v shasum >/dev/null 2>&1; then
    currentHash=$(shasum -a 256 "$file" | cut -d ' ' -f 1)
  else
    return 1
  fi

  if [ "$currentHash" == "$expectedHash" ]; then
    return 0
  else
    return 1
  fi
}

download_model() {
  local modelUrl="$1"

  echo "Starting download: $modelUrl"
  if curl -fSL -O "$modelUrl" --progress-bar --tlsv1.3 -A "Mozilla"; then
    echo "Completed download: $(basename "$modelUrl")"
  else
    echo "Failed download: $(basename "$modelUrl")"
    exit 1
  fi
}

for model in "${models[@]}"; do
  modelUrl="${model##* }"
  modelHash="${model%% *}"
  modelFile="$(basename "$modelUrl")"

  if [ -f "$modelFile" ] && verify_hash "$modelFile" "$modelHash"; then
    echo "Model $modelFile already exists and has the correct hash, skipping download."
  else
    download_model "$modelUrl"
  fi
done

echo "All downloads have been processed."
