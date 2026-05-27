#!/bin/bash

TAG="${TAG:-v$(date +%Y%m%d%H%M%S)}"
IMAGE_NAME="${IMAGE_NAME:-lighttomorrow/onellm-backend}"


echo "🚀 开始构建 Docker 镜像: ${IMAGE_NAME}:${TAG}"

docker build \
  -t "${IMAGE_NAME}:${TAG}" \
  .

echo "🏷️ 打 latest 标签..."
docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest"


echo "📦 推送版本标签 ${TAG}"
docker push "${IMAGE_NAME}:${TAG}"

echo "📦 推送 latest 标签..."
docker push "${IMAGE_NAME}:latest"

echo "✅ 构建并推送完成: ${IMAGE_NAME}:${TAG} + latest"
