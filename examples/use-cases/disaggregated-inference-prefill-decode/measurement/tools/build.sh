#!/usr/bin/env bash
# Usage: build.sh NAME; builds and pushes images/Dockerfile.NAME as $LAB_REGISTRY/aim345:measure-NAME.
set -euo pipefail
: "${LAB_REGISTRY:?}" "${BASE_IMAGE:?}"
M=$(cd "$(dirname "$0")/.." && pwd)
docker build -f "$M/images/Dockerfile.$1" --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$LAB_REGISTRY/aim345:measure-$1" "$M/patches"
docker push "$LAB_REGISTRY/aim345:measure-$1"
