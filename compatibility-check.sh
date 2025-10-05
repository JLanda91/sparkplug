#!/usr/bin/bash

# build_all_dockerfiles.sh
# Build all Dockerfile.* in the current directory
# Does not stop on failure

set -uo pipefail

echo "Obtaining CUDA devel container versions from Dockerhub..."

page=1
filtered_tags=""
response=$(curl -s "https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags?page=$page&page_size=100")
count=$(echo "$response" | jq -r '.count')
num_pages=$(( (count + 99) / 100 ))

while :; do
  tags=$(echo "$response" | jq -r '.results[].name' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+-devel-ubuntu24\.04')
  if [[ -n "$tags" ]]; then
    filtered_tags+=$'\n'"$tags"
  fi
  if [[ "$page" -eq "$num_pages" ]]; then
    break
  fi
  ((page++))
  response=$(curl -s "https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags?page=$page&page_size=100")
done

# normalize whitespace (remove leading newline if any)
filtered_tags=$(echo "$filtered_tags" | sed '/^$/d')

# extract versions (strip suffix, dedupe, sort)
cuda_versions=$(echo "$filtered_tags" | sed -E 's/^([0-9.]+)-devel-ubuntu24\.04/\1/' | sort -u)

echo "Running compatibility for CUDA versions:"
echo "$cuda_versions"
echo

trap 'echo "Build loop interrupted."; exit' INT

for cuda_version in ${cuda_versions}; do
  tag="cuda-${cuda_version}-compat"
  image_name="sparkplug-examples:${tag}"
  echo "CUDA $cuda_version ..."

  succeeded=1
  start_t=$EPOCHREALTIME
  if ! docker build -f ./docker/gcc-compat.dockerfile -t "$image_name" --build-arg CUDA_VERSION="$cuda_version" . >/dev/null 2>&1; then
    succeeded=0
  fi

  end_t=$EPOCHREALTIME
  elapsed_t=$(echo "$end_t - $start_t" | bc -l)
  echo "$elapsed_t seconds"
  if [[ $succeeded -gt 0 ]]; then
      docker run --rm "$image_name" cat /sparkplug-compat/report.txt
      echo
  else
      echo "❌ Container failed to build"
  fi
done