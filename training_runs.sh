#! /bin/bash

set -euo pipefail

docker_prefix="docker run --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace  -v /mnt/qnapdata/internal/:/mnt/qnapdata/internal/:ro autoware-ml-ptv3"

training_configs=(
  "j6gen2_aug01_drivable01"
  "j6gen2_aug01"
  "j6gen2_baseline"
)

test_runs_models=(
  "j6gen2_aug01_drivable01"
  "j6gen2_aug01"
  "j6gen2_baseline"
  "j6gen2_baseline"
)

test_runs_configs=(
  "j6gen2_augtest05_drivable01"
  "j6gen2_augtest05"
  "j6gen2_augtest05_drivable01"
  "j6gen2_augtest05"
)

config_base="projects/PTv3/configs/"
save_base="work_dirs/experiments/augmentation"

function config_path() {
  echo "$config_base/$1.py"
}

function save_path() {
  echo "$save_base/$1"
}

function weight_path() {
  local save_dir
  save_dir="$(save_path "$1")"
  echo "$save_dir/model/model_best.pth"
}

function echo_then_do() {
  echo "$@";
  eval "$@"
}

function docker_run() {
  echo_then_do $docker_prefix "$@"
}

########################################################
# Validation
########################################################

all_configs=(
  "${training_configs[@]}"
  "${test_runs_models[@]}"
  "${test_runs_configs[@]}"
)

# Validate that all config files exist
for config_name in "${all_configs[@]}"; do
  config=$(config_path "$config_name")
  if [ ! -f "$config" ]; then
    echo "Config file not found: $config"
    exit 1
  fi
done

# Validate configs from within Docker
echo "==> Validating configs from within Docker..."
for config_name in "${all_configs[@]}"; do
  config=$(config_path "$config_name")
  echo "    > Validating: $config_name"
  if ! docker_run bash -c "python -c 'from engines.defaults import default_config_parser; default_config_parser(\"$config\", None)'" > /dev/null 2>&1; then
    echo "      $config_name: Validation failed"
    echo "      Error details:"
    docker_run bash -c "python -c 'from engines.defaults import default_config_parser; default_config_parser(\"$config\", None)'" 2>&1 | sed 's/^/      /'
    exit 1
  fi
  echo "      ✓ $config_name: Valid"
done
echo "==> All configs validated successfully!"

#validate that number of test run models is equal to number of test run configs
if [ ${#test_runs_models[@]} -ne ${#test_runs_configs[@]} ]; then
  echo "Number of test run models is not equal to number of test run configs"
  exit 1
fi

########################################################
# Training
########################################################

# for train_config_name in "${training_configs[@]}"; do
#   train_config=$(config_path "$train_config_name")
#   docker_run python projects/PTv3/tools/train.py --config-file "$train_config" --num-gpus 1 \
#     --options \
#     save_path="$(save_path "$train_config_name")"
# done

########################################################
# Testing
########################################################

n_test_runs=${#test_runs_models[@]}

for i in $(seq 0 $((n_test_runs - 1))); do
  test_model_name="${test_runs_models[$i]}"
  test_config_name="${test_runs_configs[$i]}"
  test_config=$(config_path "$test_config_name")
  docker_run python projects/PTv3/tools/test.py --config-file "$test_config" --num-gpus 1 \
    --options \
    save_path="$(save_path "${test_model_name}-${test_config_name}")" \
    weight="$(weight_path "${test_model_name}")"
done