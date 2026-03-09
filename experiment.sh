#!/bin/bash

set -euo pipefail

cpa_path="$HOME/lidar_semseg_copy_paste_augmentation"
awml_path="$HOME/AWML"
output_path="/mnt/qnapdata/internal/user/max/experiments/augmentation"

base_t4_scenes_root="/mnt/qnapdata/internal/t4datasets/db_j6gen2_semaseg_v1"
if [ ! -d "$base_t4_scenes_root" ]; then
  echo "==> Error: Base T4 dataset root does not exist: $base_t4_scenes_root"
  exit 1
fi

base_split_yaml="$awml_path/autoware_ml/configs/t4dataset/db_j6gen2_semaseg_v1.yaml"
if [ ! -f "$base_split_yaml" ]; then
  echo "==> Error: Base split YAML does not exist: $base_split_yaml"
  exit 1
fi
split_yaml_root=$(dirname "$base_split_yaml")

base_cfg_py="$awml_path/autoware_ml/configs/segmentation3d/dataset/t4dataset/template.py"
if [ ! -f "$base_cfg_py" ]; then
  echo "==> Error: Base config Python file does not exist: $base_cfg_py"
  exit 1
fi
cfg_py_root=$(dirname "$base_cfg_py")

base_model_cfg_py="$awml_path/projects/PTv3/configs/template.py"
if [ ! -f "$base_model_cfg_py" ]; then
  echo "==> Error: Base model config Python file does not exist: $base_model_cfg_py"
  exit 1
fi
model_cfg_root=$(dirname "$base_model_cfg_py")

pkl_root="$output_path/info"

echo "==> Preparing experiment workspace at: $output_path"
mkdir -p "$output_path"

echo "==> Creating object stores (train/val/test) with split_object_store"
uv run --project "$cpa_path" split_object_store --splits 4 1 0.5 --seed 0 "$cpa_path/assets/" "$output_path"/object_stores/{train,val,test}


test_split_yaml="$split_yaml_root/test.yaml"
echo "==> Creating test split YAML: $test_split_yaml"
yq '.train = [] | .val = []' "$base_split_yaml" > "$test_split_yaml"

train_val_split_yaml="$split_yaml_root/train_val.yaml"
echo "==> Creating train/val split YAML: $train_val_split_yaml"
yq '.test = []' "$base_split_yaml" > "$train_val_split_yaml"

baseline_name="baseline"
# echo "==> Linking baseline dataset and split YAML"
# ln -s -f "$base_t4_scenes_root" "$output_path/$baseline_name"
# baseline_split_yaml="$split_yaml_root/$baseline_name.yaml"
# # Create relative symlink so it works inside Docker
# cd "$split_yaml_root" && ln -s -f "$(basename "$train_val_split_yaml")" "$(basename "$baseline_split_yaml")" && cd - > /dev/null

# aug_name="augtest-0.5-drivable-under-0.1"
# echo "==> Running test-only augmentation: $aug_name"
# aug_split_yaml="$split_yaml_root/$aug_name.yaml"
# cd "$split_yaml_root" && ln -s -f "$(basename "$test_split_yaml")" "$(basename "$aug_split_yaml")" && cd - > /dev/null
# uv run --project "$cpa_path" augment_batch \
#   --db-dir "$base_t4_scenes_root" \
#   --splits-yaml "$aug_split_yaml" \
#   --output-dir "$output_path/$aug_name" \
#   --test-objects "$output_path/object_stores/test/" \
#   --ratio 0.5 \
#   --seed 0 \
#   --category-refine-rule '{when: height_m <= 0.1, then: drivable_surface}'

aug_name="augtest-0.5"
echo "==> Running test-only augmentation: $aug_name"
aug_split_yaml="$split_yaml_root/$aug_name.yaml"
cd "$split_yaml_root" && ln -s -f "$(basename "$test_split_yaml")" "$(basename "$aug_split_yaml")" && cd - > /dev/null
uv run --project "$cpa_path" augment_batch \
  --db-dir "$base_t4_scenes_root" \
  --splits-yaml "$aug_split_yaml" \
  --output-dir "$output_path/$aug_name" \
  --test-objects "$output_path/object_stores/test/" \
  --ratio 0.5 \
  --seed 0

# aug_name="aug-0.1"
# echo "==> Running train/val augmentation: $aug_name"
# aug_split_yaml="$split_yaml_root/$aug_name.yaml"
# cd "$split_yaml_root" && ln -s -f "$(basename "$train_val_split_yaml")" "$(basename "$aug_split_yaml")" && cd - > /dev/null
# uv run --project "$cpa_path" augment_batch \
#   --db-dir "$base_t4_scenes_root" \
#   --splits-yaml "$aug_split_yaml" \
#   --output-dir "$output_path/$aug_name" \
#   --train-objects "$output_path/object_stores/train/" \
#   --val-objects "$output_path/object_stores/val/" \
#   --ratio 0.1 \
#   --seed 0

# aug_name="aug-0.1-drivable-under-0.1"
# echo "==> Running category-filtered augmentation: $aug_name"
# aug_split_yaml="$split_yaml_root/$aug_name.yaml"
# cd "$split_yaml_root" && ln -s -f "$(basename "$train_val_split_yaml")" "$(basename "$aug_split_yaml")" && cd - > /dev/null
# uv run --project "$cpa_path" augment_batch \
#   --db-dir "$base_t4_scenes_root" \
#   --splits-yaml "$aug_split_yaml" \
#   --output-dir "$output_path/$aug_name" \
#   --train-objects "$output_path/object_stores/train/" \
#   --val-objects "$output_path/object_stores/val/" \
#   --ratio 0.1 \
#   --seed 0 \
#   --category-refine-rule '{when: height_m <= 0.1, then: drivable_surface}'

function escape_path() {
  echo "$1" | sed 's/\//\\\//g'
}

# config_name="j6gen2_$baseline_name"
# echo "==> Generating dataset/model config pair: $config_name"
# cp "$base_cfg_py" "$cfg_py_root/$config_name.py"
# sed -i "s/\#CONFIG_NAME#/$config_name/" "$cfg_py_root/$config_name.py"
# sed -i 's/#VERSION_NAMES#/"baseline"/' "$cfg_py_root/$config_name.py"
# cp "$base_model_cfg_py" "$model_cfg_root/$config_name.py"
# sed -i "s/#DATA_ROOT#/$(escape_path "$output_path")/" "$model_cfg_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$model_cfg_root/$config_name.py"

# config_name="j6gen2_aug01"
# echo "==> Generating dataset/model config pair: $config_name"
# cp "$base_cfg_py" "$cfg_py_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$cfg_py_root/$config_name.py"
# sed -i 's/#VERSION_NAMES#/"aug-0.1"/' "$cfg_py_root/$config_name.py"
# cp "$base_model_cfg_py" "$model_cfg_root/$config_name.py"
# sed -i "s/#DATA_ROOT#/$(escape_path "$output_path")/" "$model_cfg_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$model_cfg_root/$config_name.py"

# config_name="j6gen2_aug01_drivable01"
# echo "==> Generating dataset/model config pair: $config_name"
# cp "$base_cfg_py" "$cfg_py_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$cfg_py_root/$config_name.py"
# sed -i 's/#VERSION_NAMES#/"aug-0.1-drivable-under-0.1"/' "$cfg_py_root/$config_name.py"
# cp "$base_model_cfg_py" "$model_cfg_root/$config_name.py"
# sed -i "s/#DATA_ROOT#/$(escape_path "$output_path")/" "$model_cfg_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$model_cfg_root/$config_name.py"

# config_name="j6gen2_augtest05_drivable01"
# echo "==> Generating dataset/model config pair: $config_name"
# cp "$base_cfg_py" "$cfg_py_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$cfg_py_root/$config_name.py"
# sed -i 's/#VERSION_NAMES#/"augtest-0.5-drivable-under-0.1"/' "$cfg_py_root/$config_name.py"
# cp "$base_model_cfg_py" "$model_cfg_root/$config_name.py"
# sed -i "s/#DATA_ROOT#/$(escape_path "$output_path")/" "$model_cfg_root/$config_name.py"
# sed -i "s/#CONFIG_NAME#/$config_name/" "$model_cfg_root/$config_name.py"

config_name="j6gen2_augtest05"
echo "==> Generating dataset/model config pair: $config_name"
cp "$base_cfg_py" "$cfg_py_root/$config_name.py"
sed -i "s/#CONFIG_NAME#/$config_name/" "$cfg_py_root/$config_name.py"
sed -i 's/#VERSION_NAMES#/"augtest-0.5"/' "$cfg_py_root/$config_name.py"
cp "$base_model_cfg_py" "$model_cfg_root/$config_name.py"
sed -i "s/#DATA_ROOT#/$(escape_path "$output_path")/" "$model_cfg_root/$config_name.py"
sed -i "s/#CONFIG_NAME#/$config_name/" "$model_cfg_root/$config_name.py"

echo "==> Creating pickles for all generated dataset versions"
# for db in "j6gen2_$baseline_name" "j6gen2_aug01" "j6gen2_aug01_drivable01" "j6gen2_augtest05_drivable01" "j6gen2_augtest05"; do
for db in "j6gen2_augtest05"; do
  echo "    > Making pickles for $db"
  time docker run --rm --gpus '"device=0"' --shm-size=64g --name awml-pkl-$(date +%Y%m%d%H%M%S) \
    -p 6006:6006 \
    -v $PWD/:/workspace \
    -v "$output_path/:$output_path/" \
    -v "/mnt/qnapdata/internal/:/mnt/qnapdata/internal/:ro" \
    autoware-ml-ptv3 \
    bash -c "pip install 'polars>=0.20' && python \
      tools/detection3d/create_data_t4dataset.py \
      --root_path "$output_path" \
      --config '/workspace/autoware_ml/configs/segmentation3d/dataset/t4dataset/$db.py' \
      --version '$db' \
      --max_sweeps 1 \
      --out_dir '$output_path/info'"
done

echo "==> DONE!"