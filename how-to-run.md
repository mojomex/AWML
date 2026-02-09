
```bash
# Launch docker
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v /mnt/qnapdata/internal/:/mnt/qnapdata/internal/:ro autoware-ml-ptv3

# In docker: option 1 (single GPU)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/detection3d/train.py projects/CenterPoint/configs/t4dataset/Centerpoint/concerto_centerpoint_121m_j6gen2_base_amp.py

# In docker: option 2 (multi GPU)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash tools/detection3d/dist_script.sh projects/CenterPoint/configs/t4dataset/Centerpoint/concerto_centerpoint_121m_j6gen2_base_amp.py 2 train
```
