# I-DACS
- A framework for un-posed radiance field construction.
- Our experiments were all conducted on Ubuntu 16.04.
## Dependencies and configurations
- Clone the project to your own path.
  - `git clone git@github.com:z619850002/IDACS.git`
- Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). You can follow the official guidance.
- Other dependencies have been summarized in requirement.txt and can be installed as,
  - `pip install -r requirements.txt` 
- Download pretrained weights of [RAFT](https://github.com/princeton-vl/RAFT), most important raft-kitti.pth, and store them in path RAFT/models/

## Dataset
- We use the setting of [Nope-NeRF](https://github.com/ActiveVisionLab/nope-nerf) on Tanks and Temples with some modifications in the data structure. Your can download it with our link and extract it to your own disk.

## Usage
- Modify the dataroot of config files in config/data/ to your path, such as: 
    - `data_root: dataset/Tanks/`
- Choose a sequence and then run: 
    - `python train.py --render Sampling --data ${Seq}`

  Such as: 
    - `python train.py --render Sampling --data Ballroom`
  
  Both the rendering results of the test set and the tracking results will be stored in save/.
- The PSNR, SSIM and LPIPS of all testing frames will be used to name the rendering results.
- To evaluate the localization accuracy of the trajectory: 
    - `python compute_ate.py $gt_path$  $pose_path$`
  
  Such as: 
    - `python compute_ate.py dataset/Ballroom/  save/Ballroom/poses/`

  
