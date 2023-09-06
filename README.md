# IDACS
- A framework for un-posed radiance field construction.

## Dependencies and configurations
- Clone the project to your own path
  - `git clone git@github.com:z619850002/IDACS.git`
- Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- Other dependencies have been summarized in requirement.txt
  - `pip install -r requirements.txt`
- Our experiments were all conducted on Ubuntu 16.04. 
- Download pretrained weights of [RAFT](https://github.com/princeton-vl/RAFT), most important raft-kitti.pth, and store them in path RAFT/models/

## Data
- Tanks and Temples Dataset
    - We use the setting of [Nope-NeRF](https://github.com/ActiveVisionLab/nope-nerf) with some modifications in our experiments. Your can download it with our netdisk link.


## Usage
- Modify the dataroot of config files in config/data/ to your path, such as: 
    - `data_root: /media/kyrie/000A8561000BB32A/Tanks/`
- Choose a sequence and then run: 
    - `python train.py --render Sampling --data ${Seq}`

  Such as: 
    - `python train.py --render Sampling --data Ballroom`
  
  Then the rendering results of the test set will be stored in save/
- To evaluate the localization accuracy of the trajectory: 
    - `python compute_ate.py $gt_path$  $pose_path$`
  
  Such as: 
    - `python compute_ate.py estimated_poses/GT/ballroom_gt.pt  save/Ballroom/poses/`
- The gt poses of all sequences in the Tanks and Temples dataset we utilized  are stored in estimated_poses/GT 

  
