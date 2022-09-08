# Lane Detection and turn prediction
## Overview
An algorithm for lane detection and turn prediction for a self-driving
vehicles.


# [Report](report.pdf)

## To run the code    
```bash
cd ...<repository>/code/
python3 lane_detection.py --FilePath ../data_files/challenge.mp4 --visualize True --record False
```
- FilePath -  Video file path. *Default :- ' ../data_files/challenge.mp4 '*
- visualize - Shows visualization . *Default :- 'True'*
- record - Records video (../turn_predict.mp4). *Default :- 'False'*

## Result
<p align="center">
<img src="./results/turn_predict.gif"/>
</p>


