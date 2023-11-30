#!/bin/bash
#cd /home/wjj/stereo_sim/src
#{
#gnome-terminal -t "Stereo_camera" -x bash -c "roslaunch camera_split camera_with_calibration.launch;exec bash"
#}&
#sleep 2s

cd /home/wjj/catkin_workspace/src
{
gnome-terminal -t "UGV_0&UGV_1" -x bash -c "roslaunch vins multi_ugv_slam.launch;exec bash"
}&
sleep 2s

cd /home/wjj/catkin_workspace/src
{
gnome-terminal -t "RVIZ" -x bash -c "roslaunch vins multi_vins_rviz.launch;exec bash"
}&
sleep 2s

#cd /home/wjj/stereo_sim/src
#{
#gnome-terminal -t "UGV_0" -x bash -c "rosrun ORB_SLAM2 first_car ~/stereo_sim/src/ORB_SLAM2/Vocabulary/ORBvoc.txt /home/#wjj/stereo_sim/src/ORB_SLAM2/Examples/Stereo/stereo.yaml true;exec bash"
#}&
#sleep 2s

cd /home/wjj/stereo_sim/src
{
gnome-terminal -t "Elas_ros" -x bash -c "roslaunch elas_ros multi_ugv_elas.launch;exec bash"
}&
sleep 1s

cd /home/wjj/stereo_sim/src
{
gnome-terminal -t "Mapping" -x bash -c "roslaunch pointcloud_mapping multi_ugv.launch;exec bash"
}&
sleep 2s

cd /home/wjj/stereo_sim/src
{
gnome-terminal -t "cmd_vel" -x bash -c "rqt;exec bash"
}&
sleep 1s
#rosrun loop_fusion loop_fusion_node_ugv /home/wjj/catkin_workspace/src/VINS-Fusion/config/xtdrone_sitl/multi_ugv_stereo_imu_config.yaml 

