# Gua
实现流程：

1.VINS-FUSION发布的关键帧位姿/vins_estimator/keyframe_pose";       
      https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git      
2.使用Elas包计算双目相机深度发布关键帧点云/point_cloud;       
       https://github.com/jeffdelmerico/cyphy-elas-ros.git    
3.使用建图包pointcloud_mapping订阅关键帧点云和位姿进行稠密化建图.

仿真环境使用的是XTDrone中的outdoor4        
XTDrone语雀:https://www.yuque.com/xtdrone/manual_cn
