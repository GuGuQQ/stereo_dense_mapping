/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#define SKIP_FIRST_CNT 10
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */  
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */ 

using namespace std;

queue<sensor_msgs::ImageConstPtr> image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;

queue<sensor_msgs::ImageConstPtr> image_buf2;
queue<sensor_msgs::PointCloudConstPtr> point_buf2;
queue<nav_msgs::Odometry::ConstPtr> pose_buf2;
queue<Eigen::Vector3d> odometry_buf2;


std::mutex m_buf;
std::mutex m_process;
std::mutex m_buf2;
std::mutex m_process2;

int frame_index  = -1;
int sequence = 1;
int sequence2 = 1;
int iris_flag ;
int iris_flag2 ;
PoseGraph posegraph;
int skip_first_cnt = 0;
int skip_first_cnt2 = 0;
int SKIP_CNT;
int skip_cnt = 0;
int skip_cnt2 = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;

Eigen::Vector3d tic2;
Eigen::Matrix3d qic2;

ros::Publisher pub_match_img;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_odometry_rect;

ros::Publisher pub_match_img_iris2;
ros::Publisher pub_camera_pose_visual_iris2;
ros::Publisher pub_odometry_rect_iris2;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
Eigen::Vector3d last_t2(-100, -100, -100);
double last_image_time = -1;
double last_image_time2 = -1;

ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_point_cloud_iris2, pub_margin_cloud_iris2;
void new_sequence()
{
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 5)
    {
        ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        ROS_BREAK();
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    while(!odometry_buf.empty())
        odometry_buf.pop();
    m_buf.unlock();
}
/////////////////////////////////////////////new sequence////////////////////
void new_sequence2()
{
    printf("new sequence\n");
    sequence2++;
    printf("sequence2 cnt %d \n", sequence2);
    if (sequence2 > 5)
    {
        ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
        ROS_BREAK();
    }
    posegraph.posegraph_visualization->reset();
    posegraph.publish();
    m_buf2.lock();
    while(!image_buf2.empty())
        image_buf2.pop();
    while(!point_buf2.empty())
        point_buf2.pop();
    while(!pose_buf2.empty())
        pose_buf2.pop();
    while(!odometry_buf2.empty())
        odometry_buf2.pop();
    m_buf2.unlock();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_msg->header.stamp.toSec();
}
/////////////////////////////////////////////image_callback2/////////////////
void image_callback2(const sensor_msgs::ImageConstPtr &image_msg2)
{
    //ROS_INFO("image_callback!");
    m_buf2.lock();
    image_buf2.push(image_msg2);
    m_buf2.unlock();

    // detect unstable camera stream
    if (last_image_time2 == -1)
        last_image_time2 = image_msg2->header.stamp.toSec();
    else if (image_msg2->header.stamp.toSec() - last_image_time2 > 1.0 || image_msg2->header.stamp.toSec() < last_image_time2)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence2();
    }
    last_image_time2 = image_msg2->header.stamp.toSec();
}

//////////////////////////////////////////////
void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    //ROS_INFO("point_callback!");
    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();

    // for visualization
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);
  //  pub_point_cloud_iris2.publish(point_cloud);
}
////////////////////////////////point_callback2//////////////////////////

void point_callback2(const sensor_msgs::PointCloudConstPtr &point_msg2)
{
    //ROS_INFO("point_callback!");
    m_buf2.lock();
    point_buf2.push(point_msg2);
    m_buf2.unlock();

    // for visualization
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg2->header;
    for (unsigned int i = 0; i < point_msg2->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg2->points[i].x;
        p_3d.y = point_msg2->points[i].y;
        p_3d.z = point_msg2->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    //pub_point_cloud.publish(point_cloud);
  //  pub_point_cloud_iris2.publish(point_cloud);
}

// only for visualization
void margin_point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg->header;
    for (unsigned int i = 0; i < point_msg->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg->points[i].x;
        p_3d.y = point_msg->points[i].y;
        p_3d.z = point_msg->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
    pub_margin_cloud.publish(point_cloud);
  //  pub_margin_cloud_iris2.publish(point_cloud);
}
////////////////////////////////////////////margin_point_callback2/////////////////////
void margin_point_callback2(const sensor_msgs::PointCloudConstPtr &point_msg2)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = point_msg2->header;
    for (unsigned int i = 0; i < point_msg2->points.size(); i++)
    {
        cv::Point3f p_3d;
        p_3d.x = point_msg2->points[i].x;
        p_3d.y = point_msg2->points[i].y;
        p_3d.z = point_msg2->points[i].z;
        Eigen::Vector3d tmp = posegraph.r_drift * Eigen::Vector3d(p_3d.x, p_3d.y, p_3d.z) + posegraph.t_drift;
        geometry_msgs::Point32 p;
        p.x = tmp(0);
        p.y = tmp(1);
        p.z = tmp(2);
        point_cloud.points.push_back(p);
    }
   // pub_margin_cloud.publish(point_cloud);
  //  pub_margin_cloud_iris2.publish(point_cloud);
}
///////////////////////////////////////////////////////////finish///////////////

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}
////////////////////////////////////////////pose_callback2////////////////////
void pose_callback2(const nav_msgs::Odometry::ConstPtr &pose_msg2)
{
    //ROS_INFO("pose_callback!");
    m_buf2.lock();
    pose_buf2.push(pose_msg2);
    m_buf2.unlock();
}
///////////////////////////////

void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg->pose.pose.orientation.w;
    vio_q.x() = pose_msg->pose.pose.orientation.x;
    vio_q.y() = pose_msg->pose.pose.orientation.y;
    vio_q.z() = pose_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    nav_msgs::Odometry odometry;
    odometry.header = pose_msg->header;
    odometry.header.frame_id = "map";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
    pub_odometry_rect.publish(odometry);
   // pub_odometry_rect_iris2.publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
   // cameraposevisual.publish_by(pub_camera_pose_visual_iris2, pose_msg->header);

}
/////////////////////////////////////////////////vio_callback2/////////////////////////
void vio_callback2(const nav_msgs::Odometry::ConstPtr &pose_msg2)
{
    //ROS_INFO("vio_callback!");
    Vector3d vio_t(pose_msg2->pose.pose.position.x, pose_msg2->pose.pose.position.y, pose_msg2->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = pose_msg2->pose.pose.orientation.w;
    vio_q.x() = pose_msg2->pose.pose.orientation.x;
    vio_q.y() = pose_msg2->pose.pose.orientation.y;
    vio_q.z() = pose_msg2->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    nav_msgs::Odometry odometry;
    odometry.header = pose_msg2->header;
    odometry.header.frame_id = "map";
    odometry.pose.pose.position.x = vio_t.x();
    odometry.pose.pose.position.y = vio_t.y();
    odometry.pose.pose.position.z = vio_t.z();
    odometry.pose.pose.orientation.x = vio_q.x();
    odometry.pose.pose.orientation.y = vio_q.y();
    odometry.pose.pose.orientation.z = vio_q.z();
    odometry.pose.pose.orientation.w = vio_q.w();
   // pub_odometry_rect.publish(odometry);
   // pub_odometry_rect_iris2.publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic2;
    vio_q_cam = vio_q * qic2;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
 //   cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg2->header);
   // cameraposevisual.publish_by(pub_camera_pose_visual_iris2, pose_msg2->header);
}
//////////////////////////////////////////////////////////////////////////////////
void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}
/////////////////////////////////////////////////////////////////////extrinsic_callback2/////////////////////////////
void extrinsic_callback2(const nav_msgs::Odometry::ConstPtr &pose_msg2)
{
    m_process.lock();
    tic2 = Vector3d(pose_msg2->pose.pose.position.x,
                   pose_msg2->pose.pose.position.y,
                   pose_msg2->pose.pose.position.z);
    qic2 = Quaterniond(pose_msg2->pose.pose.orientation.w,
                      pose_msg2->pose.pose.orientation.x,
                      pose_msg2->pose.pose.orientation.y,
                      pose_msg2->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}
//////////////////////////////////////////////////////////////
void process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
/*
        sensor_msgs::ImageConstPtr image_msg2 = NULL;
        sensor_msgs::PointCloudConstPtr point_msg2 = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg2 = NULL;
*/
        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() 
                && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();
/*
        m_buf2.lock();
        if(!image_buf2.empty() && !point_buf2.empty() && !pose_buf2.empty())
        {
            if (image_buf2.front()->header.stamp.toSec() > pose_buf2.front()->header.stamp.toSec())
            {
                pose_buf2.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf2.front()->header.stamp.toSec() > point_buf2.front()->header.stamp.toSec())
            {
                point_buf2.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf2.back()->header.stamp.toSec() >= pose_buf2.front()->header.stamp.toSec() 
                && point_buf2.back()->header.stamp.toSec() >= pose_buf2.front()->header.stamp.toSec())
            {
                pose_msg2 = pose_buf2.front();
                pose_buf2.pop();
                while (!pose_buf2.empty())
                    pose_buf2.pop();
                while (image_buf2.front()->header.stamp.toSec() < pose_msg2->header.stamp.toSec())
                    image_buf2.pop();
                image_msg2 = image_buf2.front();
                image_buf2.pop();

                while (point_buf2.front()->header.stamp.toSec() < pose_msg2->header.stamp.toSec())
                    point_buf2.pop();
                point_msg2 = point_buf2.front();
                point_buf2.pop();
            }
        }
        m_buf2.unlock();

        if (pose_msg2 != NULL)
        {
            // skip fisrt few
            if (skip_first_cnt2 < SKIP_FIRST_CNT)
            {
                skip_first_cnt2++;
                continue;
            }

            if (skip_cnt2 < SKIP_CNT)
            {
                skip_cnt2++;
                continue;
            }
            else
            {
                skip_cnt2 = 0;
            }

            cv_bridge::CvImageConstPtr ptr2;
            if (image_msg2->encoding == "8UC1")
            {
                sensor_msgs::Image img2;
                img2.header = image_msg2->header;
                img2.height = image_msg2->height;
                img2.width = image_msg2->width;
                img2.is_bigendian = image_msg2->is_bigendian;
                img2.step = image_msg2->step;
                img2.data = image_msg2->data;
                img2.encoding = "mono8";
                ptr2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr2 = cv_bridge::toCvCopy(image_msg2, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image2 = ptr2->image;
            // build keyframe
            Vector3d T2 = Vector3d(pose_msg2->pose.pose.position.x,
                                  pose_msg2->pose.pose.position.y,
                                  pose_msg2->pose.pose.position.z);
            Matrix3d R2 = Quaterniond(pose_msg2->pose.pose.orientation.w,
                                     pose_msg2->pose.pose.orientation.x,
                                     pose_msg2->pose.pose.orientation.y,
                                     pose_msg2->pose.pose.orientation.z).toRotationMatrix();
            if((T2 - last_t2).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d2; 
                vector<cv::Point2f> point_2d_uv2; 
                vector<cv::Point2f> point_2d_normal2;
                vector<double> point_id2;

                for (unsigned int i = 0; i < point_msg2->points.size(); i++)
                {
                    cv::Point3f p_3d2;
                    p_3d2.x = point_msg2->points[i].x;
                    p_3d2.y = point_msg2->points[i].y;
                    p_3d2.z = point_msg2->points[i].z;
                    point_3d2.push_back(p_3d2);

                    cv::Point2f p_2d_uv2, p_2d_normal2;
                    double p_id2;
                    p_2d_normal2.x = point_msg2->channels[i].values[0];
                    p_2d_normal2.y = point_msg2->channels[i].values[1];
                    p_2d_uv2.x = point_msg2->channels[i].values[2];
                    p_2d_uv2.y = point_msg2->channels[i].values[3];
                    p_id2 = point_msg2->channels[i].values[4];
                    point_2d_normal2.push_back(p_2d_normal2);
                    point_2d_uv2.push_back(p_2d_uv2);
                    point_id2.push_back(p_id2);
                }
                iris_flag2=2;
                KeyFrame* keyframe = new KeyFrame(pose_msg2->header.stamp.toSec(), frame_index, T2, R2, image2,
                                   point_3d2, point_2d_uv2, point_2d_normal2, point_id2, sequence2, iris_flag2);   
               // m_process2.lock();
               // start_flag = 1;
                //posegraph.addKeyFrame(keyframe, 1);
               // m_process2.unlock();
                frame_index++;
                posegraph.addKFIntoVoc(keyframe, frame_index);
		//cout<<GREEN<<"2号添加第"<<frame_index<<"帧关键帧至数据库"<<WHITE<<endl;
                last_t2 = T2;
            }
        }
*/
        if (pose_msg != NULL)
        {
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);
                }
                iris_flag=1;
                KeyFrame* keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence, iris_flag);   
                m_process.lock();
                start_flag = 1;
                m_process.unlock();

                frame_index++;
                posegraph.addKFIntoVocself(keyframe, frame_index);
               // cout<<GREEN<<"1号添加第"<<frame_index<<"帧关键帧至数据库"<<WHITE<<endl;
                posegraph.addKeyFrame(keyframe, frame_index, 1 );

                last_t = T;
            }
        }
        std::chrono::milliseconds dura(15);
        std::this_thread::sleep_for(dura);
    }
}
////////////////////////////////////////////////////////////////////process2/////////////////////////////////////////
///1.KeyFrame* keyframe  2.posegraph.addKFIntoVoc
void process2()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg2 = NULL;
        sensor_msgs::PointCloudConstPtr point_msg2 = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg2 = NULL;

        // find out the messages with same time stamp
        m_buf2.lock();
        if(!image_buf2.empty() && !point_buf2.empty() && !pose_buf2.empty())
        {
            if (image_buf2.front()->header.stamp.toSec() > pose_buf2.front()->header.stamp.toSec())
            {
                pose_buf2.pop();
                printf("throw pose at beginning\n");
            }
            else if (image_buf2.front()->header.stamp.toSec() > point_buf2.front()->header.stamp.toSec())
            {
                point_buf2.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf2.back()->header.stamp.toSec() >= pose_buf2.front()->header.stamp.toSec() 
                && point_buf2.back()->header.stamp.toSec() >= pose_buf2.front()->header.stamp.toSec())
            {
                pose_msg2 = pose_buf2.front();
                pose_buf2.pop();
                while (!pose_buf2.empty())
                    pose_buf2.pop();
                while (image_buf2.front()->header.stamp.toSec() < pose_msg2->header.stamp.toSec())
                    image_buf2.pop();
                image_msg2 = image_buf2.front();
                image_buf2.pop();

                while (point_buf2.front()->header.stamp.toSec() < pose_msg2->header.stamp.toSec())
                    point_buf2.pop();
                point_msg2 = point_buf2.front();
                point_buf2.pop();
            }
        }
        m_buf2.unlock();

        if (pose_msg2 != NULL)
        {
            // skip fisrt few
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            if (skip_cnt < SKIP_CNT)
            {
                skip_cnt++;
                continue;
            }
            else
            {
                skip_cnt = 0;
            }

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg2->encoding == "8UC1")
            {
                sensor_msgs::Image img;
                img.header = image_msg2->header;
                img.height = image_msg2->height;
                img.width = image_msg2->width;
                img.is_bigendian = image_msg2->is_bigendian;
                img.step = image_msg2->step;
                img.data = image_msg2->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else
                ptr = cv_bridge::toCvCopy(image_msg2, sensor_msgs::image_encodings::MONO8);
            
            cv::Mat image = ptr->image;
            // build keyframe
            Vector3d T = Vector3d(pose_msg2->pose.pose.position.x,
                                  pose_msg2->pose.pose.position.y,
                                  pose_msg2->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg2->pose.pose.orientation.w,
                                     pose_msg2->pose.pose.orientation.x,
                                     pose_msg2->pose.pose.orientation.y,
                                     pose_msg2->pose.pose.orientation.z).toRotationMatrix();
            if((T - last_t).norm() > SKIP_DIS)
            {
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg2->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg2->points[i].x;
                    p_3d.y = point_msg2->points[i].y;
                    p_3d.z = point_msg2->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg2->channels[i].values[0];
                    p_2d_normal.y = point_msg2->channels[i].values[1];
                    p_2d_uv.x = point_msg2->channels[i].values[2];
                    p_2d_uv.y = point_msg2->channels[i].values[3];
                    p_id = point_msg2->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);
                }
                iris_flag=2;
                KeyFrame* keyframe = new KeyFrame(pose_msg2->header.stamp.toSec(), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence2, iris_flag);   
                m_process.lock();
                start_flag = 1;
                //posegraph.addKeyFrame(keyframe, 1);
                m_process.unlock();

                frame_index++;
                posegraph.addKFIntoVoc(keyframe, frame_index);
		//cout<<GREEN<<"2号添加第"<<frame_index<<"帧关键帧至数据库"<<WHITE<<endl;

                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

//////////////////////////////////////////Finish//////////////////////////////////////
void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_process.lock();
            posegraph.savePoseGraph();
            m_process.unlock();
            printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
            printf("program shutting down...\n");
            ros::shutdown();
        }
        if (c == 'n')
            new_sequence();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_fusion");
    ros::NodeHandle n("~");
    posegraph.registerPub(n);
    
    VISUALIZATION_SHIFT_X = 0;
    VISUALIZATION_SHIFT_Y = 0;
    SKIP_CNT = 0;
    SKIP_DIS = 0;

    if(argc != 2)
    {
        printf("please intput: rosrun loop_fusion loop_fusion_node [config file] \n"
               "for example: rosrun loop_fusion loop_fusion_node "
               "/home/tony-ws1/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 0;
    }
    
    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);

    std::string IMAGE_TOPIC;
    std::string IMAGE_TOPIC_iris2;
    int LOAD_PREVIOUS_POSE_GRAPH;

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    std::string pkg_path = ros::package::getPath("loop_fusion");
    string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
    //string vocabulary_file = pkg_path + "/../support_files/ORBvoc.bin";
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph.loadVocabulary(vocabulary_file);

    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    printf("cam calib path: %s\n", cam0Path.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

    fsSettings["image0_topic"] >> IMAGE_TOPIC; 
    fsSettings["image1_topic"] >> IMAGE_TOPIC_iris2;        
    fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
    fsSettings["output_path"] >> VINS_RESULT_PATH;
    fsSettings["save_image"] >> DEBUG_IMAGE;

    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
    VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop_iris_1.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    int USE_IMU = fsSettings["imu"];
    posegraph.setIMUFlag(USE_IMU);
    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
        printf("load pose graph\n");
        m_process.lock();
        posegraph.loadPoseGraph();
        m_process.unlock();
        printf("load pose graph finish\n");
        load_flag = 1;
    }
    else
    {
        printf("no previous pose graph\n");
        load_flag = 1;
    }

    ros::Subscriber sub_vio = n.subscribe("/iris_1/vins_estimator/odometry", 2000, vio_callback);
    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);
    ros::Subscriber sub_pose = n.subscribe("/iris_1/vins_estimator/keyframe_pose", 2000, pose_callback);
    ros::Subscriber sub_extrinsic = n.subscribe("/iris_1/vins_estimator/extrinsic", 2000, extrinsic_callback);
    ros::Subscriber sub_point = n.subscribe("/iris_1/vins_estimator/keyframe_point", 2000, point_callback);
    ros::Subscriber sub_margin_point = n.subscribe("/iris_1/vins_estimator/margin_cloud", 2000, margin_point_callback);

    pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud_loop_rect", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud_loop_rect", 1000);
    pub_odometry_rect = n.advertise<nav_msgs::Odometry>("odometry_rect", 1000);

    ros::Subscriber sub_vio_iris2 = n.subscribe("/iris_2/vins_estimator/odometry", 2000, vio_callback2);
    ros::Subscriber sub_image_iris2 = n.subscribe(IMAGE_TOPIC_iris2, 2000, image_callback2);
    ros::Subscriber sub_pose_iris2 = n.subscribe("/iris_2/vins_estimator/keyframe_pose", 2000, pose_callback2);
    ros::Subscriber sub_extrinsic_iris2 = n.subscribe("/iris_2/vins_estimator/extrinsic", 2000, extrinsic_callback2);
    ros::Subscriber sub_point_iris2 = n.subscribe("/iris_2/vins_estimator/keyframe_point", 2000, point_callback2);
    ros::Subscriber sub_margin_point_iris2 = n.subscribe("/iris_2/vins_estimator/margin_cloud", 2000, margin_point_callback2);
/*
    pub_match_img_iris2 = n.advertise<sensor_msgs::Image>("match_image_iris2", 1000);
    pub_camera_pose_visual_iris2 = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual_iris2", 1000);
    pub_point_cloud_iris2 = n.advertise<sensor_msgs::PointCloud>("point_cloud_loop_rect_iris2", 1000);
    pub_margin_cloud_iris2 = n.advertise<sensor_msgs::PointCloud>("margin_cloud_loop_rect_iris2", 1000);
    pub_odometry_rect_iris2 = n.advertise<nav_msgs::Odometry>("odometry_rect_iris2", 1000);
*/
    std::thread measurement_process2;
    std::thread measurement_process;
   // std::thread t1(process2);
   // t1.join();
   // std::thread t1(process); 
   // process2();

   // t2.join();
   // std::thread keyboard_command_process;

    measurement_process2 = std::thread(process2);
    measurement_process = std::thread(process);
   // process2();
    cout<<"main函数"<<endl;
   // keyboard_command_process = std::thread(command);
    
    ros::spin();

    return 0;
}
