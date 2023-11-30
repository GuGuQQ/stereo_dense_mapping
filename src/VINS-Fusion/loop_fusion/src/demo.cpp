void image_callback2(const sensor_msgs::ImageConstPtr &image_msg2)
{
    //ROS_INFO("image_callback!");
    m_buf.lock();
    image_buf2.push(image_msg2);
    m_buf.unlock();

    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = image_msg2->header.stamp.toSec();
    else if (image_msg2->header.stamp.toSec() - last_image_time > 1.0 || image_msg2->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_msg2->header.stamp.toSec();
}

void point_callback2(const sensor_msgs::PointCloudConstPtr &point_msg2)
{
    //ROS_INFO("point_callback!");
    m_buf.lock();
    point_buf2.push(point_msg2);
    m_buf.unlock();

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
    pub_point_cloud.publish(point_cloud);
  //  pub_point_cloud_iris2.publish(point_cloud);
}

// only for visualization
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
    pub_margin_cloud.publish(point_cloud);
  //  pub_margin_cloud_iris2.publish(point_cloud);
}

void pose_callback2(const nav_msgs::Odometry::ConstPtr &pose_msg2)
{
    //ROS_INFO("pose_callback!");
    m_buf.lock();
    pose_buf2.push(pose_msg2);
    m_buf.unlock();
}

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
    pub_odometry_rect.publish(odometry);
   // pub_odometry_rect_iris2.publish(odometry);

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic2;
    vio_q_cam = vio_q * qic2;        

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg2->header);
   // cameraposevisual.publish_by(pub_camera_pose_visual_iris2, pose_msg2->header);


}

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

void process2()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr image_msg2 = NULL;
        sensor_msgs::PointCloudConstPtr point_msg2 = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg2 = NULL;

        // find out the messages with same time stamp
        m_buf.lock();
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
        m_buf.unlock();

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

                KeyFrame* keyframe = new KeyFrame(pose_msg2->header.stamp.toSec(), frame_index, T, R, image,
                                   point_3d, point_2d_uv, point_2d_normal, point_id, sequence, iris_flag);   
                m_process.lock();
                start_flag = 1;
                //posegraph.addKeyFrame(keyframe, 1);
                posegraph.addKFIntoVoc(keyframe);
                m_process.unlock();
                frame_index++;
                last_t = T;
            }
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

