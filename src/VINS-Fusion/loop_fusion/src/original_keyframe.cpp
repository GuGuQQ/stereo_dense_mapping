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

#include "original_keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
        computeORBBRIEFPoint();
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}


void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}
////////////////////////////By WJJ//计算ORB特征点、描述子orb_keypoints、orb_brief_descriptors////////////
void KeyFrame::computeORBBRIEFPoint()
{
     int nfeatures = 1000;
     int nlevels = 8;
     float fscaleFactor = 1.2;
     float fIniThFAST = 20;
     float fMinThFAST = 7;
     ORBextractor* pORBextractor;
     pORBextractor = new ORBextractor( nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST );
     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    (*pORBextractor)( image, mask, orb_keypoints, orb_brief_descriptors );
     chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
     chrono::duration<double> orbtime_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   //  cout << "计算ORB特征点描述子 = " << orbtime_used.count() << " seconds. " << endl;
}
///////////////////////////////////
void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
       // cv::Ptr<cv::FeatureDetector> detector=cv::ORB::create();
       // cv::Ptr<cv::DescriptorExtractor> descriptor =cv:: ORB::create();
       // std::vector<cv::KeyPoint> keypoints_orb;
        //cv::Mat descriptors_orb;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	const int fast_th = 20; // corner detector response threshold
/*
	if(1){
		cv::FAST(image, keypoints, fast_th, true);
                //detector->cv::Feature2D::detect(image, keypoints_orb);
              }
            
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
*/
	//cv::FAST(image, keypoints, fast_th, true);
	vector<cv::Point2f> tmp_pts;
	cv::goodFeaturesToTrack(image, tmp_pts, 1000, 0.01, 10);
	for(int i = 0; i < (int)tmp_pts.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = tmp_pts[i];
            keypoints.push_back(key);
	}
	extractor(image, keypoints, brief_descriptors);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        //detector->cv::Feature2D::detect(image, keypoints_orb);
       // descriptor->cv::Feature2D::compute(image, keypoints_orb, descriptors_orb);
      //  chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        chrono::duration<double> fasttime_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
      //  chrono::duration<double> orbtime_used = chrono::duration_cast<chrono::duration<double>>(t3 - t2);
       // cout << "计算Fast角点描述子 = " << fasttime_used.count() << " seconds. " << endl;
       // cout << "计算ORB特征点描述子 = " << orbtime_used.count() << " seconds. " << endl;
       // printf("得到ORB描述子");

	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}


void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	//for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	//printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic;
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}

// 利用投影法进行特征匹配
void KeyFrame::MatchByProject( const std::vector<cv::Point3f>& vp3d1, cv::Mat& desp1, 
    const vector<cv::Point2f>& vp2d2, cv::Mat& desp2, double& radian, 
    cv::Mat& K, cv::Mat& R, cv::Mat& t, std::vector<cv::DMatch>& matches ) {

    matches.clear();

    mTh_ = 50;
    mNNRatio_ = 0.6;
    // 将参考帧中的三维点投影到当前帧中查找合适的匹配点
    for (int i = 0; i < vp3d1.size(); ++i) {
        cv::Mat p3d = (cv::Mat_<double>(3,1) << vp3d1[i].x, vp3d1[i].y, vp3d1[i].z);
        cv::Mat p3d_trans = R*p3d + t;
        p3d_trans /= p3d_trans.at<double>(2,0);
        double u = K.at<double>(0,0)*p3d_trans.at<double>(0,0) + K.at<double>(0,2);
        double v = K.at<double>(1,1)*p3d_trans.at<double>(1,0) + K.at<double>(1,2);
        if (u < 0 || u > K.at<double>(0,2) || v < 0 || v > K.at<double>(1,2)) {
            continue;
        }
        // 在匹配半径中查找合适的匹配候选点
        std::vector<cv::Mat> desp_temp;
        std::vector<int> desp_index;
        for (int j = 0; j < vp2d2.size(); ++j) {
            cv::Point2f p2d = vp2d2[j];
            // u-radian < x < u+radian
            // v-radian < y < v+radian
            if ( (u-radian) < p2d.x && (u+radian) > p2d.x &&
                    (v-radian) < p2d.y && (v+radian) > p2d.y) {
                desp_temp.push_back(desp2.row(j));
                desp_index.push_back(j);
            }
        }

        // 在候选描述子中找到最合适的匹配点
        cv::Mat d1 = desp1.row(i);
        int min_dist = 256;
        int sec_min_dist = 256;
        int best_id = -1;
        for (int k = 0; k < desp_temp.size(); ++k) {
            cv::Mat d2 = desp2.row(desp_index[k]);
            int dist = ComputeMatchingScore(d1, d2);
            if (dist < min_dist) {
                sec_min_dist = min_dist;
                min_dist = dist;
                best_id = desp_index[k];
            }
            else if (dist < sec_min_dist) {
                sec_min_dist = dist;
            }
        }

        // 利用阈值条件筛选
        if (min_dist < mTh_) {
            if (min_dist < mNNRatio_*sec_min_dist) {
                cv::DMatch m1;
                m1.queryIdx = i;
                m1.trainIdx = best_id;
                m1.distance = min_dist;
                matches.push_back(m1);
            }
        }
    }

   // return matches;
}

// 计算两个描述子间的匹配分数
int KeyFrame::ComputeMatchingScore( cv::Mat& desp1, cv::Mat& desp2 ) {
    
    const int *p1 = desp1.ptr<int32_t>();
    const int *p2 = desp2.ptr<int32_t>();

    // 计算描述子匹配分数
    int dist = 0;

    // 位运算,目的是算出两个描述子之间有多少个不同的点
    for (int i = 0; i < 8; ++i, ++p1, ++p2) {
        unsigned int v = (*p1) ^ (*p2);
        v = v - ( (v >> 1) & 0x55555555 );
        v = ( v & 0x33333333 ) + ( (v >> 2) & 0x33333333 );
        dist += ( ( (v + (v >> 4)) & 0xF0F0F0F ) * 0x1010101 ) >> 24;
    }

    return dist;
}
//////////////solvePnPRansacMatch投影匹配法  By WJJ 3D_Points、desp_3d、2D_Points、desp_2d、Camera_K、D////////////////////
void KeyFrame::solvePnPRansacMatch(const vector<cv::Point2f> &matched_2d_old_orb,
                         const std::vector<cv::Point3f> &matched_3d_cur ,cv::Mat& cur_orbdesp, cv::Mat& old_orbdesp)
{
    cv::Mat D;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    //cv::Mat K = (cv::Mat_<double>(3, 3) << 435.2046959714599, 0, 367.4517211914062, 0, 435.2046959714599, 252.2008514404297, 0, 0, 1.0);
    // 运动估计
    cv::Mat rvec, tvec;
    cv::Mat inliers;
/*
    if (matched_3d_cur.size() > 10 && matched_2d_old_orb.size() > 10){
     if (CV_MAJOR_VERSION < 3)
         solvePnPRansac(matched_3d_cur, matched_2d_old_orb, K, D, rvec, tvec, true, 100, 10.0 / 460.0, 100, inliers);
     else
     {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d_cur, matched_2d_old_orb, K, D, rvec, tvec, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d_cur, matched_2d_old_orb, K, D, rvec, tvec, true, 100, 10.0 / 460.0, 0.99, inliers);
     }
    }
*/
    solvePnP(matched_3d_cur,matched_2d_old_orb, K, D, rvec, tvec);

    cv::Mat R, t;
    cv::Rodrigues(rvec, R);
    t = tvec.clone();
    std::vector<cv::DMatch> projMatch;
    if (matched_3d_cur.size() > 10) {
        double radian = 10.0;
        MatchByProject(matched_3d_cur, cur_orbdesp, matched_2d_old_orb, old_orbdesp, radian, K, R, t, projMatch);
    }
}
///////////////////////////////////////////////////////////////
// 利用暴力匹配法进行特征匹配
bool KeyFrame::MatchByBruteForce( cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches ) {
    
    matches.clear();

    // 保证描述子非空
    if (!(desp1.rows > 0 && desp2.rows > 0)) {
        return false;
    }

    std::vector<cv::DMatch> matches_new;
    cv::BFMatcher bf(cv::NORM_HAMMING);
    // 特征匹配
    bf.match(desp1, desp2, matches_new);

    // 统计所有匹配的最大最小距离
    double min_dist = 10000.0, max_dist = 0.;
    for (int i = 0; i < matches_new.size(); ++i) {
        double dist = matches_new[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    // 设置阈值并根据阈值条件找到合适的匹配
    for (int i = 0; i < matches_new.size(); ++i) {
        double dist = matches_new[i].distance;
        if (dist < std::max(min_dist*2, 30.0)) {
            matches.push_back(matches_new[i]);
        }
    }

    return matches.size() > 0 ? true : false;
}

void KeyFrame::KeyPointsToPoints(vector<cv::KeyPoint> &kpts, vector<cv::Point2f> &pts)
{
	for (int i = 0; i < kpts.size(); i++) {
		pts.push_back(kpts[i].pt);
	}
}

void KeyFrame::PointsToKeyPoints(vector<cv::Point2f> &pts, vector<cv::KeyPoint> &kpts)
{
	for (size_t i = 0; i < pts.size(); i++) {
		kpts.push_back(cv::KeyPoint(pts[i], 1.f));
	}
}


bool KeyFrame::findConnection(KeyFrame* old_kf)
{
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	//#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        vector<cv::Mat>hImgs;
	        cv::Mat old_img = old_kf->image;
	        hImgs.push_back(image);
	        hImgs.push_back(old_img);
	        hImgs.push_back(old_img);
	      //  cv::hconcat(image, old_img, gray_img);
	        cv::hconcat(hImgs,gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->orb_keypoints.size(); i++)
	        {
	            cv::Point2f orb_pt = old_kf->orb_keypoints[i].pt;
	            orb_pt.x = orb_pt.x + 2*COL;
	            cv::circle(loop_match_img, orb_pt, 5, cv::Scalar(0, 0, 255));
	        }
	        // cv::imshow("loop connection",loop_match_img);  
	        // cv::waitKey(10); 
/*
	        ostringstream path;
	        path <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
*/
	    }
	vector<cv::Point2f> matched_2d_cur_shimo;
	vector<cv::Point2f> matched_2d_old_orb;
	vector<cv::Point3f> matched_3d_cur;
	vector<cv::KeyPoint> keypoint_shimo;
        //_old_orb = old_kf->orb_keypoints;
        KeyPointsToPoints(old_kf->orb_keypoints ,matched_2d_old_orb);
      //  cout<<"matched_2d_old_orb size: "<<matched_2d_old_orb.size()<<endl;
	matched_3d_cur = point_3d;
       // cout<<"matched_3d_cur size: "<<matched_3d_cur.size()<<endl;
        matched_2d_cur_shimo = point_2d_uv;
       // PointsToKeyPoints(matched_2d_cur_shimo, keypoint_shimo);
      //  cout<<"keypoint_shimo size: "<<keypoint_shimo.size()<<endl;
        cv::Mat cur_orbdesp;
        cur_orbdesp = orb_brief_descriptors;
       // std::vector<cv::DMatch> projMatch;
        std::vector<cv::DMatch> bfMatch;
	vector<cv::KeyPoint> old_orb_keypoints = old_kf->orb_keypoints;
///暴力匹配////
        MatchByBruteForce(orb_brief_descriptors, old_kf->orb_brief_descriptors, bfMatch);
        cv::Mat BFOut;
	cv::Mat old_img = old_kf->image;
        cv::drawMatches(image, orb_keypoints, old_img, old_orb_keypoints, bfMatch, BFOut);
       // cv::imshow("BFMatch", BFOut);
        cv::waitKey(10);
  //根据bfMatch将特征点对齐,将坐标转换为float类型
    vector<cv::KeyPoint> R_keypoint01,R_keypoint02;
    for (size_t i=0;i<bfMatch.size();i++)   
    {
        R_keypoint01.push_back(orb_keypoints[bfMatch[i].queryIdx]);
        R_keypoint02.push_back(old_orb_keypoints[bfMatch[i].trainIdx]);
        //R_keypoint1存储img01中能与img02匹配的特征点，
        //bfMatch中存储了这些匹配点对的img01和img02的索引值
    }
    //坐标转换
    vector<cv::Point2f>p01,p02;
    for (size_t i=0;i<bfMatch.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }
    //利用基础矩阵剔除误匹配点
    vector<uchar> RansacStatus;
    cv::Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,cv::FM_RANSAC);
 
    vector<cv::KeyPoint> RR_keypoint01,RR_keypoint02;
    vector<cv::DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    int index=0;
    for (size_t i=0;i<bfMatch.size();i++)
    {
        if (RansacStatus[i]!=0)
        {
            RR_keypoint01.push_back(R_keypoint01[i]);
            RR_keypoint02.push_back(R_keypoint02[i]);
            bfMatch[i].queryIdx=index;
            bfMatch[i].trainIdx=index;
            RR_matches.push_back(bfMatch[i]);
            index++;
        }
    }
    cv::Mat img_RR_matches;
    cv::drawMatches(image,RR_keypoint01,old_img,RR_keypoint02,RR_matches,img_RR_matches);
    ostringstream path;
    path <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	     << index << "-"
	     << old_kf->index << "-" << "cur-keypoint.jpg";
	//     cv::imwrite( path.str().c_str(), img_RR_matches);
   // cv::imshow("GoodMatch",img_RR_matches);
   // cv::waitKey(10);
//投影法//
      //  solvePnPRansacMatch(matched_2d_old_orb,matched_3d_cur,cur_orbdesp,old_kf->orb_brief_descriptors);
/*
		if (DEBUG_IMAGE)    
	    {
	       // cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        //cv::hconcat(image, old_img, gray_img);
	       // cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
                if (projMatch.size() > 10) {
                   // cout << "Size of matching by projection is : " << projMatch.size() << endl;
                    cv::Mat projOut;
                    cv::drawMatches(image, keypoint_shimo, old_img, old_kf->orb_keypoints, projMatch, projOut);
                    cv::imshow("PNPMatching", projOut);
                    cv::waitKey(10);
                 }
             }
*/
	//#endif
	//printf("search by des\n");
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	//printf("search by des finish\n");

	//#if 0 
		if (DEBUG_IMAGE)
	    {
	      int gap = 10;
              cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
              cv::Mat gray_img, loop_match_img;
	      vector<cv::Mat>hImgs;
	        cv::Mat old_img = old_kf->image;
	   //  hImgs.push_back(old_img);
	   //  hImgs.push_back(gap_image);
	     hImgs.push_back(image);
	     hImgs.push_back(gap_image);
	     hImgs.push_back(old_img);
            //cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(hImgs,gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
/*
	        ostringstream path;
	        path <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                    << index << "-"
	                    << old_kf->index << "-" << "cur-keypoint.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
*/
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
///////////////////////////////////////////Old kf ORB/ ////////
/*
	        for(int i = 0; i< (int)old_kf->orb_keypoints.size(); i++)
	        {
	            cv::Point2f orb_pt = old_kf->orb_keypoints[i].pt;
	            orb_pt.x = orb_pt.x - (gap + COL);
	            cv::circle(loop_match_img, orb_pt, 5, cv::Scalar(0, 0, 255));
	        }
*/
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }
/*
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            orb_pt.x = orb_pt.x - (gap + COL);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 0, 255), 1, 8, 0);
	        }
*/

/*	        for(int i = 0; i< (int)old_kf->orb_keypoints.size(); i++)
	        {
	            cv::Point2f orb_pt = old_kf->orb_keypoints[i].pt;
	            orb_pt.x = orb_pt.x + 2*COL;
	            cv::circle(loop_match_img, orb_pt, 5, cv::Scalar(0, 0, 255));
	        }
*/
	        ostringstream path, path1, path2;
	        path <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	       // cv::imshow("First loop connection",loop_match_img);  
	       // cv::waitKey(10); 
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        
/*
	        path1 <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
*/
	        
	    }
//	#endif
	status.clear();
	/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
	    #if 1
	    	if (DEBUG_IMAGE)
	        {
	        	int gap = 10;
	        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
	            cv::Mat gray_img, loop_match_img;
	            cv::Mat old_img = old_kf->image;
	            cv::hconcat(image, gap_image, gap_image);
	            cv::hconcat(gap_image, old_img, gray_img);
	            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	            }
	            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
	            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

	            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            cv::vconcat(notation, loop_match_img, loop_match_img);

	            
	            ostringstream path;
	            path <<  "/home/wjj/catkin_workspace/src/VINS-Fusion/result/"
	                    << index << "-"
	                    << old_kf->index << "-" << "3pnp_match.jpg";
	            cv::imwrite( path.str().c_str(), loop_match_img);
	            
	            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	            {
	            	
	            	//cv::imshow("loop connection",loop_match_img);  
	            	//cv::waitKey(10);  
	            	
	            	cv::Mat thumbimage;
	            	cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
	    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);
	            }
	        }
	    #endif
	}

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
	    //printf("PNP relative\n");
	    //cout << "pnp relative_t " << relative_t.transpose() << endl;
	    //cout << "pnp relative_yaw " << relative_yaw << endl;
	    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	//cout << "pnp relative_t " << relative_t.transpose() << endl;
	    	//cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
	        return true;
	    }
	}
	//printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
	return false;
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


