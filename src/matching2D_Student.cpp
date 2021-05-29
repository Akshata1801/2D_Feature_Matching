#include <numeric>
#include "matching2D.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
ptInfo matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    ptInfo ret_val;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = (descriptorType.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross - check = "<<crossCheck<<endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
        }

        if(descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef, CV_32F);
        }
        
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "Flann based matching "<<endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "(NN) with matches n = " << matches.size()<<" in time t = " << 1000 * t / 1.0 << " ms" << endl;
        ret_val.elaspsedtime_ms = 1000 * t / 1.0;
        ret_val.numPoints = matches.size();
    
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knnmatches;

        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnmatches, 2); // Finds the best match for each descriptor in desc1
        
        std::vector<cv::DMatch> good_matches;
        const float ratio_thresh = 0.8f;
        for(size_t i=0;i<knnmatches.size();i++)
        {
            if(knnmatches[i][0].distance < (ratio_thresh*knnmatches[i][1].distance))
            {
                good_matches.push_back(knnmatches[i][0]);
            }
        }
        
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "(KNN + DescRatio of 0.8) with matches n = " << good_matches.size()<<" in time t = " << 1000 * t / 1.0 << " ms" << endl;
        ret_val.elaspsedtime_ms = 1000 * t / 1.0;
        ret_val.numPoints = good_matches.size();
    
    }

    return ret_val;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
ptInfo descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    ptInfo ret_val;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();

    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        cout << "Enter a valid descriptor "<<endl;
    }
    

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    ret_val.elaspsedtime_ms = 1000 * t / 1.0;
    ret_val.numPoints = keypoints.size();

    return ret_val;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
ptInfo detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    ptInfo ret_val;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    ret_val.elaspsedtime_ms = 1000 * t / 1.0;
    ret_val.numPoints = keypoints.size();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return ret_val;
}

// Detect keypoints in image using HARRIS detector
ptInfo detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    cout << "in HArris detection "<<endl;   
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    int thresh = 200;
    int max_thresh = 255;

    int minResponse = 100;

    ptInfo ret_val;

    cout << "starting harris detection"<<endl;

    double t = (double)cv::getTickCount();
    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1 );
    cv::cornerHarris( img, dst, blockSize, apertureSize, k , cv::BORDER_DEFAULT);
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    cout << "performed Harris dtecetion "<<endl;

    //  for( int i = 0; i < dst_norm.rows ; i++ )
    // {
    //     for( int j = 0; j < dst_norm.cols; j++ )
    //     {
    //         if( (int) dst_norm.at<float>(i,j) > thresh )
    //         {
    //             cv::circle( dst_norm_scaled, cv::Point(j,i), 5,  cv::Scalar(0), 2, 8, 0 );
    //         }
    //     }
    // }

    //vector<cv::KeyPoint> keypoints;
    double maxOverlap = 0.0f;

    for(size_t i=0; i < dst_norm.rows;i++)
    {
        for(size_t j=0;j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i,j);

            if(response > minResponse)
            {
                // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(j,i);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                newKeyPoint.class_id = 0;

                //perform non-maximum suppression (NMS) in local neighbourhood around new keypoint

                bool bOverlap = false;
                for(auto it = keypoints.begin(); it!=keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if(kptOverlap > maxOverlap)
                    {
                        bOverlap = true;

                        if(newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;   // replace old keypoint with new keypoint
                            break; 
                        }
                    }


                }

                // cout << " done non max suppression "<<endl;

                if(!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);  // store only new keypoint in dynamic list
                }
            }
        }
    }


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HARRIS detection with n = "<<keypoints.size()<<" and time " << 1000 * t / 1.0 << " ms" << endl;
    ret_val.elaspsedtime_ms = 1000 * t / 1.0;
    ret_val.numPoints = keypoints.size();

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "HARRIS Corner Detector Results";
        cv::namedWindow(windowName, 7);
        imshow( windowName, visImage );
        cv::waitKey(0);
    }

    return ret_val;


}


// Detect keypoints in image using FAST, BRISK, ORB, AKAZE, SIFT  detector
ptInfo detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string typeDetector, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector ;
    string windowName;

    ptInfo ret_val;

    if(typeDetector.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
        windowName = "SIFT Detector Results";
    }
    else if(typeDetector.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        windowName = "Fast Detector type Results";
    }
    else if(typeDetector.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
        windowName = "BRISK detector Results";
    }
    else if(typeDetector.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
        windowName = "AKAZE detector Results";
    }
    else if(typeDetector.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
        windowName = "ORB detector Results";
    }
    else
    {
        cout << "Enter a proper detector name"<<endl;
    }

    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << typeDetector<<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    ret_val.elaspsedtime_ms = 1000 * t / 1.0;
    ret_val.numPoints = keypoints.size();
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return ret_val;

    
}


