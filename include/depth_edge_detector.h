#pragma once
#include <opencv2/core.hpp>

class DepthEdgeDetector {
public:
	DepthEdgeDetector() {}
	~DepthEdgeDetector() {}

public:
	// Find Edge in PointCloud Image ( Point Cloud가 OpenCV 3Channel로 저장되어 있는 형태 )
	cv::Mat findEdge(const cv::Mat & pc_image, cv::Mat& edge_map, float threshold, int radius);
};