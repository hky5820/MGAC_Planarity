#pragma once
#include <opencv2/core.hpp>

class Filter;

class MorphSnake {
public:
	MorphSnake();
	~MorphSnake() {}

public:
	cv::Mat morphological_geodesic_active_contour(const cv::Mat & inv_edge_map, const cv::Mat & merged_edge, const cv::Mat & init_ls, double threshold, int iterations, int smoothing, int ballon);
	cv::Mat morphological_geodesic_active_contour(
		const cv::Mat & inv_edge_map, 
		const cv::Mat & canny,
		const cv::Mat & init_ls,
		int iterations,
		int smoothing,
		int ballon);

private:
	Filter *filter = nullptr;
};

