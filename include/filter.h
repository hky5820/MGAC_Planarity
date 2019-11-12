#pragma once
#include <vector>

#include <opencv2/core.hpp>

#include <common.h>

class Filter {
public:
	Filter();
	~Filter() {}

public:
	cv::Mat inverse_edge_map(const cv::Mat & d_edge, const cv::Mat & canny, double sigma, int k_size, const ms::EdgeSelectionParam& es_param);
	cv::Mat inverse_edge_map(const cv::Mat & gray, double alpha, double sigma, int k_size);

	cv::Mat make_init_ls(const std::pair<int, int>& img_shape, const std::pair<int, int>& circle_center, unsigned char radius);

	void gradient(const cv::Mat & img, cv::Mat& gx, cv::Mat& gy);
	void gradient_uchar(const cv::Mat & img, cv::Mat & gx, cv::Mat & gy);
	void gradient_short(const cv::Mat & img, cv::Mat & gx, cv::Mat & gy);

	cv::Mat smoothing(const cv::Mat & img, std::vector<cv::Mat>& temps);
	cv::Mat smoothing(const cv::Mat& img, std::vector<cv::Mat>& temps, cv::Mat& t, cv::Mat& output);

	cv::Mat canny(const cv::Mat& img, int low_threshold, int high_threshold, bool L2gradient);
	cv::Mat RGB2GRAY_16(const cv::Mat & img, int channel);

	std::vector<cv::Mat> get_structures() { return structures; }

private:
	std::vector<cv::Mat> structures;
	bool is_inf_sup_first = true;
};