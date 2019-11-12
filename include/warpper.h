#pragma once
#include <opencv2/core.hpp>

#include <glm/glm.hpp>

#include "common.h"

class Warpper {
public:
	Warpper(
		const ms::Intrinsic_& color_intrinsic,
		const ms::Intrinsic_& depth_intrinsic,
		const glm::fmat4x4& d2c_extrinsic);
	~Warpper();

public:
	void setHomography(const cv::Mat & color, const cv::Mat & depth, int downscale);
	
	cv::Mat warpRGB_ColorToDepth(const cv::Mat& input, cv::Mat& output, int interpolation_mode);
	cv::Mat warpGray_DepthToColor(const cv::Mat & input, cv::Mat & output);
	
	glm::fmat3x3 getD2CHomography() { return H_d2c_; };
	glm::fmat3x3 getC2DHomography() { return H_c2d_; };

private:
	void calcCorrespondenDepthToColor();
	void calcHomography();
	
	cv::Vec3b getInterpolatedRGB_FromColor(int u, int v, int interpolation_mode);
	uchar getInterpolatedGray_FromDepth(int u, int v, int interpolation_mode);

private:
	glm::fmat3x3 H_d2c_;
	glm::fmat3x3 H_c2d_;

	cv::Mat color_;
	cv::Mat depth_;
	cv::Mat input_;

	ms::Intrinsic_ color_intrinsic_;
	ms::Intrinsic_ depth_intrinsic_;
	glm::fmat4x4 d2c_extrinsic_;

	std::vector<std::vector<std::pair<float, float>>> d2c_correspondence;

	int downscale_;
};