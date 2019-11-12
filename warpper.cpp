#include "warpper.h"

#include <iostream>

#include <glm/gtc/type_ptr.hpp>

#include <opencv2/calib3d.hpp>

using namespace ms;

Warpper::Warpper(
	const Intrinsic_ & color_intrinsic, 
	const Intrinsic_ & depth_intrinsic, 
	const glm::fmat4x4 & d2c_extrinsic){

	color_intrinsic_ = color_intrinsic;
	depth_intrinsic_ = depth_intrinsic;
	d2c_extrinsic_ = d2c_extrinsic * 1000.f;
}

Warpper::~Warpper(){
	for (auto& each : d2c_correspondence)
		each.clear();
	d2c_correspondence.clear();
}

cv::Mat Warpper::warpRGB_ColorToDepth(const cv::Mat& input, cv::Mat& output, int interpolation_mode) {
	if (!input.data) {
		std::cerr << "[Error] No Color Data!!!" << std::endl;
		exit(EXIT_FAILURE);
	}

	input_ = input;

	int depth_width  = depth_intrinsic_.width  / downscale_ ;
	int depth_height = depth_intrinsic_.height / downscale_;	

	uchar* output_data = (uchar*)output.data;
#pragma omp parallel for
	for (int r = 0; r < depth_height; r++) {
		for (int c = 0; c < depth_width; c++) {
			cv::Vec3b interpolated_rgb = getInterpolatedRGB_FromColor(c, r, interpolation_mode);
			output_data[(r*depth_width + c) * 3 + 0] = interpolated_rgb[0];
			output_data[(r*depth_width + c) * 3 + 1] = interpolated_rgb[1];
			output_data[(r*depth_width + c) * 3 + 2] = interpolated_rgb[2];
		}
	}
	return output;
}

cv::Mat Warpper::warpGray_DepthToColor(const cv::Mat & input, cv::Mat& output){
	if (!input.data) {
		std::cerr << "[Error] No Color Data!!!" << std::endl;
		exit(EXIT_FAILURE);
	}

	input_ = input;

	int color_width  = color_intrinsic_.width  / downscale_;
	int color_height = color_intrinsic_.height / downscale_;

	uchar* output_data = (uchar*)output.data;
#pragma omp parallel for
	for (int r = 0; r < color_height; r++) {
		for (int c = 0; c < color_width; c++) {
			unsigned char interpolated_gray = getInterpolatedGray_FromDepth(c, r, INTERPOLATION::bilinear);
			output_data[r*color_width + c] = interpolated_gray;
		}
	}
	return output;
}

// Get Value In Dpeth Coordinate From Color Coordinate
cv::Vec3b Warpper::getInterpolatedRGB_FromColor(int u, int v, int interpolation_mode) {

	int color_width  = color_intrinsic_.width  / downscale_;
	int color_height = color_intrinsic_.height / downscale_;

	glm::fvec3 uv_in_CCS = H_d2c_ * glm::fvec3(u, v, 1);
	uv_in_CCS /= uv_in_CCS[2];
	double x = uv_in_CCS[0];
	double y = uv_in_CCS[1];

	cv::Vec3b result;
	uchar* color_data = (uchar*)input_.data;
	int width = color_.cols;
	if (x >= 0 && x < (color_width - 1) && y < (color_height - 1) && y >= 0) {
		int floored_x = (int)(x);
		int ceiled_x = (int)(x + 1);
		int floored_y = (int)(y);
		int ceiled_y = (int)(y + 1);

		cv::Vec3b interpolated_up;
		interpolated_up[0] = color_data[(floored_y* width + ceiled_x) * 3 + 0] * (x - floored_x) + color_data[(floored_y* width + floored_x) * 3 + 0] * (1 - x + floored_x);
		interpolated_up[1] = color_data[(floored_y* width + ceiled_x) * 3 + 1] * (x - floored_x) + color_data[(floored_y* width + floored_x) * 3 + 1] * (1 - x + floored_x);
		interpolated_up[2] = color_data[(floored_y* width + ceiled_x) * 3 + 2] * (x - floored_x) + color_data[(floored_y* width + floored_x) * 3 + 2] * (1 - x + floored_x);
		cv::Vec3b interpolated_down;
		interpolated_down[0] = color_data[(ceiled_y* width + ceiled_x) * 3 + 0] * (x - floored_x) +  color_data[(ceiled_y* width + floored_x) * 3 + 0] * (1 - x + floored_x);
		interpolated_down[1] = color_data[(ceiled_y* width + ceiled_x) * 3 + 1] * (x - floored_x) +  color_data[(ceiled_y* width + floored_x) * 3 + 1] * (1 - x + floored_x);
		interpolated_down[2] = color_data[(ceiled_y* width + ceiled_x) * 3 + 2] * (x - floored_x) +  color_data[(ceiled_y* width + floored_x) * 3 + 2] * (1 - x + floored_x);

		result[0] = interpolated_down[0] * (y - floored_y) + interpolated_up[0] * (1 - y + floored_y);
		result[1] = interpolated_down[1] * (y - floored_y) + interpolated_up[1] * (1 - y + floored_y);
		result[2] = interpolated_down[2] * (y - floored_y) + interpolated_up[2] * (1 - y + floored_y);
	}
	return result;
}

uchar Warpper::getInterpolatedGray_FromDepth(int u, int v, int interpolation_mode) {
	int depth_width  = depth_intrinsic_.width  / downscale_;
	int depth_height = depth_intrinsic_.height / downscale_;

	glm::fvec3 uv_in_DCS = H_c2d_ * glm::fvec3(u, v, 1);
	uv_in_DCS /= uv_in_DCS[2];
	double x = uv_in_DCS[0];
	double y = uv_in_DCS[1];

	float result = 0;
	uchar* image_data = (uchar*)input_.data;

	if (x >= 0 && x < (depth_width - 1) && y < (depth_height - 1) && y >= 0) {
		int floored_x = (int)(x);
		int ceiled_x = (int)(x + 1);
		int floored_y = (int)(y);
		int ceiled_y = (int)(y + 1);

		float interpolated_up;
		interpolated_up = image_data[floored_y*depth_width + ceiled_x] * (x - floored_x) + image_data[floored_y*depth_width + floored_x] * (1 - x + floored_x);

		float interpolated_down;
		interpolated_down = image_data[ceiled_y*depth_width + ceiled_x] * (x - floored_x) + image_data[ceiled_y*depth_width + floored_x] * (1 - x + floored_x);

		result = interpolated_down * (y - floored_y) + interpolated_up * (1 - y + floored_y);
	}
	return (unsigned char)(result);
}

void Warpper::setHomography(const cv::Mat& color, const cv::Mat& point_cloud, int downscale) {
	color_ = color;
	point_cloud_ = point_cloud;
	downscale_ = downscale;
	
	calcCorrespondenDepthToColor();
	calcHomography();
}

void Warpper::calcCorrespondenDepthToColor() {
	int color_width = color_.cols;
	int color_height = color_.rows;
	int pc_width = point_cloud_.cols;
	int pc_height= point_cloud_.rows;

	d2c_correspondence = std::vector<std::vector<std::pair<float, float>>>(pc_height, std::vector<std::pair<float, float>>(pc_width, std::make_pair(-1, -1)));

	float* pc_data = (float*)point_cloud_.data;

#pragma omp parallel for
	for (int r = 0; r < pc_height; r++) {
		for (int c = 0; c < pc_width; c++) {
			float x = pc_data[(r * pc_width + c) * 3 + 0];
			float y = pc_data[(r * pc_width + c) * 3 + 1];
			float z = pc_data[(r * pc_width + c) * 3 + 2];

			glm::fvec4 xyz_in_ccs = d2c_extrinsic_ * glm::fvec4(x, y, z, 1);
			float u_in_ccs = (xyz_in_ccs.x / xyz_in_ccs.z) * (color_intrinsic_.fx / downscale_) + (color_intrinsic_.ppx / downscale_);
			float v_in_ccs = (xyz_in_ccs.y / xyz_in_ccs.z) * (color_intrinsic_.fy / downscale_) + (color_intrinsic_.ppy / downscale_);

			if (u_in_ccs >= 0 && u_in_ccs < color_width && v_in_ccs >= 0 && v_in_ccs < color_height)
				d2c_correspondence.at(r).at(c) = std::make_pair(v_in_ccs, u_in_ccs);
		}
	}
}

void Warpper::calcHomography() {
	int color_width = color_.cols;
	int color_height = color_.rows;
	int pc_width = point_cloud_.cols;
	int pc_height = point_cloud_.rows;

	std::vector<cv::Point2f> color_2d, depth_2d;
	for (int r = 0; r < pc_height; r+= (pc_height / 10)) {
		for (int c = 0; c < pc_width; c += (pc_width / 10)) {
			if (d2c_correspondence.at(r).at(c).first != -1 && d2c_correspondence.at(r).at(c).second != -1) {
				depth_2d.emplace_back(cv::Point2f(c, r)); // x, y
				color_2d.emplace_back(cv::Point2f(d2c_correspondence.at(r).at(c).second, d2c_correspondence.at(r).at(c).first)); // x, y
			}
		}
	}

	cv::Mat H_d2c = cv::findHomography(depth_2d, color_2d, cv::RANSAC);
	cv::Mat H_c2d = cv::findHomography(color_2d, depth_2d, cv::RANSAC);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			H_d2c_[j][i] = H_d2c.at<double>(i, j);
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			H_c2d_[j][i] = H_c2d.at<double>(i, j);
		}
	}
}