#include "filter.h"

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

Filter::Filter() {
	structures.resize(4);
	for (int i = 0; i < 4; i++) structures[i] = cv::Mat::zeros(3, 3, CV_8UC1);
	uchar* s0_data = (uchar*)structures[0].data;
	uchar* s1_data = (uchar*)structures[1].data;
	uchar* s2_data = (uchar*)structures[2].data;
	uchar* s3_data = (uchar*)structures[3].data;

	for (int k = 0; k < 3; k++) {
		s0_data[k * 3 + k] = 1;
		s2_data[k * 3 + 2 - k] = 1;
		s1_data[1 * 3 + k] = 1;
		s3_data[k * 3 + 1] = 1;
	}
}

cv::Mat Filter::inverse_edge_map(
	const cv::Mat & gray,
	double alpha,
	double sigma,
	int k_size/*window size*/) {

	int rows = gray.rows;
	int cols = gray.cols;

	cv::Mat g_img;
	gray.convertTo(g_img, CV_64FC1, 1. / SHRT_MAX);
	
	cv::Mat blur_gray;
	cv::GaussianBlur(g_img, blur_gray, cv::Size(k_size, k_size), sigma);

	cv::Mat gx = cv::Mat::zeros(blur_gray.size(), CV_64FC1);
	cv::Mat gy = cv::Mat::zeros(blur_gray.size(), CV_64FC1);
	gradient(blur_gray, gx, gy);

	cv::Mat output = cv::Mat::zeros(rows, cols, CV_64FC1);
	double* output_data = (double*)output.data;
	double* gx_data = (double*)gx.data;
	double* gy_data = (double*)gy.data;

#pragma omp parallel for
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double gx_val = gx_data[r*cols + c];
			double gy_val = gy_data[r*cols + c];

			double val = sqrtl( gx_val * gx_val + gy_val * gy_val);
			output_data[r*cols + c] = 1. / sqrtl(1. + alpha * val);
		}
	}
	return output;
}

cv::Mat Filter::inverse_edge_map(
	const cv::Mat & d_edge, 
	const cv::Mat& canny,
	double sigma, 
	int k_size,
	const ms::EdgeSelectionParam& es_param ){

	int rows = d_edge.rows;
	int cols = d_edge.cols;
	
	// Depth Edge Filtering
	cv::Mat de_blur;
	cv::blur(d_edge, de_blur, cv::Size(k_size, k_size));
	
	cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);
	uchar* output_data = (uchar*)output.data;
	uchar* de_data = (uchar*)de_blur.data;
	uchar* canny_data = (uchar*)canny.data;

#pragma omp parallel for
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if(canny_data[r * cols + c]*es_param.use_depth_edge || de_data[r * cols + c]*es_param.use_canny_edge)
				output_data[r * cols + c] = 0;
			else 
				output_data[r * cols + c] = 255;
		}
	}

	return output;
}

cv::Mat Filter::make_init_ls(const std::pair<int, int>& img_shape, const std::pair<int, int>& circle_center, unsigned char radius){

	int rows = img_shape.first;
	int cols = img_shape.second;

	cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);
	uchar* output_data = (uchar*)output.data;
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double val = radius - sqrtl(powl(i - circle_center.first, 2.) + powl(j - circle_center.second, 2.));
			if (val > 0)
				output_data[i * cols + j] = 255;
		}
	}
	return output;
}

void Filter::gradient(const cv::Mat & img, cv::Mat& gx, cv::Mat& gy){
	int rows = img.rows, cols = img.cols;

	double* data = (double*)img.data;
	double* gx_data = (double*)gx.data;
	double* gy_data = (double*)gy.data;

#pragma omp parallel for
	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			gx_data[(r * cols) + c] = ( data[ (r * cols) + c + 1] - data[ (r * cols) + c - 1] ) / 2.;
			gy_data[(r * cols) + c] = ( data[ (r + 1) * cols + c] - data[ (r - 1) * cols + c] ) / 2.;
		}
	}
}

void Filter::gradient_uchar(const cv::Mat & img, cv::Mat& gx, cv::Mat& gy) {
	int rows = img.rows, cols = img.cols;

	uchar* data = (uchar*)img.data;
	uchar* gx_data = (uchar*)gx.data;
	uchar* gy_data = (uchar*)gy.data;

#pragma omp parallel for
	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			gx_data[(r * cols) + c] = (data[(r * cols) + c + 1] - data[(r * cols) + c - 1]) / 2.;
			gy_data[(r * cols) + c] = (data[(r + 1) * cols + c] - data[(r - 1) * cols + c]) / 2.;
		}
	}
}


void Filter::gradient_short(const cv::Mat & img, cv::Mat& gx, cv::Mat& gy) {
	int rows = img.rows, cols = img.cols;

	short* data = (short*)img.data;
	short* gx_data = (short*)gx.data;
	short* gy_data = (short*)gy.data;
#pragma omp parallel for
	for (int r = 1; r < rows - 1; r++) {
		for (int c = 1; c < cols - 1; c++) {
			gx_data[(r * cols) + c] = (data[(r * cols) + c + 1] - data[(r * cols) + c - 1]) / 2.;
			gy_data[(r * cols) + c] = (data[(r + 1) * cols + c] - data[(r - 1) * cols + c]) / 2.;
		}
	}
}


cv::Mat Filter::smoothing(const cv::Mat & img, std::vector<cv::Mat>& temps) {
	int rows = img.rows, cols = img.cols;
	cv::Mat t = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);
	uchar* t_data = (uchar*)t.data;
	uchar* output_data = (uchar*)output.data;

	if (is_inf_sup_first) {
		for (int i = 0; i < 4; i++) cv::dilate(img, temps[i], structures[i]);

		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						&& t_data1[r*cols + c] != 0
						&& t_data2[r*cols + c] != 0
						&& t_data3[r*cols + c] != 0)
						t_data[r*cols + c] = 255;

				}
			}
		}
		for (int i = 0; i < 4; i++) cv::erode(t, temps[i], structures[i]);

		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						|| t_data1[r*cols + c] != 0
						|| t_data2[r*cols + c] != 0
						|| t_data3[r*cols + c] != 0)
						output_data[r*cols + c] = 255;
				}
			}
		}
	}
	else {
		for (int i = 0; i < 4; i++) cv::erode(img, temps[i], structures[i]);

		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						|| t_data1[r*cols + c] != 0
						|| t_data2[r*cols + c] != 0
						|| t_data3[r*cols + c] != 0)
						t_data[r*cols + c] = 255;
				}
			}
		}
		for (int i = 0; i < 4; i++) cv::dilate(t, temps[i], structures[i]);

		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						&& t_data1[r*cols + c] != 0
						&& t_data2[r*cols + c] != 0
						&& t_data3[r*cols + c] != 0)
						output_data[r*cols + c] = 255;
				}
			}
		}
	}
	is_inf_sup_first = !is_inf_sup_first;
	return output;
}

cv::Mat Filter::smoothing(const cv::Mat & img, std::vector<cv::Mat>& temps, cv::Mat& t, cv::Mat& output){
	int rows = img.rows, cols = img.cols;

	uchar* t_data = (uchar*)t.data;
	uchar* output_data = (uchar*)output.data;

	if (is_inf_sup_first) {
		for (int i = 0; i < 4; i++) cv::dilate(img, temps[i], structures[i]);
		
		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						&& t_data1[r*cols + c] != 0
						&& t_data2[r*cols + c] != 0
						&& t_data3[r*cols + c] != 0)
						t_data[r*cols + c] = 255;
				}
			}
		}
		for (int i = 0; i < 4; i++) cv::erode(t, temps[i], structures[i]);
		
		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						|| t_data1[r*cols + c] != 0
						|| t_data2[r*cols + c] != 0
						|| t_data3[r*cols + c] != 0)
						output_data[r*cols + c] = 255;
				}
			}
		}
	}
	else {
		for (int i = 0; i < 4; i++) cv::erode(img, temps[i], structures[i]);
		
		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						|| t_data1[r*cols + c] != 0
						|| t_data2[r*cols + c] != 0
						|| t_data3[r*cols + c] != 0)
						t_data[r*cols + c] = 255;
				}
			}
		}
		for (int i = 0; i < 4; i++) cv::dilate(t, temps[i], structures[i]);
		
		{
			uchar* t_data0 = (uchar*)temps[0].data;
			uchar* t_data1 = (uchar*)temps[1].data;
			uchar* t_data2 = (uchar*)temps[2].data;
			uchar* t_data3 = (uchar*)temps[3].data;
#pragma omp parallel for
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (t_data0[r*cols + c] != 0
						&& t_data1[r*cols + c] != 0
						&& t_data2[r*cols + c] != 0
						&& t_data3[r*cols + c] != 0)
						output_data[r*cols + c] = 255;
				}
			}
		}
	}
	is_inf_sup_first = !is_inf_sup_first;

	return output;
}

cv::Mat Filter::canny(const cv::Mat & gray, int low_threshold, int high_threshold, bool L2gradient){
	int rows = gray.rows;
	int cols = gray.cols;

	// Gray Image Filtering
	cv::Mat gray_blur;
	cv::GaussianBlur(gray, gray_blur, cv::Size(3, 3), 3);

	// Finite Difference of Blurred Color Image For Canny
	cv::Mat gx = cv::Mat::zeros(rows, cols, CV_16SC1);
	cv::Mat gy = cv::Mat::zeros(rows, cols, CV_16SC1);
	gradient_short(gray_blur, gx, gy);

	// Get Canny From Gx, Gy
	cv::Mat canny;
	cv::Canny(gx, gy, canny, low_threshold, high_threshold, L2gradient);
	//cv::dilate(canny, canny, cv::Mat::ones(3, 3, CV_8UC1));

	return canny;
}

// Rgb To Gray In 16 bit
cv::Mat Filter::RGB2GRAY_16(const cv::Mat & img, int channel) {
	cv::Mat bgr[3]; cv::split(img, bgr);
	cv::Mat gray;
	if (channel < 3) {
		gray = bgr[channel];
	}
	else {
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	}

	gray.convertTo(gray, CV_16SC1, SHRT_MAX / 255.);

	return gray;
}