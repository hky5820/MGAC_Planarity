#include "morphsnake.h"

#include <opencv2/imgproc.hpp> // morphological operation
#include <opencv2/highgui.hpp>

#include "filter.h"

MorphSnake::MorphSnake() : filter(new Filter()) {}

cv::Mat MorphSnake::morphological_geodesic_active_contour(
	const cv::Mat & inv_edge_map,
	const cv::Mat& canny,
	const cv::Mat & init_ls,
	int iterations, 
	int smoothing,
	int ballon){

	uchar* inv_edge_map_data = (uchar*)inv_edge_map.data;
	uchar* c_data = (uchar*)canny.data;

	int rows = inv_edge_map.rows, 
		cols = inv_edge_map.cols;
	
	cv::Mat threshold_mask_balloon = cv::Mat::zeros(rows, cols, CV_8UC1);
	uchar* tmb_data = (uchar*)threshold_mask_balloon.data;
	int t_ballon = ballon < 0 ? -ballon : ballon;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			if (inv_edge_map_data[r * cols +c])
				tmb_data[r * cols + c] = 255;
		}
	}

	//cv::imshow("threshold_mask_balloon", threshold_mask_balloon);

	cv::Mat gx = cv::Mat::zeros(rows, cols, CV_8UC1),
			gy = cv::Mat::zeros(rows, cols, CV_8UC1);
	filter->gradient_uchar(inv_edge_map, gx, gy);
	uchar* gx_data = (uchar*)gx.data;
	uchar* gy_data = (uchar*)gy.data;

	cv::Mat u = init_ls;
	cv::Mat aux;

	cv::Mat structure = cv::Mat::ones(3, 3, CV_8UC1);
	
	cv::Mat dgx = cv::Mat::zeros(rows, cols, CV_8UC1),
		    dgy = cv::Mat::zeros(rows, cols, CV_8UC1);

	std::vector<cv::Mat> temps(4);
	for (int i = 0; i < 4; i++) 
		temps[i] = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (int c_itr = 0; c_itr < iterations; c_itr++) {
		if (ballon > 0) 
			cv::dilate(u, aux, structure);
		else if (ballon < 0) 
			cv::erode(u, aux, structure);
		
		uchar* u_data = (uchar*)u.data;
		uchar* aux_data = (uchar*)aux.data;
		if (ballon != 0) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (tmb_data[r * cols + c]) 
						u_data[r * cols + c] = aux_data[r * cols + c];
					else {
						//u_data[r * cols + c] = 0;
					}
				}
			}
		}
		filter->gradient_uchar(u, dgx, dgy);
		uchar* dgx_data = (uchar*)dgx.data;
		uchar* dgy_data = (uchar*)dgy.data;
		
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				double val =
					gx_data[r*cols + c] * dgx_data[r*cols + c]
					+ gy_data[r*cols + c] * dgy_data[r*cols + c];
				if(c_data[r*cols+c] == 0){
					if (val > 0) {
						u_data[r * cols + c] = 255;
					}
					else if (val < 0) {
						u_data[r * cols + c] = 0;
					}
				}
			}
		}
		for (int i = 0; i < smoothing; i++) 
			u = filter->smoothing(u, temps);
	}
	return u;
}