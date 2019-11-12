#include "depth_edge_detector.h"

#include <glm/glm.hpp>

cv::Mat DepthEdgeDetector::findEdge(const cv::Mat& pc_image, cv::Mat& edge_map,float threshold, int radius) {

	std::vector<std::pair<int, int>> n = { { -1, -1 },{ -1, 0 },{ -1, 1 } ,{0, 1 } };
	int num_neighborhood_chek = n.size();

	int d_height = pc_image.rows;
	int d_width = pc_image.cols;
	float* pc_image_data = (float*)pc_image.data;

	//cv::Mat edge_map = cv::Mat::zeros(d_height, d_width, CV_8UC1);

	memset(edge_map.data, 0, d_height*d_width * sizeof(uchar));
	uchar* edge_map_data = (uchar*)edge_map.data;
	int edge_val = 255;

#pragma omp parallel for
	for (int r = 1 * radius; r < d_height - 1 * radius; r++) { // Boundary Condition 을 위해 radius 만큼 떨어진 곳들만 search
		for (int c = 1 * radius; c < d_width - 1 * radius; c++) {

			//glm::fvec3 X = glm::fvec3(
			//	pc_image.at<cv::Vec3f>(r, c)[0],
			//	pc_image.at<cv::Vec3f>(r, c)[1],
			//	pc_image.at<cv::Vec3f>(r, c)[2]);

			glm::fvec3 X = glm::fvec3(
				pc_image_data[(r * d_width + c) * 3 + 0],
				pc_image_data[(r * d_width + c) * 3 + 1],
				pc_image_data[(r * d_width + c) * 3 + 2]);

			if (X.x == 0 || X.y == 0 || X.z == 0) continue;
			//if (!pc_image_data[((r + radius) * d_width + (c + radius)) * 3 + 2] ||
			//	!pc_image_data[((r + radius) * d_width + (c - radius)) * 3 + 2] ||
			//	!pc_image_data[((r + -radius) * d_width + (c + radius)) * 3 + 2] ||
			//	!pc_image_data[((r + -radius) * d_width + (c -radius)) * 3 + 2] ||
			//	!pc_image_data[(r * d_width + c) * 3 + 2]) {
			//	continue;
			//}

			float max = -INFINITY;
			int tau = 0;
			//for (int t = 1; t < radius; t++) {
				for (int k = 0; k < num_neighborhood_chek; k++) {
					//glm::fvec3 A = glm::fvec3(
					//	pc_image.at<cv::Vec3f>( r + (n.at(k).first * t), c + (n.at(k).second * t) )[0],
					//	pc_image.at<cv::Vec3f>( r + (n.at(k).first * t), c + (n.at(k).second * t) )[1],
					//	pc_image.at<cv::Vec3f>( r + (n.at(k).first * t), c + (n.at(k).second * t) )[2]);
					//glm::fvec3 B = glm::fvec3(
					//	pc_image.at<cv::Vec3f>(r - (n.at(k).first * t), c - (n.at(k).second * t))[0],
					//	pc_image.at<cv::Vec3f>(r - (n.at(k).first * t), c - (n.at(k).second * t))[1],
					//	pc_image.at<cv::Vec3f>(r - (n.at(k).first * t), c - (n.at(k).second * t))[2]);
		
					glm::fvec3 A = glm::fvec3(
						pc_image_data[( (r + (n.at(k).first * radius)) * d_width + (c + (n.at(k).second * radius)) ) * 3 + 0],
						pc_image_data[( (r + (n.at(k).first * radius)) * d_width + (c + (n.at(k).second * radius)) ) * 3 + 1],
						pc_image_data[( (r + (n.at(k).first * radius)) * d_width + (c + (n.at(k).second * radius)) ) * 3 + 2]);
					glm::fvec3 B = glm::fvec3(
						pc_image_data[((r - (n.at(k).first * radius)) * d_width + (c - (n.at(k).second * radius))) * 3 + 0],
						pc_image_data[((r - (n.at(k).first * radius)) * d_width + (c - (n.at(k).second * radius))) * 3 + 1],
						pc_image_data[((r - (n.at(k).first * radius)) * d_width + (c - (n.at(k).second * radius))) * 3 + 2]);

					float amx = glm::length(A - X); // || A - X ||
					float bmx = glm::length(B - X); // || B - X ||
					float amb = std::max(glm::length(A - B), 0.000001f); // || A - B ||

					float val = (amx + bmx) / amb;

					max = std::max(val, max);

					float apb = glm::length(A + B); // || A + B ||
					float x = glm::length(X);	   //   || X ||

					if ((apb / 2) < x) // Convex or Concave
						tau++;
					else
						tau--;
				}
				if (max > threshold) {
					//if (tau < 0) 
					//	edge_map_data[r * d_width + c] = edge_val;
					//else 
					//	edge_map_data[r * d_width + c] = edge_val / 2;
					edge_map_data[r * d_width + c] = edge_val;
				}
				else {
					edge_map_data[r * d_width + c] = 0;
				}
			//}
		}
	}
	return edge_map;
}

//cv::Mat DepthEdgeDetector::findEdge(const pcl::PointCloud<pcl::PointXYZ>& pc, float threshold, int radius) {
//
//	std::vector<std::pair<int, int>> n = { { -1, -1 },{ 0, -1 },{ -1, 1 } ,{ -1, 0 } };
//
//	int d_height = pc.height;
//	int d_width = pc.width;
//
//	cv::Mat edge_map = cv::Mat::zeros(d_height, d_width, CV_8UC1);
//	int edge_val = 255;
//#pragma omp parallel for
//	for (int r = 1 * radius; r < d_height - 1 * radius; r++) {
//		for (int c = 1 * radius; c < d_width - 1 * radius; c++) {
//
//			glm::fvec3 X = glm::fvec3(
//				pc.at(c, r).x,
//				pc.at(c, r).y,
//				pc.at(c, r).z);
//
//			//if (X.x == 0 || X.y == 0 || X.z == 0) continue;
//
//			float max = -INFINITY;
//			int tau = 0;
//			for (int t = 1; t < radius; t++) {
//				for (int k = 0; k < 4; k++) {
//
//					glm::fvec3 A = glm::fvec3(
//						pc.at(c + (n.at(k).first * t), r + (n.at(k).second * t)).x,
//						pc.at(c + (n.at(k).first * t), r + (n.at(k).second * t)).y,
//						pc.at(c + (n.at(k).first * t), r + (n.at(k).second * t)).z);
//					glm::fvec3 B = glm::fvec3(
//						pc.at(c - (n.at(k).first * t), r - (n.at(k).second * t)).x,
//						pc.at(c - (n.at(k).first * t), r - (n.at(k).second * t)).y,
//						pc.at(c - (n.at(k).first * t), r - (n.at(k).second * t)).z);
//					float amx = MinusABS(A, X); // || A - X ||
//					float bmx = MinusABS(B, X); // || B - X ||
//					float amb = MinusABS(A, B); // || A - B ||
//
//					float val = (amx + bmx) / amb;
//
//					max = val > max ? val : max;
//
//					float apb = PlusABS(A, B); // || A + B ||
//					float x = SelfABS(X);	   //   || X ||
//
//					if ((apb / 2) < x) // Convex or Concave
//						tau++;
//					else
//						tau--;
//				}
//				if (max > threshold) {
//					if (tau < 0)
//						edge_map.at<uchar>(r, c) = edge_val;
//					else
//						edge_map.at<uchar>(r, c) = edge_val / 2;
//				}
//				else
//					edge_map.at<uchar>(r, c) = 0;
//			}
//		}
//	}
//	return edge_map;
//}