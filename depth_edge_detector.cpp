#include "depth_edge_detector.h"

#include <glm/glm.hpp>

cv::Mat DepthEdgeDetector::findEdge(const cv::Mat& pc_image, cv::Mat& edge_map,float threshold, int radius) {

	std::vector<std::pair<int, int>> n = { { -1, -1 },{ -1, 0 },{ -1, 1 } ,{0, 1 } };
	int num_neighborhood_chek = n.size();

	int d_height = pc_image.rows;
	int d_width = pc_image.cols;
	float* pc_image_data = (float*)pc_image.data;

	uchar* edge_map_data = (uchar*)edge_map.data;
	
	int edge_val_max = 255;
#pragma omp parallel for
	for (int r = 1 * radius; r < d_height - 1 * radius; r++) { // Boundary Condition 을 위해 radius 만큼 떨어진 곳들만 search
		for (int c = 1 * radius; c < d_width - 1 * radius; c++) {
			glm::fvec3 X = glm::fvec3(
				pc_image_data[(r * d_width + c) * 3 + 0],
				pc_image_data[(r * d_width + c) * 3 + 1],
				pc_image_data[(r * d_width + c) * 3 + 2]);

			if (X.x == 0 && X.y == 0 && X.z == 0) continue;

			float max = -INFINITY;
			bool A_or_B_false = false;
			for (int k = 0; k < num_neighborhood_chek; k++) {
				bool A_false = false;
				bool B_false = false;
				glm::fvec3 A = glm::fvec3(
					pc_image_data[( (r + (n.at(k).first * radius)) * d_width + (c + (n.at(k).second * radius)) ) * 3 + 0],
					pc_image_data[( (r + (n.at(k).first * radius)) * d_width + (c + (n.at(k).second * radius)) ) * 3 + 1],
					pc_image_data[( (r + (n.at(k).first * radius)) * d_width + (c + (n.at(k).second * radius)) ) * 3 + 2]);
				glm::fvec3 B = glm::fvec3(
					pc_image_data[((r - (n.at(k).first * radius)) * d_width + (c - (n.at(k).second * radius))) * 3 + 0],
					pc_image_data[((r - (n.at(k).first * radius)) * d_width + (c - (n.at(k).second * radius))) * 3 + 1],
					pc_image_data[((r - (n.at(k).first * radius)) * d_width + (c - (n.at(k).second * radius))) * 3 + 2]);

				if (A.x == 0 && A.y == 0 && A.z == 0) A_false = true;
				if (B.x == 0 && B.y == 0 && B.z == 0) B_false = true;

				if (A_false && B_false) continue;
				else if ( A_false || B_false )  A_or_B_false = true;

				float amx = glm::length(A - X); // || A - X ||
				float bmx = glm::length(B - X); // || B - X ||
				float amb = std::max(glm::length(A - B), 0.000001f); // || A - B ||
				float val = (amx + bmx) / amb;
				max = std::max(val, max);
			}
			if (max > threshold) edge_map_data[r * d_width + c] = edge_val_max;
			else if (A_or_B_false) edge_map_data[r * d_width + c] = edge_val_max;
			else edge_map_data[r * d_width + c] = 0;
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