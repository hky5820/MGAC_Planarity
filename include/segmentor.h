#pragma once
#ifdef DLL_EXPORT
#define MYDLL __declspec(dllexport)
#else
#define MYDLL __declspec(dllimport)
#endif

#include <opencv2/core.hpp>

#include <glm/gtc/type_ptr.hpp>

#include "common.h"

class DepthEdgeDetector;
class Filter;
class MorphSnake;
class Warpper;

namespace ms {
	class /*MYDLL*/ Segmentor {

	public:
		Segmentor(
			const Intrinsic_& color_intrinsic,
			const Intrinsic_& depth_intrinsic,
			const glm::fmat4x4& d2c_extrinsic);
		~Segmentor();
		
	public:
		cv::Mat doSegmentation(
			const cv::Mat & color, const cv::Mat & depth,
			const DepthEdgeParam & de_param, const CannyParam & cn_param,
			const MorphSnakeParam & ms_param, const InitLevelSetParam& ls_param,
			int downscale, int mask_in_depth_or_color,
			const VisualizationParam& vs_param,
			const EdgeSelectionParam& es_param);
		
	private:
		Warpper* warpper_ = nullptr;
		DepthEdgeDetector* d_edge_detector_ = nullptr;
		Filter* filter_ = nullptr;
		MorphSnake* ms_ = nullptr;

		Intrinsic_ depth_intrinsic_;

		int h_size = 0;
		float h_sum = 0;
		//int warp_size = 0;
		//float warp_sum = 0;
		//int findEdge_size = 0;
		//float findEdge_sum = 0;
		//int iem_size = 0;
		//float iem_sum = 0;
		//int ms_size = 0;
		//float ms_sum = 0;

	};
};