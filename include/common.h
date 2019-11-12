#pragma once
#include <opencv2/core.hpp>

#include <glm/glm.hpp>

namespace ms {
	enum INTERPOLATION {
		bilinear = 0
	};


	enum CHANNEL {
		RED = 0,
		GREEN,
		BLUE,
		GRAY
	};

	enum MASK_AT {
		COLOR = 0,
		DEPTH
	};

	struct DepthEdgeParam {
		DepthEdgeParam() {};

		DepthEdgeParam(
			float threshold_,
			int radius_) :
			threshold(threshold_),
			radius(radius_) {}

		// [Coplanarity Check] : || A - X || + || B - X || ) / || A - B ||  값의 threshold,
		// threshold가 높을수록, 각이 크게 꺾이는 line위에 놓여져 있는 point들만 검출되게 됨
		// range : 1~2 정도
		float threshold;
		int radius; // Coplanarity Check 할 때, neighborhood의 범위
	};

	struct MorphSnakeParam {
		MorphSnakeParam() {};
		MorphSnakeParam(
			double sigma_,
			int channel_,
			int iteration_,
			int smoothing_,
			int ballon_) :
			sigma(sigma_),
			channel(channel_),
			iteration(iteration_),
			smoothing(smoothing_),
			ballon(ballon_) {}
		double sigma;  // Gaussian에 사용되는 sigma
		int channel;   // RGB에서 Gray로 변경시, 사용될 채널 종류(first,second,third) 혹은 OpenCV cvtColor 함수 ( GRAY )
		int iteration;
		int smoothing; // smoothing 횟수
		int ballon;    // 1 or -1 ( 팽창 or 축소 )
	};

	struct CannyParam {
		CannyParam() {};
		CannyParam(
			int low_threshold_,
			int high_threshold_,
			bool L2gradient_) :
			low_threshold(low_threshold_),
			high_threshold(high_threshold_),
			L2gradient(L2gradient_) {}

		int low_threshold;
		int high_threshold;
		bool L2gradient;
		// a flag, indicating whether a more accurate L2 norm =sqrt((dI/dx)2+(dI/dy)2) should be used to calculate the image gradient magnitude ( L2gradient=true ),
		// or whether the default L1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
	};

	struct InitLevelSetParam {
		InitLevelSetParam() {};
		InitLevelSetParam(
			int center_row_,
			int center_col_,
			int radius_) :
			center_row(center_row_),
			center_col(center_col_),
			radius(radius_) {}

		int center_row;
		int center_col;
		int radius;
	};

	struct VisualizationParam {
		VisualizationParam(bool depth_edge_on_, bool canny_edge_on_, bool warpping_on_) :
			depth_edge_on(depth_edge_on_), canny_edge_on(canny_edge_on_), warpping_on(warpping_on_) {}
		bool depth_edge_on;
		bool canny_edge_on;
		bool warpping_on;
	};

	struct Intrinsic_ {
		Intrinsic_() {};
		Intrinsic_(
			float fx_,
			float fy_,
			float ppx_,
			float ppy_,
			int width_,
			int height_) :
			fx(fx_),
			fy(fy_),
			ppx(ppx_),
			ppy(ppy_),
			width(width_),
			height(height_) {}
		float fx;
		float fy;
		float ppx;
		float ppy;
		int width;
		int height;
	};
};