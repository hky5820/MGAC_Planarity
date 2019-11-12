#include <iostream>

#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_types.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <segmentor.h>

typedef std::chrono::high_resolution_clock::time_point c_time;

cv::Mat	overlay(const cv::Mat &img1, float w1, const cv::Mat &img2, float w2) {
	cv::Mat ol = cv::Mat::zeros(img1.size(), CV_8UC3);
	uchar* ol_data = (uchar*)ol.data;
	uchar* img1_data = (uchar*)img1.data;
	uchar* img2_data = (uchar*)img2.data;

	uchar chans = 1 + (img2.type() >> CV_CN_SHIFT);

	int width = ol.cols;
	int height = ol.rows;

	if (chans == 1) {
		for (int r = 0; r < height; r++) {
			for (int c = 0; c < width; c++) {
				ol_data[(r*width + c) * 3 + 0] = (int)(img1_data[(r*width + c) * 3 + 0] * w1 + img2_data[r*width + c] * w2);
				ol_data[(r*width + c) * 3 + 1] = (int)(img1_data[(r*width + c) * 3 + 1] * w1 + img2_data[r*width + c] * w2);
				ol_data[(r*width + c) * 3 + 2] = (int)(img1_data[(r*width + c) * 3 + 2] * w1 + img2_data[r*width + c] * w2);
			}
		}
	}
	if (chans == 3) {
		for (int r = 0; r < height; r++) {
			for (int c = 0; c < width; c++) {
				ol_data[(r*width + c) * 3 + 0] = (int)(img1_data[(r*width + c) * 3 + 0] * w1 + img2_data[(r*width + c) * 3 + 0] * w2);
				ol_data[(r*width + c) * 3 + 1] = (int)(img1_data[(r*width + c) * 3 + 1] * w1 + img2_data[(r*width + c) * 3 + 1] * w2);
				ol_data[(r*width + c) * 3 + 2] = (int)(img1_data[(r*width + c) * 3 + 2] * w1 + img2_data[(r*width + c) * 3 + 2] * w2);
			}
		}
	}
	return ol;
}

int main() {
#pragma region RealSense Initialize
	
	rs2::context ctx;
	if (ctx.query_devices().size() == 0) {
		std::cout << "[Error; realsense] realsense stream is not available" << std::endl;
		return false;
	}

	rs2::config cfg;
	//cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8);
	//cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16);
	cfg.enable_stream(RS2_STREAM_COLOR);
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);
	rs2::pipeline pipe;
	rs2::pipeline_profile selection = pipe.start(cfg);
	auto sensor = selection.get_device().first<rs2::depth_sensor>();
	std::cout << "Depth Scale : " << sensor.get_depth_scale() << std::endl;

	const rs2::stream_profile to = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::stream_profile>();
	auto const intrin_color = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
	auto const intrin_depth = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
	auto extrin_d2c = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_extrinsics_to(to);

	int c_width = pipe.wait_for_frames().get_color_frame().get_width();
	int c_height = pipe.wait_for_frames().get_color_frame().get_height();
	int d_width = pipe.wait_for_frames().get_depth_frame().get_width();
	int d_height = pipe.wait_for_frames().get_depth_frame().get_height();

	std::cout << "[COLOR] INTRINSIC" << std::endl;
	std::cout << "fx : " << intrin_color.fx << std::endl;
	std::cout << "fy : " << intrin_color.fy << std::endl;
	std::cout << "ppx : " << intrin_color.ppx << std::endl;
	std::cout << "ppy : " << intrin_color.ppy << std::endl;
	std::cout << "width : " << intrin_color.width << std::endl;
	std::cout << "height : " << intrin_color.height << std::endl;

	std::cout << "[DEPTH] INTRINSIC" << std::endl;
	std::cout << "fx : " << intrin_depth.fx << std::endl;
	std::cout << "fy : " << intrin_depth.fy << std::endl;
	std::cout << "ppx : " << intrin_depth.ppx << std::endl;
	std::cout << "ppy : " << intrin_depth.ppy << std::endl;
	std::cout << "width : " << intrin_depth.width << std::endl;
	std::cout << "height : " << intrin_depth.height << std::endl;

	glm::fmat3 rot = glm::make_mat3x3(extrin_d2c.rotation);
	glm::fvec3 trans = glm::make_vec3(extrin_d2c.translation);
	glm::fmat4 extrinsic = glm::mat4(1.0f);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			extrinsic[i][j] = rot[i][j];
	for (int i = 0; i < 3; i++)
		extrinsic[3][i] = trans[i] * 1000.f;
	//std::cout << "[EXTRINSIC]" << std::endl;
	//std::cout << "Rotation" << std::endl;
	//std::cout << glm::to_string(rot) << std::endl;
	//std::cout << "Translation" << std::endl;
	//std::cout << glm::to_string(trans) << std::endl;
	//std::cout << glm::to_string(extrinsic) << std::endl;

	rs2::align aling_to_depth(RS2_STREAM_DEPTH);
	rs2::align aling_to_color(RS2_STREAM_COLOR);
	rs2::colorizer colorizer;
#pragma endregion

	const char* color_file_name = "color.png";
	const char* depth_file_name = "depth.png";

	cv::Mat color_mat = cv::Mat::zeros(cv::Size(c_width, c_height), CV_8UC3);
	cv::Mat depth_mat = cv::Mat::zeros(cv::Size(d_width, d_height), CV_16UC1);
	cv::Mat saved_img = cv::Mat::zeros(cv::Size(c_width, c_height), CV_8UC3);
	cv::Mat saved_depth = cv::Mat::zeros(cv::Size(d_width, d_height), CV_16UC1);
	cv::Mat c_depth_mat = cv::Mat::zeros(cv::Size(d_width, d_height), CV_8UC3);

	int key = -1;

	ms::Intrinsic_
		color_int(intrin_color.fx, intrin_color.fy, intrin_color.ppx, intrin_color.ppy, intrin_color.width, intrin_color.height),
		depth_int(intrin_depth.fx, intrin_depth.fy, intrin_depth.ppx, intrin_depth.ppy, intrin_depth.width, intrin_depth.height);

	ms::Segmentor segmentor(color_int, depth_int, extrinsic);

	ms::DepthEdgeParam de_param(1.2, 2);
	ms::MorphSnakeParam ms_param(2000, 3, ms::CHANNEL::RED, 50, 0.15, 1, 1);
	ms::CannyParam cn_param(700, 1500, true);
	ms::InitLevelSetParam ls_param(240, 320, 10);
	ms::VisualizationParam vs_param(false, false, false, false, false);
	ms::EdgeSelectionParam es_param(true, true);
	bool streaming_segmentation_on = false;
	bool is_image_saved = false;
	bool load_img_segmnetation_on = false;
	int size = 0;
	float sum = 0;

	while (key != 'q') {
#pragma region GetRealSenseFrame
		// Original FrameSet
		auto frameSet = pipe.wait_for_frames();
		auto color_frame = frameSet.get_color_frame();
		auto depth_frame = frameSet.get_depth_frame();
		auto c_depth = colorizer.colorize(depth_frame);
		std::memcpy(color_mat.data, color_frame.get_data(), sizeof(unsigned char)* c_height*c_width * 3);
		std::memcpy(depth_mat.data, depth_frame.get_data(), sizeof(unsigned short)* d_height*d_width);
		std::memcpy(c_depth_mat.data, c_depth.get_data(), sizeof(unsigned char)* d_height*d_width * 3);
		cv::cvtColor(color_mat, color_mat, cv::COLOR_BGR2RGB);
#pragma endregion

		if (key == 'c') {
			cv::imwrite(color_file_name, color_mat);
			cv::imwrite(depth_file_name, depth_mat);
			is_image_saved = true;
		}
		if (key == 'l' && is_image_saved) {
			saved_img = cv::imread(color_file_name);
			saved_depth = cv::imread(depth_file_name, cv::IMREAD_ANYDEPTH);
			load_img_segmnetation_on = !load_img_segmnetation_on;
			cv::destroyAllWindows();
		}
		if (load_img_segmnetation_on) {
			cv::Mat mask = segmentor.doSegmentation(saved_img, saved_depth, de_param, cn_param, ms_param, ls_param, 2, ms::MASK_AT::COLOR, vs_param, es_param);
			cv::Mat ol = overlay(saved_img, 0.4, mask, 0.6);
			cv::imshow("ol", ol);
		}
		
		if (key == 's' /* Streaming Image Segmentation */) {
			streaming_segmentation_on = !streaming_segmentation_on;
			cv::destroyAllWindows();
		}
		if (streaming_segmentation_on) {
			c_time start = std::chrono::high_resolution_clock::now();
			cv::Mat mask = segmentor.doSegmentation(color_mat, depth_mat, de_param, cn_param, ms_param, ls_param, 2, ms::MASK_AT::COLOR, vs_param, es_param);
			c_time end = std::chrono::high_resolution_clock::now();
			//cv::Mat mask = segmentor.doSegmentation(color, aligned_depth, de_param, cn_param, ms_param, ls_param, downscale, ms::MASK_AT::COLOR);
			std::chrono::duration<double> time = end - start;
			
			std::cout << 1. / (time.count()) << "FPS";
			std::cout << std::endl;

			cv::Mat ol = overlay(color_mat, 0.4, mask, 0.6);
			cv::imshow("ol", ol);
		}


		if (key == 'w'/*Warp Color Image To Depth Space*/) {
			vs_param.warpping_on = !vs_param.warpping_on;
			cv::destroyAllWindows();
		}
		if (vs_param.warpping_on) {
			cv::Mat mask = segmentor.doSegmentation(color_mat, depth_mat, de_param, cn_param, ms_param, ls_param, 2, ms::MASK_AT::COLOR, vs_param, es_param);
			cv::Mat ol = overlay(c_depth_mat, 0.4, mask, 0.6);
			cv::imshow("ol", ol);
		}
		
		if (key == '1') {
			vs_param.depth_edge_on = !vs_param.depth_edge_on;
			cv::destroyWindow("Depth_Edge");
		}
		if (key == '2') {
			vs_param.canny_edge_on = !vs_param.canny_edge_on;
			cv::destroyWindow("Canny_Edge");
		}
		if (key == '3') {
			vs_param.merge_edge_on = !vs_param.merge_edge_on;
			cv::destroyWindow("Merged_Edge");
		}
		if (key == '4') {
			vs_param.inv_edge_on = !vs_param.inv_edge_on;
			cv::destroyWindow("Inv_Edge");
		}


		if (key == 'o') {
			es_param.use_depth_edge = !es_param.use_depth_edge;
		}
		if (key == 'p') {
			es_param.use_canny_edge= !es_param.use_canny_edge;
		}

		
		cv::imshow("depth", c_depth_mat);
		cv::imshow("color", color_mat);
		key = cv::waitKey(1);
	}
	return 0;
}