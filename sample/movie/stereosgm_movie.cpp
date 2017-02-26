/*
Copyright 2016 fixstars

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libsgm.h>

#include "demo.h"
#include "renderer.h"

//e:\Downloads\venus\im2.ppm e:\Downloads\venus\im6.ppm
//e:\Downloads\barn1\im2.ppm e:\Downloads\barn1\im6.ppm
//e:\Downloads\sawtooth\im2.ppm e:\Downloads\sawtooth\im6.ppm
//e:\Downloads\cones\im2.png e:\Downloads\cones\im6.png
//e:\Downloads\teddy\im2.png e:\Downloads\teddy\im6.png
//e:\Downloads\data_scene_flow\image_2\000000_10.png e:\Downloads\data_scene_flow\image_3\000000_10.png
//d:\tmp\video1156956.mp4 d:\tmp\video2156956.mp4


//template<typename T>
//void writeFalseColors(const cv::Mat & disp, cv::Mat & dispColor, float max_val,
//    int32_t width_, int32_t height_) {
//
//    // color map
//    float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
//    { 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
//    float sum = 0;
//    for (int32_t i = 0; i<8; i++)
//        sum += map[i][3];
//
//    float weights[8]; // relative weights
//    float cumsum[8];  // cumulative weights
//    cumsum[0] = 0;
//    for (int32_t i = 0; i<7; i++) {
//        weights[i] = sum / map[i][3];
//        cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
//    }
//
//    // for all pixels do
//    for (int32_t v = 0; v<height_; v++) {
//        for (int32_t u = 0; u<width_; u++) {
//
//            // get normalized value
//            float val = std::min(std::max(disp.at<T>(v, u) / max_val, 0.0f), 1.0f);
//
//            // find bin
//            int32_t i;
//            for (i = 0; i<7; i++)
//                if (val<cumsum[i + 1])
//                    break;
//
//            // compute red/green/blue values
//            float   w = 1.0 - (val - cumsum[i])*weights[i];
//            uint8_t r = (uint8_t)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
//            uint8_t g = (uint8_t)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
//            uint8_t b = (uint8_t)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
//
//            // set pixel
//            dispColor.at<cv::Vec3b>(v, u) = cv::Vec3b(b, g, r);
//        }
//    }
//
//}



int main(int argc, char* argv[]) {

	if (argc < 3) {
		std::cerr << "usage: stereosgm left_img_fmt right_img_fmt" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::string left_filename_fmt, right_filename_fmt;
	left_filename_fmt = argv[1];
	right_filename_fmt = argv[2];

    cv::VideoCapture cap0(left_filename_fmt);
    cv::VideoCapture cap1(right_filename_fmt);

	// dangerous
    cv::Mat leftc, rightc;
    cv::Mat left;
    cv::Mat right;

    cap0 >> leftc;
    cap1 >> rightc;

    //if (leftc.cols % 2 != 0)
    //{
    //    leftc = leftc(cv::Rect(0, 0, leftc.cols - 1, leftc.rows));
    //    rightc = rightc(cv::Rect(0, 0, rightc.cols - 1, rightc.rows));
    //}

    //if (leftc.rows % 2 != 0)
    //{
    //    leftc = leftc(cv::Rect(0, 0, leftc.cols, leftc.rows - 1));
    //    rightc = rightc(cv::Rect(0, 0, rightc.cols, rightc.rows - 1));
    //}
    cv::Mat half_left, half_right;
    double scale = 0.25;
    cv::resize(leftc, half_left, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::resize(rightc, half_right, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cv::cvtColor(half_left, left, CV_BGR2GRAY);
    cv::cvtColor(half_right, right, CV_BGR2GRAY);

	int disp_size = 64;


	if (left.size() != right.size() || left.type() != right.type()) {
		std::cerr << "mismatch input image size" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int bits = 0;

	switch (left.type()) {
	case CV_8UC1: bits = 8; break;
	case CV_16UC1: bits = 16; break;
	default:
		std::cerr << "invalid input image color format" << left.type() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int width = left.cols;
	int height = left.rows;

	//cudaGLSetGLDevice(0);

	SGMDemo demo(width, height);

	if (demo.init()) {
		printf("fail to init SGM Demo\n");
		std::exit(EXIT_FAILURE);
	}

	sgm::StereoSGM ssgm(width, height, disp_size, bits, 32, sgm::EXECUTE_INOUT_HOST2CUDA);

	Renderer renderer(width, height);
	
	float* d_output_buffer = NULL;

	int frame_no = 0;
	while (!demo.should_close() && true) {

        cv::resize(leftc, half_left, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::resize(rightc, half_right, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::cvtColor(half_left, left, CV_BGR2GRAY);
        cv::cvtColor(half_right, right, CV_BGR2GRAY);

		ssgm.execute(left.data, right.data, (void**)&d_output_buffer); // , sgm::DST_TYPE_CUDA_PTR, 16);
        static cv::Mat left_disp_subpx(height, width, CV_32FC1);
        //static cv::Mat left_disp_subpix_color(height, width, CV_8UC3);
        cudaMemcpy(left_disp_subpx.data, d_output_buffer, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

        //writeFalseColors<float>(left_disp_subpx, left_disp_subpix_color, 128.0f, width, height);
       
        if (demo.get_flag() == 0)
        {
            renderer.render_input((uint8_t*)half_left.data);
        }
        else if (demo.get_flag() == 1)
        {
            renderer.render_disparity_color((float*)left_disp_subpx.data, disp_size);
        }
        else
        {
            renderer.render_disparity((float*)left_disp_subpx.data, disp_size);
        }

        //renderer.render_disparity(nullptr, 128);
        demo.swap_buffer();
        //cv::imshow("colored disparity", left_disp_subpix_color);
        //cv::imshow("left gray image", left);
        //int key = cv::waitKey(1);
        //if (key == 27)
          //break;
        frame_no++;
        if (!(cap0.read(leftc) && cap1.read(rightc)))
            break;
	}
}
