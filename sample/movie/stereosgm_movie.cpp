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

    cv::cvtColor(leftc, left, CV_BGR2GRAY);
    cv::cvtColor(rightc, right, CV_BGR2GRAY);

	int disp_size = 128;


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

	sgm::StereoSGM ssgm(width, height, disp_size, bits, 16, sgm::EXECUTE_INOUT_HOST2CUDA);

	Renderer renderer(width, height);
	
	uint16_t* d_output_buffer = NULL;

	int frame_no = 0;
	while (!demo.should_close() && true) {

        cv::cvtColor(leftc, left, CV_BGR2GRAY);
        cv::cvtColor(rightc, right, CV_BGR2GRAY);

		ssgm.execute(left.data, right.data, (void**)&d_output_buffer); // , sgm::DST_TYPE_CUDA_PTR, 16);
      
       
        renderer.render_input((uint8_t*)leftc.data);
        //renderer.render_disparity(nullptr, 128);
        demo.swap_buffer();
        //cv::imshow("left color image", leftc);
        //cv::imshow("left gray image", left);
        int key = cv::waitKey(0);
        if (key == 27)
          break;
        frame_no++;
        //if (!(cap0.read(leftc) && cap1.read(rightc)))
            //break;

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


	}
}
