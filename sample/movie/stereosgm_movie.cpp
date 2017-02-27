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
#include <adasworks/io/imagestream.h>
#include <thread>

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

#include <opencv2/opencv.hpp>

struct RectData
{
    cv::Mat rmap[2][2], R, T, P1, P2;

};

void initRectification(const std::string & intrinsic_filename, 
    const std::string & extrinsics_filename,
    int img_w, int img_h, RectData & r)
{
    // reading intrinsic parameters
    //std::string  = "./data/intrinsics.yml";
    cv::FileStorage fs_intrinsic(intrinsic_filename, cv::FileStorage::READ);
    if (!fs_intrinsic.isOpened()) {
        printf("Failed to open file %s", intrinsic_filename.c_str());
        exit(1);
    }
    cv::Mat M1, D1, M2, D2;
    fs_intrinsic["M1"] >> M1;
    fs_intrinsic["D1"] >> D1;
    fs_intrinsic["M2"] >> M2;
    fs_intrinsic["D2"] >> D2;

    // reading extrinsics parameters
    //std::string extrinsics_filename = "./data/extrinsics.yml";
    cv::FileStorage fs_extrinsics(extrinsics_filename, cv::FileStorage::READ);
    if (!fs_extrinsics.isOpened()) {
        printf("Failed to open file %s", extrinsics_filename.c_str());
        exit(1);
    }
    cv::Mat T, R, R1, R2, P1, P2, Q;
    
    fs_extrinsics["R"] >> R;
    fs_extrinsics["T"] >> T;
    cv::Size img_size = cv::Size(img_w, img_h);
    cv::Rect roi1, roi2;
    cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2);
    //Precompute maps for cv::remap()
    cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, r.rmap[0][0], r.rmap[0][1]);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, r.rmap[1][0], r.rmap[1][1]);//LOG_INFO("P1, P2, Q sizes: %dx%d, %dx%d, %dx%d", P1.cols, P1.rows, P2.cols, P2.rows, Q.cols, Q.rows);
    r.P1 = P1;
    r.P2 = P2;
    r.T = T;
    r.R = R;
}

#include <adasworks/io/imagestreamfactory.h>

int main(int argc, char* argv[]) {

	if (argc < 3) {
		std::cerr << "usage: stereosgm left_img_fmt right_img_fmt" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::string left_filename_fmt, right_filename_fmt;
	left_filename_fmt = argv[1];
	right_filename_fmt = argv[2];


    std::string uri = argv[1];
    adasworks::io::ImageStreamFactory factory;
    std::unique_ptr<adasworks::io::ImageStream> stream0(factory.create(uri));
    CHECK(stream0, "Failed to create stream for: %s", uri.c_str());
    uri = argv[2];
    std::unique_ptr<adasworks::io::ImageStream> stream1(factory.create(uri));
    CHECK(stream1, "Failed to create stream for: %s", uri.c_str());

    std::vector<adasworks::io::ImageHandlePtr> imgs0;
    std::vector<adasworks::io::ImageHandlePtr> imgs1;
    adasworks::io::ImageStream::Status status;

    status = stream0->read(imgs0, 100);
    status = stream1->read(imgs1, 100);
    adasworks::io::ImageBuffer buf0 = imgs0[0]->lock();
    adasworks::io::ImageBuffer buf1 = imgs1[0]->lock();
    int imgw = buf0.width();
    int imgh = buf0.height();


    RectData rd;
    initRectification(argv[3], argv[4], imgw, imgh, rd);

    //cv::VideoCapture cap0(left_filename_fmt);
    //cv::VideoCapture cap1(right_filename_fmt);

	// dangerous
    cv::Mat leftc, rightc, leftc_full_r, rightc_full_r;
    cv::Mat left(imgh, imgw, CV_8UC1);
    cv::Mat right(imgh, imgw, CV_8UC1);;
    memcpy(left.data, buf0.data(), imgh * imgw);
    memcpy(right.data, buf1.data(), imgh * imgw);
    cv::Mat left_gray, right_gray, leftc_full, rightc_full;
    cv::cvtColor(left, leftc_full, CV_BayerRG2BGR);
    cv::cvtColor(right, rightc_full, CV_BayerRG2BGR);
    cv::remap(leftc_full, leftc_full_r, rd.rmap[0][0], rd.rmap[0][1], cv::INTER_LINEAR);
    cv::remap(rightc_full, rightc_full_r, rd.rmap[1][0], rd.rmap[1][1], cv::INTER_LINEAR);


    //cap0 >> leftc;
    //cap1 >> rightc;

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
    double scale = 1.0;
    cv::resize(leftc_full, half_left, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::resize(rightc_full, half_right, cv::Size(), scale, scale, cv::INTER_LINEAR);

    cv::cvtColor(half_left, left_gray, CV_BGR2GRAY);
    cv::cvtColor(half_right, right_gray, CV_BGR2GRAY);

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

	int width = half_left.cols;
	int height = half_left.rows;

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

        static cv::Mat left_disp_subpx(height, width, CV_32FC1);

        if (!demo.isPaused())
        {
            cv::resize(leftc_full_r, half_left, cv::Size(), scale, scale, cv::INTER_LINEAR);
            cv::resize(rightc_full_r, half_right, cv::Size(), scale, scale, cv::INTER_LINEAR);
            cv::cvtColor(half_left, left_gray, CV_BGR2GRAY);
            cv::cvtColor(half_right, right_gray, CV_BGR2GRAY);

            ssgm.execute(left_gray.data, right_gray.data, (void**)&d_output_buffer); // , sgm::DST_TYPE_CUDA_PTR, 16);

        //static cv::Mat left_disp_subpix_color(height, width, CV_8UC3);
            cudaMemcpy(left_disp_subpx.data, d_output_buffer, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
        }
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


        if (demo.get_calc_dist())
        {
            demo.set_calc_dist(false);

            double x_pos = demo.getXpos();
            double y_pos = demo.getYpos();

            int hws = 5;
            int from_w = std::max(0, (int)x_pos - hws);
            int to_w = std::min(width, (int)x_pos + hws);
            int from_h = std::max(0, (int)y_pos - hws);
            int to_h = std::min(height, (int)y_pos + hws);

            float avg_disp = 0.0f;
            int cont = 0;
            for (int j = from_h; j < to_h; ++j)
            {
                for (int i = from_w; i < to_w; ++i)
                {
                    float disp = left_disp_subpx.at<float>(j, i);
                    
                    if (disp > 0.5f)
                    {
                        avg_disp += disp;
                        ++cont;
                    }
                }
            }
            avg_disp /= cont;
            double b = cv::norm(rd.T);
            double z = rd.P1.at<double>(0) * b / avg_disp;
            double x = (x_pos - rd.P1.at<double>(0, 2)) * z / rd.P1.at<double>(0, 0);
            double y = (y_pos - rd.P1.at<double>(1, 2)) * z / rd.P1.at<double>(1, 1);
            
            std::cout << x << "  " << y << "   " << z << " norm: " << sqrt(x*x + y*y + z*z) << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        //renderer.render_disparity(nullptr, 128);
        demo.swap_buffer();
        //cv::imshow("colored disparity", left_disp_subpix_color);
        //cv::imshow("left gray image", left);
        //int key = cv::waitKey(1);
        //if (key == 27)
          //break;
        if (!demo.isPaused())
        {
            status = stream0->read(imgs0, 100);
            status = stream1->read(imgs1, 100);
            adasworks::io::ImageBuffer buf0 = imgs0[0]->lock();
            adasworks::io::ImageBuffer buf1 = imgs1[0]->lock();

            frame_no++;
            memcpy(left.data, buf0.data(), imgh * imgw);
            memcpy(right.data, buf1.data(), imgh * imgw);
            cv::cvtColor(left, leftc_full, CV_BayerBG2BGR);
            cv::cvtColor(right, rightc_full, CV_BayerBG2BGR);
            cv::remap(leftc_full, leftc_full_r, rd.rmap[0][0], rd.rmap[0][1], cv::INTER_LINEAR);
            cv::remap(rightc_full, rightc_full_r, rd.rmap[1][0], rd.rmap[1][1], cv::INTER_LINEAR);
        }

    }
}
