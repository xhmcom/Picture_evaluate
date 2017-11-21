#include <opencv.hpp>
#include <iostream>
#include <string>
//#include <fstream>
#include "Harmonious.h"
#define M_PI 3.1415926

using namespace std;
using namespace cv;

int cnt_area[8];
//ofstream of;

bool segment_huidu(const Mat& src, Mat& label, int k) {
	Mat centers(k, 1, CV_32F);
	Mat p = Mat::zeros(src.cols * src.rows, 1, CV_32F);
	
	for (int i = 0; i < src.cols * src.rows; i++) {
		p.at<float>(i, 0) = src.at<uchar>(i / src.cols, i % src.cols);
	}
	
	kmeans(p, k, label, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
		KMEANS_PP_CENTERS, centers);

	
	//算centers距离
	float min_centers = 100000;
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++){
			float dist = abs(centers.at<float>(i, 0) - centers.at<float>(j, 0));
			//cout << src.at<Vec3b>(first / src.cols, first % src.cols)[1] << endl;
			if (dist < min_centers) {
				min_centers = dist;
			}
		}
	}
	//输出最小颜色差距
	//cout << min_centers << endl;
	//of << min_centers << endl;
	if (min_centers > 33) {
		return true;
	}
	return false;
}

bool segment_by_hsv(const Mat& src, Mat& label, int k) {
	Mat centers(k, 1, CV_32F);
	Mat p = Mat::zeros(src.cols * src.rows, 1, CV_32F);
	for (int i = 0; i < src.cols * src.rows; i++) {
		p.at<float>(i, 0) = src.at<Vec3b>(i / src.cols, i % src.cols)[1];
	}
	kmeans(p, k, label, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
		KMEANS_PP_CENTERS, centers);
	//算centers距离
	float min_centers = 100000;
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++){
			float dist = abs(centers.at<float>(i, 0) - centers.at<float>(j, 0));
			//cout << src.at<Vec3b>(first / src.cols, first % src.cols)[1] << endl;
			if (dist < min_centers) {
				min_centers = dist;
			}
		}
	}
	//输出最小颜色差距
	//cout << min_centers << endl;
	//of << min_centers << endl;
	if (min_centers > 35) {
		return true;
	}
	return false;
}
bool segment(const Mat& src, Mat& label, int k) {
	Mat centers(k, 3, CV_32F);
	Mat p = Mat::zeros(src.cols * src.rows, 3, CV_32F);
	for (int i = 0; i < src.cols * src.rows; i++) {
		p.at<float>(i, 0) = src.at<Vec3b>(i / src.cols, i % src.cols)[1];
		p.at<float>(i, 1) = src.at<Vec3b>(i / src.cols, i % src.cols)[2];
		p.at<float>(i, 2) = src.at<Vec3b>(i / src.cols, i % src.cols)[0];
	}
	kmeans(p, k, label, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
		KMEANS_PP_CENTERS, centers);
	//算centers距离
	float min_centers = 100000;
	for (int i = 0; i < k-1; i++) {
		for (int j = i + 1; j < k; j++){
			float dist = sqrt(pow(centers.at<float>(i, 0) - centers.at<float>(j, 0), 2) +
								pow(centers.at<float>(i, 1) - centers.at<float>(j, 1), 2) +
									pow(centers.at<float>(i, 2) - centers.at<float>(j, 2), 2));
			//cout << src.at<Vec3b>(first / src.cols, first % src.cols)[1] << endl;
			if (dist < min_centers) {
				min_centers = dist;
			}
		}
	}
	//输出最小颜色差距
	//cout << min_centers << endl;
	//of << min_centers << endl;
	if (min_centers > 50) {
		return true;
	}
	return false;
}
int color_area[8];
void show_color(const Mat& src, const Mat& label, int color, int num, int id) {
	Mat im = src.clone();
	for (int i = 0; i < src.rows * src.cols; i++) {
		if (label.at<int>(i, 0) != color) {
			im.at<Vec3b>(i / src.cols, i % src.cols)[0] = 0;
			im.at<Vec3b>(i / src.cols, i % src.cols)[1] = 0;
			im.at<Vec3b>(i / src.cols, i % src.cols)[2] = 0;
		}
	}
	string s_id = to_string((int)id);
	//imshow("color" + id, im);
	imwrite("result/" +to_string(num) + "_color_" + s_id + ".jpg", im);
}
void show_gray_color(const Mat& src, const Mat& label, int color, int num, int id) {
	Mat im = src.clone();
	float gray_color = 0;
	for (int i = 0; i < src.rows * src.cols; i++) {
		if (label.at<int>(i, 0) != color) {
			im.at<uchar>(i / src.cols, i % src.cols) = 0;
		}
	}
	string s_id = to_string((int)id);
	//imshow("color" + id, im);
	imwrite("result/" + to_string(num) + "_color_" + s_id + ".jpg", im);
}

// calculate the mean color for each group
int cnt_x[256 * 256 * 256];

Vec3b mean_color(const Mat& src, const Mat& label, int color) {
	//cerr << "mean_color" << endl;
	typedef unsigned char uchar;
	const int base = 256;
	const int base2 = 256 * 256;
	fill(cnt_x, cnt_x + (256 * 256 * 256), 0);
	auto encode = [&](uchar B, uchar G, uchar R) {
		return B * base2 + G * base + R;
	};
	auto decode = [&](int code)->Vec3b {
		//cerr << (code / base2) << " " << ((code / base) % base) << " " << (code % base) << endl;
		return Vec3b(code / base2, (code / base) % base, code % base);
	};
	color_area[color] = 0;
	for (int i = 0; i < src.rows * src.cols; i++) {
		if (label.at<int>(i, 0) == color) {
			uchar B = src.at<Vec3b>(i / src.cols, i % src.cols)[0];
			uchar G = src.at<Vec3b>(i / src.cols, i % src.cols)[1];
			uchar R = src.at<Vec3b>(i / src.cols, i % src.cols)[2];
			color_area[color] ++;
			cnt_x[encode(B, G, R)]++;
		}
	}
	int mx = 0;
	for (int i = 1; i < 256 * 256 * 256; i++) {
		if (cnt_x[i] > cnt_x[mx]) {
			mx = i;
		}
	}
	return decode(mx);
}

double norm_dis(double miu, double sig, double x) {
	return (1 / sqrt(2 * M_PI * sig * sig)) * exp(-pow(x - miu, 2) / (2 * sig * sig));
}

float mean_gray(Mat& src, Mat& label, int color) {
	color_area[color] = 0;
	float color_sum = 0;
	for (int i = 0; i < src.rows * src.cols; i++) {
		if (label.at<int>(i, 0) == color) {
			color_area[color] ++;
			color_sum += (float)src.at<uchar>(i / src.cols, i % src.cols);
		}
	}
	float color_mean = color_sum / color_area[color];
	return color_mean;
}

double seg_by_gray(string path, int num) {
	Mat src = imread(path);
	Mat dst;
	cvtColor(src, dst, COLOR_RGB2GRAY); // RGB 转 gray
	imwrite("result/" + to_string(num) + "__gray.jpg", dst);
	
	int k;
	Mat label(src.rows * src.cols, 1, CV_8UC1);
	k = 5;
	segment_huidu(dst, label, k);
	//自动选k值
	//for (k = 8; k > 1; k--) {
	//	if (segment_huidu(dst, label, k)) break;
	//}
	//if (k == 1) k = 2;

	vector<uchar> m_color(k);
	int bj[8];
	for (int i = 0; i < k; i++) {
		bj[i] = i;
		m_color[i] = mean_gray(dst, label, i);
	}
	//sort 按照灰度排序
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (m_color[bj[i]] < m_color[bj[j]]) {
				int tmp = bj[i];
				bj[i] = bj[j];
				bj[j] = tmp;
			}
		}
	}
	// 调试输出各个块
	for (int i = 0; i < k; i++) {
		show_gray_color(dst, label, bj[i], num, i);
	}
	float area_ratio[8];

	for (int i = 0; i < k; i++) {
		area_ratio[i] = (float)color_area[bj[i]] / (src.cols*src.rows);
		//cout << i + 1 << ":" << area_ratio[i] << endl;
		//of << i + 1 << ":" << area_ratio[i] << endl;
	}
	double high_value, mid_value, low_value;
	high_value = norm_dis(0.191, 0.191 / 3, area_ratio[0]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	mid_value = norm_dis(0.618, 0.618 / 3, area_ratio[1] + area_ratio[2] + area_ratio[3]) / norm_dis(0.618, 0.618 / 3, 0.618) * 61.8;
	low_value = norm_dis(0.191, 0.191 / 3, area_ratio[4]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	double final_value = high_value + mid_value + low_value;
	//cout << "score: " << final_value << endl;
	//of << "score: " << final_value << endl;
	//norm_dis()
	//cout << "color:" << endl;
	for (int i = 0; i < k; i++) {

		uchar color = m_color[bj[i]];

		//cout << i + 1 << ":" << " g=" << (int)color << endl;
		//of << i + 1 << ":" << " g=" << (int)color << endl;
	}

	//cout << "------------------------" << endl;
	//cout << endl;
	//of << "------------------------" << endl;
	//of << endl;

	return final_value;
}

uchar mean_hsv(Mat& src, Mat& label, int color) {
	color_area[color] = 0;
	double color_sum = 0;
	
	for (int i = 0; i < src.rows * src.cols; i++) {
		if (label.at<int>(i, 0) == color) {
			color_area[color] ++;
			color_sum += src.at<Vec3b>(i / src.cols, i % src.cols)[1];
		}
	}
	uchar color_mean = color_sum / color_area[color];
	return color_mean;
}

double seg_by_hsv(string path, int num) {
	Mat src = imread(path);
	Mat dst;
	cvtColor(src, dst, CV_BGR2HSV); // RGB 转 Hsv
	
	int k;
	Mat label(src.rows * src.cols, 1, CV_8UC1);
	k = 5;
	segment_by_hsv(dst, label, k);
	//自动选k值
	//for (k = 8; k > 1; k--) {
	//	if (segment_by_hsv(dst, label, k)) break;
	//}
	//if (k == 1) k = 2;
	

	vector<uchar> m_color(k);
	vector<Vec3b> m_rgb_color(k);
	int bj[8];
	for (int i = 0; i < k; i++) {
		bj[i] = i;
		m_rgb_color[i] = mean_color(src, label, i);
		m_color[i] = mean_hsv(dst, label, i);
		
	}
	
	//sort 按照纯度排序(bj代表新的顺序)
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (m_color[bj[i]] < m_color[bj[j]]) {
				int tmp = bj[i];
				bj[i] = bj[j];
				bj[j] = tmp;
			}
		}
	}
	// 输出各个色块
	for (int i = 0; i < k; i++) {
		show_color(src, label, bj[i], num, i);
	}

	float area_ratio[8];
	// 输出各个色块比例
	for (int i = 0; i < k; i++) {
		area_ratio[i] = (float)color_area[bj[i]] / (src.cols*src.rows);
		//cout << i + 1 << ":" << area_ratio[i] << endl;
		//of << i + 1 << ":" << area_ratio[i] << endl;
	}
	// 评分
	double high_sat, mid_sat, low_sat;
	high_sat = norm_dis(0.191, 0.191 / 3, area_ratio[0]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	mid_sat = norm_dis(0.618, 0.618 / 3, area_ratio[1] + area_ratio[2] + area_ratio[3]) / norm_dis(0.618, 0.618 / 3, 0.618) * 61.8;
	low_sat = norm_dis(0.191, 0.191 / 3, area_ratio[4]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	double final_value = high_sat + mid_sat + low_sat;
	//cout << "score: " << final_value << endl;
	//of << "score: " << final_value << endl;
	// 输出颜色
	//cout << "color:" << endl;
	for (int i = 0; i < k; i++) {

		Vec3b color = m_rgb_color[bj[i]];

		//cout << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
		//of << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
	}

	//cout << "------------------------" << endl;
	//cout << endl;
	//of << "------------------------" << endl;
	//of << endl;

	return final_value;
}
double seg_by_lab(string path, int num) {
	Mat src = imread(path);
	Mat dst;
	cvtColor(src, dst, CV_BGR2Lab); // RGB 转 Lab
	
	int k;
	Mat label(src.rows * src.cols, 1, CV_8UC1);
	k = 3;
	segment(src, label, k);
	//自动选k值
	/*for (k = 8; k > 1; k--) {
		if (segment(src, label, k)) break;// 直接用rgb色彩空间用分割
	}
	if (k == 1) k = 2;*/
	

	vector<Vec3b> m_color(k);
	int bj[8];
	for (int i = 0; i < k; i++) {
		bj[i] = i;
		m_color[i] = mean_color(src, label, i);
	}
	//sort 按照类别面积大小排序
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (color_area[bj[i]] < color_area[bj[j]]) {
				int tmp = bj[i];
				bj[i] = bj[j];
				bj[j] = tmp;
			}
		}
	}
	// 调试输出各个块
	for (int i = 0; i < k; i++) {
		show_color(src, label, bj[i], num, i);
	}
	float area_ratio[8];

	for (int i = 0; i < k; i++) {
		area_ratio[i] = (float)color_area[bj[i]] / (src.cols*src.rows);
		//cout << i + 1 << ":" << area_ratio[i] << endl;
		//of << i + 1 << ":" << area_ratio[i] << endl;
	}
	//评分
	double largest_hue, o_hue1, o_hue2;
	o_hue1 = norm_dis(0.191, 0.191 / 3, area_ratio[1]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	largest_hue = norm_dis(0.618, 0.618 / 3, area_ratio[0]) / norm_dis(0.618, 0.618 / 3, 0.618) * 61.8;
	o_hue2 = norm_dis(0.191, 0.191 / 3, area_ratio[2]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	double final_hue = largest_hue + o_hue1 + o_hue2;
	//cout << "score: " << final_hue << endl;
	//of << "score: " << final_hue << endl;

	//cout << "color:" << endl;
	for (int i = 0; i < k; i++) {

		Vec3b color = m_color[bj[i]];

		//cout << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
		//of << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
	}

	//cout << "------------------------" << endl;
	//cout << endl;
	//of << "------------------------" << endl;
	//of << endl;

	return final_hue;
}

