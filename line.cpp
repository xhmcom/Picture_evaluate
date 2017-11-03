#include <opencv.hpp>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <ctime>
#include <set>
#include <direct.h>
#include <math.h>
#include <fstream>
#define M_PI 3.1415926

using namespace std;
using namespace cv;

struct Slice {
public:
	int minx, miny, maxx, maxy;
	uchar color;
	int area;
	Slice(int _minx, int _miny, int _maxx, int _maxy, uchar _color)
		:minx(_minx), miny(_miny), maxx(_maxx), maxy(_maxy), color(_color), area(1) {

	}
};

// the const defintion
//const int k = 4;
int cnt_area[8];
ofstream of;
// segment the image by kmeans in Lab color space

bool segment_huidu(const Mat& src, Mat& label, int k) {
	Mat centers(k, 1, CV_32F);
	Mat p = Mat::zeros(src.cols * src.rows, 1, CV_32F);
	
	for (int i = 0; i < src.cols * src.rows; i++) {
		p.at<float>(i, 0) = src.at<uchar>(i / src.cols, i % src.cols);
	}
	
	kmeans(p, k, label, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
		KMEANS_PP_CENTERS, centers);

	
	//��centers����
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
	//�����С��ɫ���
	cout << min_centers << endl;
	of << min_centers << endl;
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
	//��centers����
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
	//�����С��ɫ���
	cout << min_centers << endl;
	of << min_centers << endl;
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
	//��centers����
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
	//�����С��ɫ���
	cout << min_centers << endl;
	of << min_centers << endl;
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

void fill_mean(Mat& src, const Mat& label, const vector<Vec3b>& m_color) {
	
	for (int i = 0; i < src.rows * src.cols; i++) {
		src.at<Vec3b>(i / src.cols, i % src.cols) = m_color[label.at<int>(i, 0)];
		cnt_area[label.at<int>(i, 0)]++;
	}
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

void seg_by_gray(Mat& src, int& num) {
	Mat dst;
	cvtColor(src, dst, COLOR_RGB2GRAY); // RGB ת gray
	imwrite("result/" + to_string(num) + "__gray.jpg", dst);
	
	int k;
	Mat label(src.rows * src.cols, 1, CV_8UC1);
	k = 5;
	segment_huidu(dst, label, k);
	//�Զ�ѡkֵ
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
	//sort ���ջҶ�����
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (m_color[bj[i]] < m_color[bj[j]]) {
				int tmp = bj[i];
				bj[i] = bj[j];
				bj[j] = tmp;
			}
		}
	}
	// �������������
	for (int i = 0; i < k; i++) {
		show_gray_color(dst, label, bj[i], num, i);
	}
	float area_ratio[8];

	for (int i = 0; i < k; i++) {
		area_ratio[i] = (float)color_area[bj[i]] / (src.cols*src.rows);
		cout << i + 1 << ":" << area_ratio[i] << endl;
		of << i + 1 << ":" << area_ratio[i] << endl;
	}
	double high_value, mid_value, low_value;
	high_value = norm_dis(0.191, 0.191 / 3, area_ratio[0]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	mid_value = norm_dis(0.618, 0.618 / 3, area_ratio[1] + area_ratio[2] + area_ratio[3]) / norm_dis(0.618, 0.618 / 3, 0.618) * 61.8;
	low_value = norm_dis(0.191, 0.191 / 3, area_ratio[4]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	double final_value = high_value + mid_value + low_value;
	cout << "score: " << final_value << endl;
	of << "score: " << final_value << endl;
	//norm_dis()
	cout << "color:" << endl;
	for (int i = 0; i < k; i++) {

		uchar color = m_color[bj[i]];

		cout << i + 1 << ":" << " g=" << (int)color << endl;
		of << i + 1 << ":" << " g=" << (int)color << endl;
	}

	cout << "------------------------" << endl;
	cout << endl;
	of << "------------------------" << endl;
	of << endl;
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

void seg_by_hsv(Mat& src, int& num) {
	Mat dst;
	cvtColor(src, dst, CV_BGR2HSV); // RGB ת Hsv
	
	int k;
	Mat label(src.rows * src.cols, 1, CV_8UC1);
	k = 5;
	segment_by_hsv(dst, label, k);
	//�Զ�ѡkֵ
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
	
	//sort ���մ�������(bj�����µ�˳��)
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (m_color[bj[i]] < m_color[bj[j]]) {
				int tmp = bj[i];
				bj[i] = bj[j];
				bj[j] = tmp;
			}
		}
	}
	// �������ɫ��
	for (int i = 0; i < k; i++) {
		show_color(src, label, bj[i], num, i);
	}

	float area_ratio[8];
	// �������ɫ�����
	for (int i = 0; i < k; i++) {
		area_ratio[i] = (float)color_area[bj[i]] / (src.cols*src.rows);
		cout << i + 1 << ":" << area_ratio[i] << endl;
		of << i + 1 << ":" << area_ratio[i] << endl;
	}
	// ����
	double high_sat, mid_sat, low_sat;
	high_sat = norm_dis(0.191, 0.191 / 3, area_ratio[0]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	mid_sat = norm_dis(0.618, 0.618 / 3, area_ratio[1] + area_ratio[2] + area_ratio[3]) / norm_dis(0.618, 0.618 / 3, 0.618) * 61.8;
	low_sat = norm_dis(0.191, 0.191 / 3, area_ratio[4]) / norm_dis(0.191, 0.191 / 3, 0.191) * 19.1;
	double final_value = high_sat + mid_sat + low_sat;
	cout << "score: " << final_value << endl;
	of << "score: " << final_value << endl;
	// �����ɫ
	cout << "color:" << endl;
	for (int i = 0; i < k; i++) {

		Vec3b color = m_rgb_color[bj[i]];

		cout << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
		of << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
	}

	cout << "------------------------" << endl;
	cout << endl;
	of << "------------------------" << endl;
	of << endl;
}
void seg_by_lab(Mat& src, int& num) {
	Mat dst;
	cvtColor(src, dst, CV_BGR2Lab); // RGB ת Lab
	//�Զ�ѡkֵ
	int k;
	Mat label(src.rows * src.cols, 1, CV_8UC1);
	
	for (k = 8; k > 1; k--) {
		if (segment(src, label, k)) break;// ֱ����rgbɫ�ʿռ��÷ָ�
	}
	if (k == 1) k = 2;
	

	vector<Vec3b> m_color(k);
	int bj[8];
	for (int i = 0; i < k; i++) {
		bj[i] = i;
		m_color[i] = mean_color(src, label, i);
	}
	//sort ������������С����
	for (int i = 0; i < k - 1; i++) {
		for (int j = i + 1; j < k; j++) {
			if (color_area[bj[i]] < color_area[bj[j]]) {
				int tmp = bj[i];
				bj[i] = bj[j];
				bj[j] = tmp;
			}
		}
	}
	// �������������
	for (int i = 0; i < k; i++) {
		show_color(src, label, bj[i], num, i);
	}
	float area_ratio[8];

	for (int i = 0; i < k; i++) {
		area_ratio[i] = (float)color_area[bj[i]] / (src.cols*src.rows);
		cout << i + 1 << ":" << area_ratio[i] << endl;
		of << i + 1 << ":" << area_ratio[i] << endl;
	}
	cout << "color:" << endl;
	for (int i = 0; i < k; i++) {

		Vec3b color = m_color[bj[i]];

		cout << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
		of << i + 1 << ":" << " B=" << (int)color[0] << " G=" << (int)color[1] << " R=" << (int)color[2] << endl;
	}

	cout << "------------------------" << endl;
	cout << endl;
	of << "------------------------" << endl;
	of << endl;
}

int main(int argc, char* argv[]) {
	
	of.open("result.txt");
	/*for (float x = 0.3; x > 0; x -= 0.02) {
		cout << x << ": " << norm_dis(0.191, 0.191 / 3, x) / norm_dis(0.191, 0.191 / 3, 0.191) << endl;
	}
	
	system("pause");
	return 0;*/
	for (int num = 1; num <= 162; num++){
		//if (num != 40) continue;
		//if (num != 1 && num != 3 && num != 40 && num != 87) continue;
		cout << "-----------" << num << "-----------" << endl;
		of << "-----------" << num << "-----------" << endl;
		Mat src;

		if (argc > 1)
			src = imread(argv[1]);
		else
			src = imread("pic/raw_" + to_string(num) + ".jpg");

		imwrite("result/" + to_string(num) + ".jpg", src);
		CV_Assert(!src.empty());
		//cout << "src.rows = " << src.rows << " src.cols = " << src.cols << endl;
		// convert image to Lab space
		//imshow("src", src);
		
		//seg_by_lab(src, num);
		//seg_by_gray(src, num);
		seg_by_hsv(src, num);
	}
	system("pause");
	of.close();
	
	return 0;
}