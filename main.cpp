#include "Harmonious.h"
#include <opencv.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

ofstream of;
int main(int argc, char* argv[]) {

	of.open("result.pdf");

	for (int num = 1; num <= 89; num++){

		cout << "-----------" << num << "-----------" << endl;
		of << "-----------" << num << "-----------" << endl;
		string path = "pic/raw_" + to_string(num) + ".jpg";

		//double hue_score = seg_by_lab(path, num);
		//double value_score = seg_by_gray(path, num);
		double saturation_score = seg_by_hsv(path, num);
		//cout << hue_score << "  " << value_score << "  " << saturation_score << "  " << (saturation_score + hue_score + value_score)/3 << endl;
	}
	of.close();

	return 0;
}