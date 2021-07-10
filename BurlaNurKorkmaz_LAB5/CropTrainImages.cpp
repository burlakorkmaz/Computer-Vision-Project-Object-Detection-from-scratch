#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/filesystem.hpp>

#include <fstream>
#include <vector>
#include <sstream>
#include <map>
using namespace cv;

int divideImage(const cv::Mat& img, const int gridWidth, const int gridHeight, std::vector<cv::Mat>& grids)
{	
	int y0 = 0;
	while (y0 < 256)
	{
		int x0 = 0;
		while (x0 < 256)
		{
			
			// crop block
			grids.push_back(img(cv::Rect(x0, y0, 32, 32)).clone());

			// update x-coordinate
			x0 = x0 + gridWidth;
		}

		// update y-coordinate
		y0 = y0 + gridHeight;
	}
	return EXIT_SUCCESS;
}

std::string readFileIntoString(const std::string& path) {
	auto ss = std::ostringstream{};
	std::ifstream input_file(path);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file - '"
			<< path << "'" << std::endl;
		exit(EXIT_FAILURE);
	}
	ss << input_file.rdbuf();
	return ss.str();
}



int main(int argc, char** argv)
{
	//read label.csv
	std::string filename("banana-detection/bananas_train/label.csv"); 
	std::string file_contents;
	std::map<int, std::vector<std::string>> csv_contents;
	char delimiter = ',';

	file_contents = readFileIntoString(filename);

	std::istringstream sstream(file_contents);
	std::vector<std::string> items;
	std::string record;

	int counter = 0;
	while (std::getline(sstream, record)) {
		std::istringstream line(record);
		while (std::getline(line, record, delimiter)) {
			items.push_back(record);
		}

		csv_contents[counter] = items;
		items.clear();
		counter += 1;
	}

	
	int bananaymin;
	int bananaymax;
	int bananaxmin;
	int bananaxmax;

	// grids coordinates calculation
	int xmins[64];
	int ymins[64];
	int xmaxs[64];
	int ymaxs[64];
	int grid_tracker = 0;

	for (int j = 0; j < 256; j += 32) {
		for (int m = 0; m < 256; m += 32) {
			xmins[grid_tracker] = m;
			xmaxs[grid_tracker] = m + 32;
			ymins[grid_tracker] = j;
			ymaxs[grid_tracker] = j + 32;

			grid_tracker++;
		}
	}

	// Create an output filestream object
	std::ofstream myFile("gridslabel.csv"); 
	// Send data to the stream
	myFile << "gridimage_name,label" << std::endl;

	// init vars
	const int gridw = 32;
	const int gridh = 32;
	int gridCounter = 0;
	cv::utils::fs::createDirectory("GridsFolder"); 
	Mat img;

	for (int a = 0; a < 1000; a++) { 

		std::istringstream(csv_contents[a + 1][2]) >> bananaxmin;
		std::istringstream(csv_contents[a + 1][4]) >> bananaxmax;
		std::istringstream(csv_contents[a + 1][3]) >> bananaymin;
		std::istringstream(csv_contents[a + 1][5]) >> bananaymax;

		std::string image_path = "banana-detection/bananas_train/images/" + std::to_string(a) + ".png"; 
		
		img = imread(image_path, IMREAD_COLOR);
		if (img.empty())
		{
			std::cout << "Could not read the image: " << image_path << std::endl;
			return 1;
		}

		std::vector<cv::Mat> grids;
		std::string gridNames[64];
		// divide image into multiple grids
		int divideStatus = divideImage(img, gridw, gridh, grids);
		// debug: save blocks
		for (int j = 0; j < grids.size(); j++)
		{
			std::string gridImgName = "GridsFolder/" + std::to_string(gridCounter) + ".png"; 
			imwrite(gridImgName, grids[j]);

			gridNames[j] = std::to_string(gridCounter) + ".png"; // array for grid images names 
			gridCounter++;
		}



		// compare label.csv with grids coordinates
		int bananaExistOrNot[64];

		for (int i = 0; i < 64; i++) {

			int intersectionBananaPoints = 0;

			for (int gridymin = ymins[i]; gridymin < ymaxs[i]; gridymin++) {

				for (int gridxmin = xmins[i]; gridxmin < xmaxs[i]; gridxmin++) {

					for (int bananay = bananaymin; bananay < bananaymax; bananay++) {

						for (int bananax = bananaxmin; bananax < bananaxmax; bananax++) {

							if (gridxmin == bananax && gridymin == bananay) {

								intersectionBananaPoints++;

							}
						}
					}
				}

			}
			if (intersectionBananaPoints >= 256) { // 32 * 32 * 0.25
				bananaExistOrNot[i] = 1; //banana exist
				std::cout <<"Image = " + std::to_string(a) + " Grid = " << i << " " << " BANANA" << std::endl;
			}
			else {
				bananaExistOrNot[i] = 0; //banana does not exist
				std::cout << "Image = " + std::to_string(a) + " Grid = " << i << " " << " NOT" << std::endl;
			}
		}


		for (int i = 0; i < 64; i++) {
			myFile << gridNames[i] + "," + std::to_string(bananaExistOrNot[i]) << std::endl;
		}

	}

	// Close the file
	myFile.close();

	return 0;
	exit(EXIT_SUCCESS);
	
	getchar();
}