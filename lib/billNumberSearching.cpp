#include"billNumberSearching.h"

void print(cv::Mat src, std::string window_name)
{
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, 600, 400);
	imshow(window_name, src);
}

using namespace cv;
using namespace std;
int main()
{
	string path = "E:/Projects/billNumberSearching/data/test2.jpg"; // путь к тесту
	Mat large = cv::imread(path, cv::IMREAD_COLOR);
	print(large, "Source");
	Mat rgb;
	// Уменьшим изображение
	pyrDown(large, rgb);
	std::cout << "large: cols=" << large.cols << " rows=" << large.rows << endl;
	std::cout << "rgb: cols=" << rgb.cols << " rows=" << rgb.rows << endl;
	Mat small;
	cvtColor(rgb, small, COLOR_BGR2GRAY);
	//print(small, "Gray");
	// морфологический градиент
	Mat grad;
	Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
	//print(grad, "Grad");
	// Бинаризация
	Mat bw;
	threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
	//print(bw, "Binary");
	// объединим горизонтально ориентированные регионы
	Mat connected;
	morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
	//print(connected, "Connected");
	// найдем контуры
	Mat mask = Mat::zeros(bw.size(), CV_8UC1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
	// Отфильтруем контуры
	vector<Rect> samples;
	
	Rect rect;
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		Mat sampleMat;
		vector<vector<Point>> cont;
		rect = boundingRect(contours[idx]);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		// Нарисуем контуры
		drawContours(mask, contours, idx, Scalar(255, 255, 255), FILLED);
		// соотношение ненулевых пикселей в заполненной области
		double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

		if (r > .45 /* предположим, что по крайней мере 45 % области заполнено, если она содержит текст*/
			&&
			(rect.height > 8 && rect.width > 8 && rect.width>2* rect.height) /*ограничения на размер региона*/
			)
		{
			sampleMat=small(rect);
			resize(sampleMat, sampleMat, Size(100, 100));
			threshold(sampleMat, sampleMat, 100, 255, THRESH_OTSU);
			morphKernel = getStructuringElement(MORPH_RECT, Size(1, 10));
			morphologyEx(sampleMat, sampleMat, MORPH_OPEN, morphKernel);
			sampleMat = ~sampleMat;
			vector<Vec4i> hierarchy;
			findContours(sampleMat, cont, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);
			if (cont.size() == 7)
			{
				
					rect.x = rect.x * large.cols/rgb.cols;
					rect.y = rect.y * large.rows / rgb.rows;
					rect.height = rect.height * large.rows / rgb.rows;
					rect.width= rect.width* large.cols / rgb.cols;
				rectangle(large, rect, Scalar(0, 255, 0), 2);
			}
		}
	}	
	print(large, "result");
	cv::waitKey();
	return 0;
}