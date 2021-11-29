#include "..\..\barcode_detecting\lib\barcode_detecting.h"
#include"billNumberSearching.h"

using namespace cv;
using namespace std;

inline void print(cv::Mat src, std::string window_name)
{
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, 600, 400);
	imshow(window_name, src);
}

Mat sharp(Mat src)
{
	Mat result;
	Mat kernel=(Mat_<int>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
	filter2D(src, result, -1, kernel);
	return result;
}

std::vector<std::vector<cv::Point>>  sortContours(std::vector<std::vector<cv::Point>>& contours)
{
	for (size_t i = 0; i < contours.size() - 1; i++) {
		for (size_t ind = 0; ind < contours.size() - 1; ind++)
		{
			if (cv::contourArea(contours[ind]) > cv::contourArea(contours[ind + 1]))
			{
				std::vector<cv::Point> tmp;
				tmp = contours[ind];
				contours[ind] = contours[ind + 1];
				contours[ind + 1] = tmp;
			}
		}
	}
	return contours;
}
cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points)//Поворт изображения происходит относительн двух линий: Левого края изображения и отрезка заданного двумя точками
{
	//Координаты вектора с помощью которого будет производится поворот
	cv::Point2f vec = b - a;
	//Левый край изображения
	cv::Point2f dir(5, 0);
	//Косинус угла поворота
	float cos = (vec.x * dir.x + vec.y * dir.y) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	//Синус угла поворота
	float sin = (vec.x * dir.y - vec.y * dir.x) / (sqrt(vec.x * vec.x + vec.y * vec.y) * sqrt(dir.x * dir.x + dir.y * dir.y));
	//Угол поворота в градусах
	float angle = 57.3 * acosf(cos);
	
	//Проверка на пространственное расположение векторов относительно друг друга и изменение угла для избежания поворота на 180 градусов 
	if (sin < 0 && cos < 0 || cos>0 && sin > 0)
	{
		angle = -angle;
	}
	
	//Центр изображения
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	//Матрица поворота 
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	//Новые размеры при поворте
	cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
	rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
	rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;
	//Результат работы функции
	cv::Mat result;
	//Поворот изображения
	cv::warpAffine(img, result, rot, bbox.size());
	//Новая координата контура
	cv::Point2f newPoint;
	//Для каждой вершины контура находим новую координата для повернутого изображения
	for (int i = 0; i < 4; ++i) {
		newPoint.x = (rect_points[i].x - center.x) * cos - (rect_points[i].y - center.y) * sin + center.x * (result.cols / (float)img.cols);
		newPoint.y = (rect_points[i].x - center.x) * sin + (rect_points[i].y - center.y) * cos + center.y * (result.rows / (float)img.rows);

		rect_points[i] = newPoint;
	}
	return result;
}

double lineLenght(cv::Point2f a, cv::Point2f b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
void contoursScaler(cv::Mat img, cv::Point2f* rect_points, double modelSizeCols,double modelSizeRows)
{
	//Сравниваем значения ширины изображения в масштабе на которой мы хотим перенести контур
	//и изображения в масштабе на котором мы нашли котур
	//Меняем координаты вершин контура под новый масштаб
	double scaleX = img.cols / modelSizeCols;
	double scaleY = img.rows / modelSizeRows;
	if (img.cols > modelSizeCols) {
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i].x = int(rect_points[i].x * scaleX);
			rect_points[i].y = int(rect_points[i].y * scaleY);
		}
	}
	else if (modelSizeCols > img.cols) {
		for (size_t i = 0; i < 4; i++)
		{
			rect_points[i].x = int(rect_points[i].x / scaleX);
			rect_points[i].y = int(rect_points[i].y / scaleY);
		}
	}
}

int ChangeCoordX(Mat img, int coord, int value, int PlusOrMinus)
{
	
	if (PlusOrMinus == 0 )
	{
		coord = coord - value;
		if(coord < 0)
			coord = 0;
	}
	if (PlusOrMinus == 1)
	{
		coord = coord + value;
		if(coord  > img.cols)
			coord = img.cols-5;
	}
	
	return coord;
}

bool checkCont(std::vector<std::vector<cv::Point>>& contours)
{
	sortContours(contours);
	if (contourArea(contours[contours.size() - 1]) / 100 > contourArea(contours[0]))
		return false;
	else return true;

}
using namespace cv;
using namespace std;
int main()
{
	string path = "E:/Projects/billNumberSearching/data/test10.jpg"; 
	Mat large = cv::imread(path, cv::IMREAD_COLOR);
	print(large, "Source");
	Mat rgb;
	
	//pyrDown(large, rgb);
	resize(large, rgb, Size(800, 400));
	
	Mat small;
	cvtColor(rgb, small, COLOR_BGR2GRAY);
	//print(small, "Gray");
	sharp(small);
	Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	//morphologyEx(small, small, MORPH_GRADIENT, morphKernel);
	
	medianBlur(small, small, 5);
	//print(small, "gradmo");
	
	Mat grad;
	threshold(small, grad, 100, 255, THRESH_OTSU);
	grad = ~grad;
	morphKernel = getStructuringElement(MORPH_RECT, Size(9,9));
	morphologyEx(grad, grad, MORPH_CLOSE, morphKernel);
	vector<vector<Point>> cont;
	findContours(grad, cont, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	
	/*for (size_t i = 0; i < cont.size(); i++)
	{
		drawContours(rgb, cont, i, Scalar(0, 255, 0), -1);
	}*/
	sortContours(cont);
	
	
	//print(grad, "grad");
	double eps = 30;
	//approxPolyDP(cont[cont.size() - 1], cont[cont.size() - 1], eps, false);
	drawContours(rgb, cont, cont.size() - 1, Scalar(255, 0, 0), -1);
	RotatedRect rectCont = minAreaRect(cont[cont.size() - 1]);
	Point2f rect_points[4];
	rectCont.points(rect_points);
	for (int j = 0; j < 4; j++)
	{
		line(rgb, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 3);
	}
	//print(grad, "Grad");
	print(rgb, "Contours");
	Point2f a, b;
	contoursScaler(large, rect_points, rgb.cols, rgb.rows);
	if (lineLenght(rect_points[0], rect_points[1]) > lineLenght(rect_points[1], rect_points[2]))
	{
		a = rect_points[0];
		b = rect_points[1];
	}
	else {
		a = rect_points[1];
		b = rect_points[2];
	}
	if (a.x > b.x)
	{
		std::swap(a, b);
	}
	
	rgb=Rotation(large, a, b, rect_points);
	
	if (lineLenght(rect_points[0], rect_points[1]) > lineLenght(rect_points[1], rect_points[2])) {
		if (rect_points[0].x < rect_points[1].x) {
			rect_points[0].x = ChangeCoordX(rgb, rect_points[0].x, lineLenght(rect_points[0], rect_points[1]) /2 , 0);
			rect_points[1].x = ChangeCoordX(rgb, rect_points[1].x, lineLenght(rect_points[0], rect_points[1]) / 2, 1);
			rect_points[2].x = rect_points[1].x;
			rect_points[3].x = rect_points[0].x;
		}
		else {
			rect_points[0].x = ChangeCoordX(rgb, rect_points[0].x, lineLenght(rect_points[0], rect_points[1]) / 2, 1);
			rect_points[1].x = ChangeCoordX(rgb, rect_points[1].x, lineLenght(rect_points[0], rect_points[1]) / 2, 0);
			rect_points[2].x = rect_points[1].x;
			rect_points[3].x = rect_points[0].x;
		}
	}
	else {
		if (rect_points[1].x < rect_points[2].x) {
			rect_points[1].x = ChangeCoordX(rgb, rect_points[1].x, lineLenght(rect_points[1], rect_points[2]) / 3, 0);
			rect_points[2].x = ChangeCoordX(rgb, rect_points[2].x, lineLenght(rect_points[1], rect_points[2]) / 3, 1);
			rect_points[3].x = rect_points[2].x;
			rect_points[0].x = rect_points[1].x;
		}
		else {
			rect_points[1].x = ChangeCoordX(rgb, rect_points[0].x, lineLenght(rect_points[1], rect_points[2]) / 3, 1);
			rect_points[2].x = ChangeCoordX(rgb, rect_points[0].x, lineLenght(rect_points[1], rect_points[2]) / 3, 0);
			rect_points[3].x = rect_points[2].x;
			rect_points[0].x = rect_points[1].x;
		}
	}
	
	for (int j = 0; j < 4; j++)
	{
		line(rgb, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 3);
	}
//	print(rgb, "Rotate");
	Rect roi(rect_points[0], rect_points[2]);
	rgb = rgb(roi);
//	print(rgb, "Roi");

	large = rgb;
	resize(rgb, rgb, Size(600, 400));
	cvtColor(rgb, small, COLOR_BGR2GRAY);
	morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
	Mat bw;
	threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
	//print(bw, "Binary");
	
	Mat connected;
	morphKernel = getStructuringElement(MORPH_RECT, Size(8, 2));
	morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
	print(connected, "Connected");
	
	Mat mask = Mat::zeros(bw.size(), CV_8UC1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Rect> samples;
	
	
	Rect rect;
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		Mat sampleMat;
		vector<vector<Point>> cont;
		RotatedRect RectBound = minAreaRect(contours[idx]);
		rect = boundingRect(contours[idx]);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		
		drawContours(mask, contours, idx, Scalar(255, 255, 255), FILLED);
		
		double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

		if (r > .45 
			&&
			(rect.height > 8 && rect.width > 8 && rect.width >  2*rect.height || RectBound.size.width>RectBound.size.height)
			)
		{
			sampleMat = small(rect);
			resize(sampleMat, sampleMat, Size(100, 100));
			threshold(sampleMat, sampleMat, 100, 255, THRESH_OTSU);
			morphKernel = getStructuringElement(MORPH_RECT, Size(1, 15));
			morphologyEx(sampleMat, sampleMat, MORPH_OPEN, morphKernel);
			sampleMat = ~sampleMat;
			vector<Vec4i> hierarchy;
			
			findContours(sampleMat, cont, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);
			cvtColor(sampleMat, sampleMat, COLOR_GRAY2BGR);
			for (size_t i = 0; i < cont.size(); i++)
			{
				drawContours(sampleMat, cont, i, Scalar(0, 255, 0), -1);
			}
			//print(sampleMat, "Sample" + to_string(idx));
			cout << cont.size() << endl;
			if (cont.size() > 6 && cont.size() < 10)
			{
				cout << "Sample" + to_string(idx) << " " << cont.size() << endl;
				rect.x = rect.x * large.cols / rgb.cols;
				rect.y = rect.y * large.rows / rgb.rows;
				rect.height = rect.height * large.rows / rgb.rows;
				rect.width = rect.width * large.cols / rgb.cols;
				rectangle(large, rect, Scalar(0, 255, 0), 6);
			}
		}
	}
	print(large, "result");
	cv::waitKey();
	return 0;
}