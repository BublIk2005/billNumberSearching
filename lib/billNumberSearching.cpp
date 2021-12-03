#include"billNumberSearching.h"


 void print(cv::Mat src, std::string window_name)
{
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, 600, 400);
	imshow(window_name, src);
}

cv::Mat sharp(cv::Mat src)
{
	cv::Mat result;
	cv::Mat kernel=(cv::Mat_<int>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
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

int ChangeCoordX(cv::Mat img, int coord, int value, int PlusOrMinus)
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



void search_normalization(const cv::Mat& src, cv::Mat& dst)
{
	cv::Mat rgb;
	cv::resize(src, rgb, cv::Size(800, 400));

	cv::Mat small;
	cvtColor(rgb, small, cv::COLOR_BGR2GRAY);
	cv::Mat morphKernel;

	medianBlur(small, small, 5);

	cv::Mat grad;
	threshold(small, grad, 100, 255, cv::THRESH_OTSU);
	grad = ~grad;

	morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	morphologyEx(grad, grad, cv::MORPH_CLOSE, morphKernel);

	std::vector<std::vector<cv::Point>> cont;
	findContours(grad, cont, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);


	sortContours(cont);

	double eps = 30;

	drawContours(rgb, cont, cont.size() - 1, cv::Scalar(255, 0, 0), -1);
	cv::RotatedRect rectCont = cv::minAreaRect(cont[cont.size() - 1]);
	cv::Point2f rect_points[4];
	rectCont.points(rect_points);
	for (int j = 0; j < 4; j++)
	{
		line(rgb, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0), 3);
	}

	cv::Point2f a, b;
	contoursScaler(src, rect_points, rgb.cols, rgb.rows);
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

	rgb = Rotation(src, a, b, rect_points);


	if (lineLenght(rect_points[0], rect_points[1]) > lineLenght(rect_points[1], rect_points[2])) {
		if (rect_points[0].x < rect_points[1].x) {
			rect_points[0].x = ChangeCoordX(rgb, rect_points[0].x, lineLenght(rect_points[0], rect_points[1]) / 2, 0);
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

	cv::Rect roi(rect_points[0], rect_points[2]);
	dst = rgb(roi);

}
void billNumSerch(const cv::Mat& src, cv::Mat& norm, std::vector<cv::Rect>& results)
{
	cv::Mat large = src;
	cv::Mat rgb, small, grad;
	search_normalization(large, rgb);
	large = rgb;
	
	resize(rgb, rgb, cv::Size(600, 400));
	cvtColor(rgb, small, cv::COLOR_BGR2GRAY);
	
	cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	morphologyEx(small, grad, cv::MORPH_GRADIENT, morphKernel);
	
	cv::Mat bw;
	threshold(grad, bw, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
	

	cv::Mat connected;
	morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 2));
	morphologyEx(bw, connected, cv::MORPH_CLOSE, morphKernel);
	

	cv::Mat mask = cv::Mat::zeros(bw.size(), CV_8UC1);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::vector<cv::Rect> samples;


	cv::Rect rect;
	//std::vector<cv::Rect> results;
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		cv::Mat sampleMat;
		std::vector<std::vector<cv::Point>> cont;
		cv::RotatedRect RectBound = cv::minAreaRect(contours[idx]);
		rect = cv::boundingRect(contours[idx]);
		cv::Mat maskROI(mask, rect);
		maskROI = cv::Scalar(0, 0, 0);

		drawContours(mask, contours, idx, cv::Scalar(255, 255, 255), cv::FILLED);

		double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

		if (r > .45
			&&
			(rect.height > 8 && rect.width > 8 && (rect.width > 2 * rect.height || RectBound.size.width > 2 * RectBound.size.height))
			)
		{
			sampleMat = small(rect);
			resize(sampleMat, sampleMat, cv::Size(100, 100));
			threshold(sampleMat, sampleMat, 100, 255, cv::THRESH_OTSU);
			morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 15));
			morphologyEx(sampleMat, sampleMat, cv::MORPH_OPEN, morphKernel);
			//cv::dilate(sampleMat, sampleMat, morphKernel);
			sampleMat = ~sampleMat;
			std::vector<cv::Vec4i> hierarchy;

			findContours(sampleMat, cont, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
			cvtColor(sampleMat, sampleMat, cv::COLOR_GRAY2BGR);
			for (size_t i = 0; i < cont.size(); i++)
			{
				drawContours(sampleMat, cont, i, cv::Scalar(0, 255, 0), -1);
			}
			//print(sampleMat, "Sample" + std::to_string(idx));
			std::cout << cont.size() << std::endl;
			if (cont.size() > 6 && cont.size() < 12)
			{
				
				rect.x = rect.x * large.cols / rgb.cols;
				rect.y = rect.y * large.rows / rgb.rows;
				rect.height = rect.height * large.rows / rgb.rows;
				rect.width = rect.width * large.cols / rgb.cols;
				//rectangle(large, rect, cv::Scalar(0, 255, 0), 6);
				results.push_back(rect);
			}
		}
	}
	norm = large;
}
