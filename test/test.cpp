#include"billNumberSearching.h"


int main()
{
	std::string pathData = "data/";
	std::string pathResult = "results/";
	std::string test = "test4";
	cv::Mat large = cv::imread(pathData + test + ".jpg", cv::IMREAD_COLOR);
	print(large, "Source");

	std::vector<cv::Rect> results;
	billNumSerch(large, large, results);
	for (auto& item : results)
	{
		rectangle(large, item, cv::Scalar(0, 255, 0), 6);
	}
	print(large, "result");
	//cv::imwrite(pathResult + test + "_res.jpg", large);
	cv::waitKey();
	return 0;
}