#pragma once
#ifndef BILL_NUMBER_SEARCHING_H
#define BILL_NUMBER_SEARCHING_H
#include<opencv2/opencv.hpp>
#include<math.h>

/// <summary>
///	‘ункци€ позвол€юща€ вывести изображение на экран
/// </summary>
/// <param name="src">»сходное изображение</param>
/// <param name="window_name">»м€ окна</param>
void print(cv::Mat src, std::string window_name);

cv::Mat filtred(cv::Mat src_bin);

cv::Mat scaler(cv::Mat& src, double& scale, double modelSize);

void contoursScaler(cv::Mat img, cv::Point2f* rect_points, double scale, double modelSize);

std::vector<std::vector<cv::Point>>  sortContours(std::vector<std::vector<cv::Point>>& contours);

double lineLenght(cv::Point2f a, cv::Point2f b);

cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b);

cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points);

cv::Mat Rotation(cv::Mat img, cv::RotatedRect rect, cv::Point2f* rect_points);

cv::Mat Gradient(cv::Mat src_gray);

void contours(cv::Mat& grad, cv::Point2f* rect_points);

bool check(std::vector<int> vecBit);

std::vector<int> countStrokes(const cv::Mat& imeg_bin, int intensiv, int location);

std::vector<int> normalizeVec(std::vector<int> histB, std::vector<int> histW);

std::vector<int> normalizeVecBit(std::vector<int> histB, std::vector<int> histW);

cv::Mat drawCode(cv::Mat& imeg_bin, std::vector<int> normalizeVec);

std::vector<int> decoder(std::vector<int> vecBit);

void findBarcode(cv::Mat src, cv::Point2f* rect_points, double scale);

#endif