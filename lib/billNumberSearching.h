#pragma once
#ifndef BILL_NUMBER_SEARCHING_H
#define BILL_NUMBER_SEARCHING_H
#include<opencv2/opencv.hpp>
#include<math.h>

/// <summary>
///	������� ����������� ������� ����������� �� �����
/// </summary>
/// <param name="src">�������� �����������</param>
/// <param name="window_name">��� ����</param>
 void print(cv::Mat src, std::string window_name);

cv::Mat sharp(cv::Mat src);

/// @brief ������� ��������������� ��� ���������� �������� �� �������
/// @param contours ������� ��������� ����� ��������
/// @return ��������� ����� �������� ��������������� �� ���������� �������
std::vector<std::vector<cv::Point>>  sortContours(std::vector<std::vector<cv::Point>>& contours);
/// @brief ������� ��������������� ��� �������� �����������
/// @param img ������� �����������
/// @param a ����� � ����� ������������ ������� ���������� �������
/// @param b ����� B ����� ������������ ������� ���������� �������
/// @param rect_points ������ ������ �������������� �������
/// @return ���������� �����������
cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points);

/// @brief ������� ��������� ��������� ����� ����� �������
/// @param a ����� A
/// @param b ����� B
/// @return ���������� ����� ����� �������
double lineLenght(cv::Point2f a, cv::Point2f b);
/// @brief ������� ���������� ������ �������������� ������� ��� �������� �� ����������� � ���������� ���������
/// @param img ����������� � ���������� ��������� �� ������� ����� ��������� ������������� ������
/// @param rect_points ������ ������ �������������� �������
/// @param scale ����� ������� � �������� ����� �������� ������� �������
/// @param modelSizeCols ������ ������������ �����������
/// @param modelSizeRows ������ ������������ �����������
void contoursScaler(cv::Mat img, cv::Point2f* rect_points, double modelSizeCols, double modelSizeRows);

/// @brief ������� ����������� �������� ���������� ����� � ������ �������� �����������
/// @param img ����������� �� ������� ��������� �����
/// @param coord ���������� �����
/// @param value �������� ������� �� ���������� ��� ��������
/// @param PlusOrMinus 0 ���� ��������, 1 ���� ����������
/// @return ����� ����������
int ChangeCoordX(cv::Mat img, int coord, int value, int PlusOrMinus);

/// @brief  ������� ��� ������ ������ �� ����������� � � ������������
/// @param src ����������� �� ������� ���� ������
/// @param dst ������� �������� �� ������� ����� ������
void search_normalization(const cv::Mat& src, cv::Mat& dst);
/// @brief ������� ��� ����� ��������� ������ ������ �� �����������
/// @param src �������� �����������
/// @param norm ��������������� ����������� ��������� ������
/// @param results ������ ��������������� �������������� ������� � ������� ��������� �������� ����� ������
void billNumSerch(const cv::Mat& src, cv::Mat& norm, std::vector<cv::Rect>& results);


#endif