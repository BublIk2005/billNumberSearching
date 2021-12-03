#pragma once
#ifndef BILL_NUMBER_SEARCHING_H
#define BILL_NUMBER_SEARCHING_H
#include<opencv2/opencv.hpp>
#include<math.h>

/// <summary>
///	Функция позволяющая вывести изображение на экран
/// </summary>
/// <param name="src">Исходное изображение</param>
/// <param name="window_name">Имя окна</param>
 void print(cv::Mat src, std::string window_name);

cv::Mat sharp(cv::Mat src);

/// @brief Функция предназначенная для сортировки контуров по площади
/// @param contours Входное множество точек контуров
/// @return множество точек контуров отсортированное по уменьшению площади
std::vector<std::vector<cv::Point>>  sortContours(std::vector<std::vector<cv::Point>>& contours);
/// @brief Функция предназначенная для поворота изображения
/// @param img Входное изображение
/// @param a Точка А линии относительно которой происходит поворот
/// @param b Точка B линии относительно которой происходит поворот
/// @param rect_points Массив вершин прямоугольного контура
/// @return Повернутое изображение
cv::Mat Rotation(cv::Mat img, cv::Point2f a, cv::Point2f b, cv::Point2f* rect_points);

/// @brief Функция находящая растояние между двумя точками
/// @param a Точка A
/// @param b Точка B
/// @return Расстояние между двумя точками
double lineLenght(cv::Point2f a, cv::Point2f b);
/// @brief Функция изменяющая размер прямоугольного контура для переноса на изображение с измененным масштабом
/// @param img Изображение с измененным масштабом на которое нужно перенести прямоугольный контур
/// @param rect_points Массив вершин прямоугольного контура
/// @param scale Новый масштаб к которому нужно привести размеры контура
/// @param modelSizeCols Ширина изначального изображения
/// @param modelSizeRows Высота изначального изображения
void contoursScaler(cv::Mat img, cv::Point2f* rect_points, double modelSizeCols, double modelSizeRows);

/// @brief Функция позволяющая изменять координаты точек с учетом размеров изображения
/// @param img Изображение на котором находится точка
/// @param coord Координата точки
/// @param value Значение которое мы прибавляем или отнимаем
/// @param PlusOrMinus 0 если вычитаем, 1 если складываем
/// @return Новая координата
int ChangeCoordX(cv::Mat img, int coord, int value, int PlusOrMinus);

/// @brief  Функция для поиска купюры на изображении и её нормализации
/// @param src Изображение на котором ищем купюру
/// @param dst Область интереса на которой нашли купюру
void search_normalization(const cv::Mat& src, cv::Mat& dst);
/// @brief Функция для поиск серийного номера купюры на изображении
/// @param src Исходное изображение
/// @param norm Нормализованное изображение найденной купюры
/// @param results Вектор прямоугольников локализирующих области в которых находится серийный номер купюры
void billNumSerch(const cv::Mat& src, cv::Mat& norm, std::vector<cv::Rect>& results);


#endif