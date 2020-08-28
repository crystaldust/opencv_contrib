// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_UTILS_HPP__
#define __OPENCV_MCC_UTILS_HPP__

#include <functional>
#include <vector>
#include <string>
#include <iostream>

namespace cv 
{
namespace ccm 
{

template<typename F>
cv::Mat elementWise(const cv::Mat& src, F&& lambda) 
{
    cv::Mat dst = src.clone();
    const int channel = src.channels();
    switch (channel)
    {
    case 1:
    {
        cv::MatIterator_<double> it, end;
        for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it)
        {
            (*it) = lambda((*it));
        }
        break;
    }
    case 3:
    {
        cv::MatIterator_<cv::Vec3d> it, end;
        for (it = dst.begin<cv::Vec3d>(), end = dst.end<cv::Vec3d>(); it != end; ++it) 
        {
            for (int j = 0; j < 3; j++) 
            {
                (*it)[j] = lambda((*it)[j]);
            }
        }
        break;
    }
    default:
        throw std::invalid_argument { "Wrong channel!" };
        break;
    }
    return dst;
}

template<typename F>
cv::Mat channelWise(const cv::Mat& src, F&& lambda) 
{
    cv::Mat dst = src.clone();
    cv::MatIterator_<cv::Vec3d> it, end;
    for (it = dst.begin<cv::Vec3d>(), end = dst.end<cv::Vec3d>(); it != end; ++it) 
    {
        *it = lambda(*it);
    }
    return dst;
}

template<typename F>
cv::Mat distanceWise(cv::Mat& src, cv::Mat& ref, F&& lambda) 
{
    cv::Mat dst = cv::Mat(src.size(), CV_64FC1);
    cv::MatIterator_<cv::Vec3d> it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>(),
        it_ref = ref.begin<cv::Vec3d>(), end_ref = ref.end<cv::Vec3d>();
    cv::MatIterator_<double> it_dst = dst.begin<double>(), end_dst = dst.end<double>();
    for (; it_src != end_src; ++it_src, ++it_ref, ++it_dst) 
    {
        *it_dst = lambda(*it_src, *it_ref);
    }
    return dst;
}

/* gamma correction; see ColorSpace.pdf for details; */
double gammaCorrection_(const double& element, const double& gamma) 
{
    return (element >= 0 ? pow(element, gamma) : -pow((-element), gamma));
}

cv::Mat gammaCorrection(const cv::Mat& src, const double& gamma) 
{
    return elementWise(src, [gamma](double element)->double {return gammaCorrection_(element, gamma); });
}

cv::Mat maskCopyTo(const cv::Mat& src, const cv::Mat& mask) 
{
    cv::Mat dst(countNonZero(mask), 1, src.type());
    const int channel = src.channels();
    auto it_mask = mask.begin<uchar>(), end_mask = mask.end<uchar>();
    switch (channel)
    {
    case 1:
    {
        auto it_src = src.begin<double>(), end_src = src.end<double>();
        auto it_dst = dst.begin<double>(), end_dst = dst.end<double>();
        for (; it_src != end_src; ++it_src, ++it_mask) 
        {
            if (*it_mask) 
            {
                (*it_dst) = (*it_src);
                ++it_dst;
            }
        }
        break;
    }
    case 3:
    {
        auto it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>();
        auto it_dst = dst.begin<cv::Vec3d>(), end_dst = dst.end<cv::Vec3d>();
        for (; it_src != end_src; ++it_src, ++it_mask) 
        {
            if (*it_mask) 
            {
                (*it_dst) = (*it_src);
                ++it_dst;
            }
        }
        break;
    }
    default:
        throw std::invalid_argument { "Wrong channel!" };
        break;
    }
    return dst;
}

cv::Mat multiple(const cv::Mat& xyz, const cv::Mat& ccm) 
{
    cv::Mat tmp = xyz.reshape(1, xyz.rows * xyz.cols);
    cv::Mat res = tmp * ccm;
    res = res.reshape(res.cols, xyz.rows);
    return res;
}

/* return the mask of unsaturated colors */
cv::Mat saturate(cv::Mat& src, const double& low, const double& up) 
{
    cv::Mat dst = cv::Mat::ones(src.size(), CV_8UC1);
    cv::MatIterator_<cv::Vec3d> it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>();
    cv::MatIterator_<uchar> it_dst = dst.begin<uchar>(), end_dst = dst.end<uchar>();
    for (; it_src != end_src; ++it_src, ++it_dst) 
    {
        for (int i = 0; i < 3; ++i) 
        {
            if ((*it_src)[i] > up || (*it_src)[i] < low) 
            {
                *it_dst = 0.;
                break;
            }
        }
    }
    return dst;
}

const static cv::Mat m_gray = (cv::Mat_<double>(3, 1) << 0.2126, 0.7152, 0.0722);

/* it is an approximation grayscale function for relative RGB color space;
    see Miscellaneous.pdf for details; */
cv::Mat rgb2gray(cv::Mat rgb) 
{
    return multiple(rgb, m_gray);
}

} // namespace ccm
} // namespace cv


#endif