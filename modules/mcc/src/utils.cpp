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

#include "precomp.hpp"

namespace cv
{
namespace ccm
{

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
    auto it_mask = mask.begin<uchar>();
    switch (channel)
    {
    case 1:
    {
        auto it_src = src.begin<double>(), end_src = src.end<double>();
        auto it_dst = dst.begin<double>();
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
        auto it_dst = dst.begin<cv::Vec3d>();
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

cv::Mat saturate(cv::Mat& src, const double& low, const double& up)
{
    cv::Mat dst = cv::Mat::ones(src.size(), CV_8UC1);
    cv::MatIterator_<cv::Vec3d> it_src = src.begin<cv::Vec3d>(), end_src = src.end<cv::Vec3d>();
    cv::MatIterator_<uchar> it_dst = dst.begin<uchar>();
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

cv::Mat rgb2gray(cv::Mat rgb)
{
    return multiple(rgb, m_gray);
}

} // namespace ccm
} // namespace cv