#ifndef __OPENCV_MCC_UTILS_HPP__
#define __OPENCV_MCC_UTILS_HPP__

#include <functional>
#include <vector>
#include <string>
#include <iostream>

namespace cv {
    namespace ccm {

        template<typename F>
        cv::Mat _elementwise(cv::Mat src, F&& lambda) {
            cv::Mat dst = src.clone();
            const int channel = src.channels();
            switch (channel)
            {
            case 1:
            {
                cv::MatIterator_<double> it, end;
                for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it)
                    (*it) = lambda((*it));
                break;
            }
            case 3:
            {
                cv::MatIterator_<Vec3d> it, end;
                for (it = dst.begin<Vec3d>(), end = dst.end<Vec3d>(); it != end; ++it) {
                    for (int j = 0; j < 3; j++) {
                        (*it)[j] = lambda((*it)[j]);
                    }
                }
                break;
            }
            default:
                throw;//todo
                break;
            }
            return dst;
        }
     
        template<typename F>
        cv::Mat _channelwise(cv::Mat src, F&& lambda) {
            cv::Mat dst = src.clone();
            cv::MatIterator_<Vec3d> it, end;
            for (it = dst.begin<Vec3d>(), end = dst.end<Vec3d>(); it != end; ++it) {
                *it = lambda(*it);
            }
            return dst;
        }

        template<typename F>
        cv::Mat _distancewise(cv::Mat src, cv::Mat ref, F&& lambda) {
            cv::Mat dst = cv::Mat(src.size(), CV_64FC1);
            cv::MatIterator_<Vec3d> it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>(),
                it_ref = ref.begin<Vec3d>(), end_ref = ref.end<Vec3d>();
            cv::MatIterator_<double> it_dst = dst.begin<double>(), end_dst = dst.end<double>();
            for (; it_src != end_src; ++it_src, ++it_ref, ++it_dst) {
                *it_dst = lambda(*it_src, *it_ref);
            }
            return dst;
        }

        /* gamma correction; see ColorSpace.pdf for details; */
        double _gamma_correction(double element, double gamma) {
            return (element >= 0 ? pow(element, gamma) : -pow((-element), gamma));
        }

        cv::Mat gamma_correction(cv::Mat src, double gamma) {
            return _elementwise(src, [gamma](double element)->double {return _gamma_correction(element, gamma); });
        }

        cv::Mat mask_copyto(cv::Mat src, cv::Mat mask) {
            cv::Mat dst(countNonZero(mask), 1, src.type());
            const int channel = src.channels();
            auto it_mask = mask.begin<uchar>(), end_mask = mask.end<uchar>();
            switch (channel)
            {
            case 1:
            {
                auto it_src = src.begin<double>(), end_src = src.end<double>();
                auto it_dst = dst.begin<double>(), end_dst = dst.end<double>();
                for (; it_src != end_src; it_src++, it_mask++) {
                    if (*it_mask) {
                        (*it_dst) = (*it_src);
                        ++it_dst;
                    }
                }
                break;
            }
            case 3:
            {
                auto it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>();
                auto it_dst = dst.begin<Vec3d>(), end_dst = dst.end<Vec3d>();
                for (; it_src != end_src; it_src++, it_mask++) {
                    if (*it_mask) {
                        (*it_dst) = (*it_src);
                        ++it_dst;
                    }
                }
            }
            }
            //todo
            return dst;
        }

        cv::Mat multiple(cv::Mat xyz, cv::Mat ccm) {
            cv::Mat tmp = xyz.reshape(1, xyz.rows * xyz.cols);
            cv::Mat res = tmp * ccm;
            res = res.reshape(res.cols, xyz.rows);
            return res;
        }

        /* return the mask of unsaturated colors */
        cv::Mat saturate(cv::Mat src, double low, double up) {
            cv::Mat dst = cv::Mat::ones(src.size(), CV_8UC1);
            cv::MatIterator_<Vec3d> it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>();
            cv::MatIterator_<uchar> it_dst = dst.begin<uchar>(), end_dst = dst.end<uchar>();
            for (; it_src != end_src; ++it_src, ++it_dst) {
                for (int i = 0; i < 3; ++i) {
                    if ((*it_src)[i] > up || (*it_src)[i] < low) {
                        *it_dst = 0.;
                        break;
                    }
                }
            }
            return dst;
        }

        const cv::Mat M_gray = (cv::Mat_<double>(3, 1) << 0.2126, 0.7152, 0.0722);

        /* it is an approximation grayscale function for relative RGB color space;
           see Miscellaneous.pdf for details; */
        cv::Mat rgb2gray(cv::Mat rgb) {
            return multiple(rgb, M_gray);
        }
    }
}


#endif