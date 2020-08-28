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

#ifndef __OPENCV_MCC_CCM_HPP__
#define __OPENCV_MCC_CCM_HPP__

#include<iostream>
#include<cmath>
#include<string>
#include<vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/mcc/linearize.hpp"

namespace cv 
{
namespace ccm 
{

enum CCM_TYPE 
{
    CCM_3x3,
    CCM_4x3
};

enum INITIAL_METHOD_TYPE 
{
    WHITE_BALANCE,
    LEAST_SQUARE
};

/* After being called, the method produce a ColorCorrectionModel instance for inference.*/
class ColorCorrectionModel
{
public:
    // detected colors, the referenceand the RGB colorspace for conversion
    cv::Mat src;
    Color dst;
    CCM_TYPE ccm_type;
    int shape;

    // linear method
    RGBBase_& cs;
    std::shared_ptr<Linear> linear;
    DISTANCE_TYPE distance;

    // weights 
    cv::Mat weights;
    cv::Mat ccm;
    cv::Mat ccm0;

    int max_count;
    double epsilon;

    ColorCorrectionModel(cv::Mat src, Color dst, RGBBase_& cs, CCM_TYPE ccm_type, DISTANCE_TYPE distance,
        LINEAR_TYPE linear, double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list,
        double weights_coeff, INITIAL_METHOD_TYPE initial_method_type, int max_count, double epsilon) :
        src(src), dst(dst), cs(cs), ccm_type(ccm_type), distance(distance), max_count(max_count), epsilon(epsilon) 
        {
        cv::Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
        this->linear = getLinear(gamma, deg, this->src, this->dst, saturate_mask, this->cs, linear);
        calWeightsMasks(weights_list, weights_coeff, saturate_mask);

        src_rgbl = this->linear->linearize(maskCopyTo(this->src, mask));
        this->dst = this->dst[mask];
        dst_rgbl = maskCopyTo(this->dst.to(*(this->cs.l)).colors, mask);

        // empty for CCM_3x3, not empty for CCM_4x3
        src_rgbl = prepare(src_rgbl);

        // distance function may affect the loss function and the fitting function
        switch (this->distance)
        {
        case cv::ccm::RGBL:
            initialLeastSquare(true);
            break;
        default:
            switch (initial_method_type)
            {
            case cv::ccm::WHITE_BALANCE:
                initialWhiteBalance();
                break;
            case cv::ccm::LEAST_SQUARE:
                initialLeastSquare();
                break;
            default:
                throw std::invalid_argument{ "Wrong initial_methoddistance_type!" };
                break;
            }
            break;
        }

        fitting();
    }

    // make no change for ColorCorrectionModel_3x3 class
    // convert matrix A to [A, 1] in ColorCorrectionModel_4x3 class
    cv::Mat prepare(const cv::Mat& inp) 
    {
        switch (ccm_type)
        {
        case cv::ccm::CCM_3x3:
            shape = 9;
            return inp;
        case cv::ccm::CCM_4x3:
        {
            shape = 12;
            cv::Mat arr1 = cv::Mat::ones(inp.size(), CV_64F);
            cv::Mat arr_out(inp.size(), CV_64FC4);
            cv::Mat arr_channels[3];
            split(inp, arr_channels);
            merge(std::vector<Mat>{ arr_channels[0], arr_channels[1], arr_channels[2], arr1 }, arr_out);
            return arr_out;
        }
        default:
            throw std::invalid_argument{ "Wrong ccm_type!" };
            break;
        }
    };

    // fitting nonlinear - optimization initial value by white balance :
    // res = diag(mean(s_r) / mean(d_r), mean(s_g) / mean(d_g), mean(s_b) / mean(d_b))
    // see CCM.pdf for details;
    cv::Mat initialWhiteBalance(void) 
    {
        cv::Mat schannels[3];
        split(src_rgbl, schannels);
        cv::Mat dchannels[3];
        split(dst_rgbl, dchannels);
        std::vector <double> initial_vec = { sum(dchannels[0])[0] / sum(schannels[0])[0], 0, 0, 0,
            sum(dchannels[1])[0] / sum(schannels[1])[0], 0, 0, 0, sum(dchannels[2])[0] / sum(schannels[2])[0], 0, 0, 0 };
        std::vector <double> initial_vec_(initial_vec.begin(), initial_vec.begin() + shape);
        cv::Mat initial_white_balance = cv::Mat(initial_vec_, true).reshape(0, shape / 3);

        return initial_white_balance;
    };

    // fitting nonlinear-optimization initial value by least square:
    // res = np.linalg.lstsq(src_rgbl, dst_rgbl)
    // see CCM.pdf for details;
    // if fit==True, return optimalization for rgbl distance function;

    void initialLeastSquare(bool fit = false) 
    {
        cv::Mat A, B, w;
        if (weights.empty()) 
        {
            A = src_rgbl;
            B = dst_rgbl;
        }

        else 
        {
            pow(weights, 0.5, w);
            cv::Mat w_;
            merge(std::vector<Mat>{ w, w, w }, w_);
            A = w_.mul(src_rgbl);
            B = w_.mul(dst_rgbl);
        }

        solve(A.reshape(1, A.rows), B.reshape(1, B.rows), ccm0, DECOMP_SVD);

        if (fit) 
        {
            ccm = ccm0;
            cv::Mat residual = A.reshape(1, A.rows) * ccm.reshape(0, shape / 3) - B.reshape(1, B.rows);
            Scalar s = residual.dot(residual);
            double sum = s[0];
            loss = sqrt(sum / masked_len);
        }
    };


    class LossFunction : public cv::MinProblemSolver::Function 
    {
    public:
        ColorCorrectionModel* ccm_loss;
        LossFunction(ColorCorrectionModel* ccm) : ccm_loss(ccm) {};

        int getDims() const 
        {
            return ccm_loss->shape;
        }

        double calc(const double* x) const 
        {
            cv::Mat ccm(ccm_loss->shape, 1, CV_64F);
            for (int i = 0; i < ccm_loss->shape; i++) 
            {
                ccm.at<double>(i, 0) = x[i];
            }
            ccm = ccm.reshape(0, ccm_loss->shape / 3);
            Mat reshapecolor = ccm_loss->src_rgbl.reshape(1, 0) * ccm;
            cv::Mat dist = Color(reshapecolor.reshape(3, 0), ccm_loss->cs).diff(ccm_loss->dst, ccm_loss->distance);
            cv::Mat dist_;
            pow(dist, 2, dist_);
            if (!ccm_loss->weights.empty()) 
            {
                dist_ = ccm_loss->weights.mul(dist_);
            }
            Scalar ss = sum(dist_);
            return ss[0];
        }
    };

    // fitting ccm if distance function is associated with CIE Lab color space
    void fitting(void) 
    {
        cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
        cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
        solver->setFunction(ptr_F);
        cv::Mat reshapeccm = ccm0.reshape(0, 1);
        cv::Mat step = cv::Mat::ones(reshapeccm.size(), CV_64F);
        solver->setInitStep(step * 10);
        /* TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_count, epsilon);
            solver->setTermCriteria(termcrit);*/
        double res = solver->minimize(reshapeccm);
        ccm = reshapeccm.reshape(0, shape);
        double loss = pow((res / masked_len), 0.5);
        //std::cout << " ccm " << ccm << std::endl;
        //std::cout << " loss " << loss << std::endl;
    };

    cv::Mat infer(const cv::Mat& img, bool islinear = false) 
    {
        if (!ccm.data)
        {
            throw "No CCM values!";
        }
        cv::Mat img_lin = linear->linearize(img);
        cv::Mat img_ccm(img_lin.size(), img_lin.type());
        cv::Mat ccm_ = ccm.reshape(0, shape / 3);
        img_ccm = multiple(prepare(img_lin), ccm_);
        if (islinear == true) 
        {
            return img_ccm;
        }
        return cs.fromL(img_ccm);
    };

    // infer image and output as an BGR image with uint8 type
    // mainly for test or debug!
    cv::Mat inferImage(std::string imgfile, bool islinear = false) 
    {
        const int inp_size = 255;
        const int out_size = 255;
        cv::Mat img = imread(imgfile);
        cv::Mat img_;
        cvtColor(img, img_, COLOR_BGR2RGB);
        img_.convertTo(img_, CV_64F);
        img_ = img_ / inp_size;
        cv::Mat out = this->infer(img_, islinear);
        cv::Mat out_ = out * out_size;
        out_.convertTo(out_, CV_8UC3);
        cv::Mat img_out = min(max(out_, 0), out_size);
        cv::Mat out_img;
        cvtColor(img_out, out_img, COLOR_RGB2BGR);
        return out_img;
    };

private:
    cv::Mat mask;
    cv::Mat dist;
    int masked_len;
    double loss;

    // RGBl of detected data and the reference
    cv::Mat src_rgbl;
    cv::Mat dst_rgbl;

    // calculate weights and mask
    void calWeightsMasks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask) 
    {
        // weights
        if (!weights_list.empty()) 
        {
            weights = weights_list;
        }
        else if (weights_coeff != 0) 
        {
            pow(dst.toLuminant(dst.cs.io), weights_coeff, weights);
        }

        // masks
        cv::Mat weight_mask = cv::Mat::ones(src.rows, 1, CV_8U);
        if (!weights.empty()) 
        {
            weight_mask = weights > 0;
        }
        this->mask = (weight_mask) & (saturate_mask);

        // weights' mask
        if (!weights.empty()) 
        {
            cv::Mat weights_masked = maskCopyTo(this->weights, this->mask);
            weights = weights_masked / mean(weights_masked);
        }
        masked_len = sum(mask)[0];
    };
};

} // namespace ccm
} // namespace cv

#endif