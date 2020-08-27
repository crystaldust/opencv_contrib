#ifndef __OPENCV_MCC_OPERATIONS_HPP__
#define __OPENCV_MCC_OPERATIONS_HPP__

#include <functional>
#include <vector>
#include "opencv2/ccm/utils.hpp"

namespace cv {
	namespace ccm {

		typedef std::function<cv::Mat(cv::Mat)> MatFunc;

		class Operation {
		public:
			bool linear;
			cv::Mat M;
			MatFunc f;

			Operation() : linear(true), M(cv::Mat()) {};

			Operation(cv::Mat M) :linear(true), M( M ) {};

			Operation(MatFunc f) : linear(false), f(f) {};

			virtual ~Operation() {};

			cv::Mat operator()(cv::Mat& abc) {
				if (!linear) { return f(abc); }
				if (M.empty()) { return abc; }
				return multiple(abc, M);
			};

			void add(Operation& other) {
				if (M.empty()) {
					M = other.M.clone();
				}
				else {
					M = M * other.M;
				}
			};

			void clear() {
				M = cv::Mat();
		
			};
		};

		const Operation IDENTITY_OP( [](cv::Mat x) {return x; } );

		class Operations {
		public:
			std::vector<Operation> ops;

			Operations() :ops{ } {};

			Operations(std::initializer_list<Operation> op) :ops{ op } {};

			virtual ~Operations() {};

			Operations& add(const Operations& other) {
				ops.insert(ops.end(), other.ops.begin(), other.ops.end());
				return *this;
			};

			cv::Mat run(cv::Mat abc) {
				Operation hd;
				for (auto& op : ops) {
					if (op.linear) {
						hd.add(op);
					}
					else {
						abc = hd(abc);
						hd.clear();
						abc = op(abc);
					}
				}
				abc = hd(abc);
				return abc;
			};
		};

		const Operations IDENTITY_OPS{ IDENTITY_OP };
	}
}


#endif